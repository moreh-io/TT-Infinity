# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0
"""Per-chip PUT: each chip packs its own shard, stores it, and gets it back byte for byte.

The mode's whole claim is that a chunk survives the trip unchanged while never leaving its owning
chip until it goes on the wire. A layout error here does not raise -- it moves the right number of
bytes to the wrong offset, and the model produces slightly wrong output much later -- so the checks
are byte equality of what was covered AND that nothing outside the chunk moved.

Needs EIGHT visible devices (see kvcache_manager_multiclient_setup).
"""

import pytest

from .test_kvcache_manager_helper import server_fabric_node

# One block per (member, chip). 32-byte aligned, as the ops require: a DRAM NoC read rounds its
# source address DOWN to 32 bytes, so a smaller alignment reads a shifted window instead of failing.
SHARD_BLOCK_BYTES = 2048
BLOCKS = 4
N_MEMBERS = 6  # stands in for 3 layers x (K, V)
CHUNK_BLOCKS = [1, 2]  # the blocks this chunk covers; 0 and 3 must stay untouched
_MAGIC = 0x524B564E
_HEADER_ALIGN = 32


def _shard_plan(n_members, n_blocks):
    """The per-shard blob layout: header, then each member's K/V region back to back."""
    alloc = n_blocks * SHARD_BLOCK_BYTES
    words = [_MAGIC, n_members] + [alloc] * n_members
    words += [0] * ((-len(words) * 4 % _HEADER_ALIGN) // 4)
    header_bytes = 4 * len(words)
    region_bases = [header_bytes + i * alloc for i in range(n_members)]
    return words, region_bases, header_bytes + n_members * alloc


def _make_members(model_submesh, chips, seed0=1000):
    """Distinct bytes per (member, chip) so a swap between either shows up as a mismatch."""
    import torch
    import ttnn

    words_per_member = BLOCKS * SHARD_BLOCK_BYTES // 4
    members, host = [], []
    for m in range(N_MEMBERS):
        g = torch.Generator().manual_seed(seed0 + m)
        h = torch.randint(
            0, 2**31 - 1, (chips, words_per_member), dtype=torch.int32, generator=g
        )
        host.append(h)
        members.append(
            ttnn.from_torch(
                h,
                device=model_submesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(model_submesh, dim=0),
            )
        )
    return members, host


def _read_member(dev, chip):
    import ttnn

    return ttnn.to_torch(ttnn.get_device_tensors(dev)[chip]).reshape(-1)


def _blob(model_submesh, value_len):
    import torch
    import ttnn

    words = (value_len + 3) // 4
    pages = (words + 8191) // 8192
    return ttnn.from_torch(
        torch.zeros(pages * 8192, dtype=torch.int32).reshape(pages, 8192),
        device=model_submesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model_submesh),
    )


@pytest.fixture
def perchip_setup(kvcache_manager_multiclient_setup):
    from kvcache_manager import Server

    _parent, model_submesh, server_submesh = kvcache_manager_multiclient_setup
    server = Server(server_submesh)
    try:
        yield model_submesh, server_submesh, server
    finally:
        server.close()


def test_every_chip_round_trips_its_own_shard_byte_exact(perchip_setup):
    """Pack locally, PUT, clear the cache, GET, scatter back -- and land on the original bytes."""
    import ttnn
    from ttnn import kvcache_manager
    from kvcache_manager import Client

    model_submesh, server_submesh, server = perchip_setup
    chips = int(model_submesh.shape[1])
    kvcache_manager.reserve_gather_resources(model_submesh)

    members, host = _make_members(model_submesh, chips)
    header_words, region_bases, value_len = _shard_plan(N_MEMBERS, len(CHUNK_BLOCKS))
    blob = _blob(model_submesh, value_len)
    clients = [
        Client(
            model_submesh,
            server_fabric_node(server_submesh),
            client_coord=ttnn.MeshCoordinate(0, t),
        )
        for t in range(chips)
    ]
    try:
        keys = [f"perchip-shard-{t}".encode() for t in range(chips)]
        coords = [ttnn.MeshCoordinate(0, t) for t in range(chips)]
        # One batched pack: every chip fills its own blob replica simultaneously, no fabric. This is
        # the shipping path -- per-coordinate calls serialize behind blocking enqueues.
        kvcache_manager.pack_local_shards(
            members,
            blob,
            coords,
            CHUNK_BLOCKS,
            SHARD_BLOCK_BYTES,
            region_bases,
            header_words,
        )
        for t in range(chips):
            clients[t].put(keys[t], blob, value_len, sync_commit=True)

        for t in range(chips):
            assert clients[t].exists(
                keys[t]
            ), f"chip {t}: its own shard is missing after PUT"

        originals = [
            [_read_member(m, t).clone() for m in members] for t in range(chips)
        ]

        import torch

        zeros = ttnn.from_torch(
            torch.zeros(chips, BLOCKS * SHARD_BLOCK_BYTES // 4, dtype=torch.int32),
            device=model_submesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(model_submesh, dim=0),
        )
        for m in members:
            ttnn.copy(zeros, m)
        assert (
            int(_read_member(members[0], 0).abs().sum()) == 0
        ), "the clear did not take"

        # Fetch every shard before writing any back: a late miss must not leave the cache with some
        # chips refilled and others stale.
        for t in range(chips):
            assert clients[t].get(
                keys[t], blob
            ), f"chip {t}: GET missed the shard it stored"
        kvcache_manager.unpack_local_shards(
            blob, members, coords, CHUNK_BLOCKS, SHARD_BLOCK_BYTES, region_bases
        )

        for t in range(chips):
            for mi in range(N_MEMBERS):
                now = _read_member(members[mi], t)
                for blk in CHUNK_BLOCKS:
                    lo, hi = (
                        blk * SHARD_BLOCK_BYTES // 4,
                        (blk + 1) * SHARD_BLOCK_BYTES // 4,
                    )
                    assert torch.equal(
                        originals[t][mi][lo:hi], now[lo:hi]
                    ), f"chip {t} member {mi} block {blk} came back different"
                # Blocks outside the chunk were cleared and must still be zero: a scatter that
                # spills is the failure mode a byte-equality check alone would miss.
                for blk in range(BLOCKS):
                    if blk in CHUNK_BLOCKS:
                        continue
                    lo, hi = (
                        blk * SHARD_BLOCK_BYTES // 4,
                        (blk + 1) * SHARD_BLOCK_BYTES // 4,
                    )
                    assert (
                        int(now[lo:hi].abs().sum()) == 0
                    ), f"chip {t} member {mi}: scatter spilled into block {blk}"
    finally:
        for c in clients:
            try:
                c.close()
            except (
                Exception
            ):  # noqa: BLE001 - teardown must not mask the assertion that ran
                pass
        kvcache_manager.release_gather_resources(model_submesh)


def test_a_shard_is_visible_from_every_chip(perchip_setup):
    """Shards are separate values but one directory, so every established client can see them."""
    import ttnn
    from ttnn import kvcache_manager
    from kvcache_manager import Client

    model_submesh, server_submesh, server = perchip_setup
    chips = int(model_submesh.shape[1])
    kvcache_manager.reserve_gather_resources(model_submesh)

    members, _ = _make_members(model_submesh, chips, seed0=2000)
    header_words, region_bases, value_len = _shard_plan(N_MEMBERS, len(CHUNK_BLOCKS))
    blob = _blob(model_submesh, value_len)
    clients = [
        Client(
            model_submesh,
            server_fabric_node(server_submesh),
            client_coord=ttnn.MeshCoordinate(0, t),
        )
        for t in range(chips)
    ]
    try:
        keys = [f"perchip-visible-{t}".encode() for t in range(chips)]
        coords = [ttnn.MeshCoordinate(0, t) for t in range(chips)]
        kvcache_manager.pack_local_shards(
            members,
            blob,
            coords,
            CHUNK_BLOCKS,
            SHARD_BLOCK_BYTES,
            region_bases,
            header_words,
        )
        for t in range(chips):
            clients[t].put(keys[t], blob, value_len, sync_commit=True)

        for owner in range(chips):
            for reader in range(chips):
                assert clients[reader].exists(
                    keys[owner]
                ), f"chip {reader} cannot see chip {owner}'s shard"
    finally:
        for c in clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        kvcache_manager.release_gather_resources(model_submesh)


def test_pack_rejects_unaligned_pages_and_cross_page_header(
    perchip_setup, expect_error
):
    """The kernels round transfer sizes and cannot split the value header across DRAM pages."""
    import torch
    import ttnn
    from ttnn import kvcache_manager

    model_submesh, _server_submesh, _server = perchip_setup
    chips = int(model_submesh.shape[1])
    coord = ttnn.MeshCoordinate(0, 0)

    valid_member = ttnn.from_torch(
        torch.zeros(chips, 8, dtype=torch.int32),
        device=model_submesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(model_submesh, dim=0),
    )
    invalid_member = ttnn.from_torch(
        torch.zeros(chips, 9, dtype=torch.int32),
        device=model_submesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(model_submesh, dim=0),
    )
    valid_blob = ttnn.from_torch(
        torch.zeros(64, 8, dtype=torch.int32),
        device=model_submesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model_submesh),
    )
    invalid_blob = ttnn.from_torch(
        torch.zeros(64, 9, dtype=torch.int32),
        device=model_submesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model_submesh),
    )

    with expect_error(RuntimeError, "blob page size .* 32-byte aligned"):
        kvcache_manager.pack_local_shards(
            [valid_member], invalid_blob, [coord], [0], 32, [0], []
        )
    with expect_error(RuntimeError, "member 0 page size .* 32-byte aligned"):
        kvcache_manager.pack_local_shards(
            [invalid_member], valid_blob, [coord], [0], 32, [0], []
        )
    with expect_error(RuntimeError, "header exceeds .* blob page"):
        kvcache_manager.pack_local_shards(
            [valid_member], valid_blob, [coord], [0], 32, [64], [0] * 16
        )
