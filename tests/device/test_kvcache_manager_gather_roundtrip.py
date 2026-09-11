# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0
"""Gather-to-one-chip wire round trip over the 2D KV cache topology.

The selected model chip gathers every TP shard, performs the wire PUT/GET, then scatters the
returned blob back to the owning chips. The gathered blob is checked independently before the
round trip so matching gather/scatter mistakes cannot cancel each other out.

Needs EIGHT visible devices (see kvcache_manager_multiclient_setup).
"""

import pytest

from .test_kvcache_manager_helper import server_fabric_node

SHARD_BLOCK_BYTES = 2048
BLOCKS = 4
N_MEMBERS = 6
CHUNK_BLOCKS = [1, 2]
_MAGIC = 0x524B564E
_HEADER_ALIGN = 32


def _gathered_plan(n_members: int, n_blocks: int, chips: int):
    block_bytes = chips * SHARD_BLOCK_BYTES
    allocation_bytes = n_blocks * block_bytes
    words = [_MAGIC, n_members] + [allocation_bytes] * n_members
    words += [0] * ((-len(words) * 4 % _HEADER_ALIGN) // 4)
    header_bytes = 4 * len(words)
    region_bases = [
        header_bytes + member * allocation_bytes for member in range(n_members)
    ]
    return words, region_bases, header_bytes + n_members * allocation_bytes


def _make_members(model_submesh, chips: int):
    import torch
    import ttnn

    words_per_member = BLOCKS * SHARD_BLOCK_BYTES // 4
    members, host = [], []
    for member in range(N_MEMBERS):
        generator = torch.Generator().manual_seed(3000 + member)
        values = torch.randint(
            0,
            2**31 - 1,
            (chips, words_per_member),
            dtype=torch.int32,
            generator=generator,
        )
        host.append(values)
        members.append(
            ttnn.from_torch(
                values,
                device=model_submesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(model_submesh, dim=0),
            )
        )
    return members, host


def _replicated_blob(model_submesh, value_len: int):
    import torch
    import ttnn

    words = (value_len + 3) // 4
    pages = (words + 8191) // 8192
    return ttnn.from_torch(
        torch.zeros(pages, 8192, dtype=torch.int32),
        device=model_submesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model_submesh),
    )


def _expected_gathered_bytes(host, header_words, rank_indices) -> bytes:
    import torch

    expected = bytearray(
        torch.tensor(header_words, dtype=torch.int32).numpy().tobytes()
    )
    for member in host:
        member_bytes = (
            member.contiguous().view(torch.uint8).reshape(member.shape[0], -1)
        )
        for block in CHUNK_BLOCKS:
            begin = block * SHARD_BLOCK_BYTES
            end = begin + SHARD_BLOCK_BYTES
            for rank_index in rank_indices:
                expected.extend(member_bytes[rank_index, begin:end].tolist())
    return bytes(expected)


def _read_shard_bytes(tensor, chip: int, value_len: int) -> bytes:
    import torch
    import ttnn

    shard = ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip]).reshape(-1)
    return bytes(shard.contiguous().view(torch.uint8)[:value_len].tolist())


def _read_member(tensor, chip: int):
    import ttnn

    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip]).reshape(-1)


@pytest.fixture
def gather_server(kvcache_manager_gather_2d_setup):
    from kvcache_manager import Server

    _parent, model_submesh, server_submesh = kvcache_manager_gather_2d_setup
    server = Server(server_submesh)
    try:
        yield model_submesh, server_submesh, server
    finally:
        server.close()


def test_selected_chip_gathers_puts_gets_and_scatters_byte_exact(gather_server):
    import torch
    import ttnn
    from ttnn import kvcache_manager
    from kvcache_manager import Client

    model_submesh, server_submesh, server = gather_server
    chips = int(model_submesh.get_num_devices())
    rank_indices = [0, 1, 2]
    master_chip = 5
    master_coord = ttnn.MeshCoordinate(1, 2)
    rank_coords = [ttnn.MeshCoordinate(0, rank) for rank in rank_indices]
    kvcache_manager.reserve_gather_resources(model_submesh)

    members, host = _make_members(model_submesh, chips)
    header_words, region_bases, value_len = _gathered_plan(
        N_MEMBERS, len(CHUNK_BLOCKS), len(rank_indices)
    )
    blob = _replicated_blob(model_submesh, value_len)
    expected = _expected_gathered_bytes(host, header_words, rank_indices)
    client = Client(
        model_submesh,
        server_fabric_node(server_submesh),
        client_coord=master_coord,
    )
    try:
        kvcache_manager.gather_tp_shards_to_chip(
            members,
            blob,
            master_coord,
            rank_coords,
            CHUNK_BLOCKS,
            SHARD_BLOCK_BYTES,
            region_bases,
            header_words,
        )
        assert _read_shard_bytes(blob, master_chip, value_len) == expected
        client.put(b"gather-one-chip-roundtrip", blob, value_len, sync_commit=True)

        empty_blob = _replicated_blob(model_submesh, value_len)
        ttnn.copy(empty_blob, blob)
        assert _read_shard_bytes(blob, master_chip, value_len) == bytes(value_len)

        words_per_member = BLOCKS * SHARD_BLOCK_BYTES // 4
        zeros = ttnn.from_torch(
            torch.zeros(chips, words_per_member, dtype=torch.int32),
            device=model_submesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(model_submesh, dim=0),
        )
        for member in members:
            ttnn.copy(zeros, member)
        assert (
            int(_read_member(members[0], 0).abs().sum()) == 0
        ), "the cache clear did not take"

        assert client.get(b"gather-one-chip-roundtrip", blob)
        assert _read_shard_bytes(blob, master_chip, value_len) == expected
        kvcache_manager.scatter_chunk_from_chip(
            blob,
            members,
            master_coord,
            rank_coords,
            CHUNK_BLOCKS,
            SHARD_BLOCK_BYTES,
            region_bases,
        )

        for rank in range(chips):
            for member_index, member in enumerate(members):
                restored = _read_member(member, rank)
                for block in CHUNK_BLOCKS:
                    begin = block * SHARD_BLOCK_BYTES // 4
                    end = (block + 1) * SHARD_BLOCK_BYTES // 4
                    if rank in rank_indices:
                        assert torch.equal(
                            host[member_index][rank, begin:end], restored[begin:end]
                        ), f"chip {rank} member {member_index} block {block} came back different"
                    else:
                        assert (
                            int(restored[begin:end].abs().sum()) == 0
                        ), f"non-owner chip {rank} member {member_index} block {block} was modified"
                for block in range(BLOCKS):
                    if block in CHUNK_BLOCKS:
                        continue
                    begin = block * SHARD_BLOCK_BYTES // 4
                    end = (block + 1) * SHARD_BLOCK_BYTES // 4
                    assert (
                        int(restored[begin:end].abs().sum()) == 0
                    ), f"chip {rank} member {member_index}: scatter spilled into block {block}"
    finally:
        try:
            client.close()
        finally:
            kvcache_manager.release_gather_resources(model_submesh)
