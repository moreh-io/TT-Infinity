# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0

"""KV cache one-hop fabric topology invariants on a full T3K mesh."""

import gc

import pytest

from .test_kvcache_manager_helper import server_fabric_node


@pytest.fixture(scope="module")
def kvcache_manager_topology_mesh():
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    try:
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
        try:
            yield parent_mesh
        finally:
            parent_mesh.quiesce_devices()
            ttnn.close_mesh_device(parent_mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _coordinates(ttnn):
    return {
        (row, col): ttnn.MeshCoordinate(row, col)
        for row in range(2)
        for col in range(4)
    }


def _orthogonal_pairs(ttnn):
    coords = _coordinates(ttnn)
    pairs = []
    for row in range(2):
        for col in range(4):
            if row + 1 < 2:
                pairs.append((coords[(row, col)], coords[(row + 1, col)]))
            if col + 1 < 4:
                pairs.append((coords[(row, col)], coords[(row, col + 1)]))
    return pairs


def _chip_id(parent_mesh, coord):
    return int(parent_mesh.get_fabric_node_id(coord).chip_id)


def _allocated_bytes(submesh):
    import ttnn

    return {
        "l1": ttnn.get_memory_view(
            submesh, ttnn.BufferType.L1
        ).total_bytes_allocated_per_bank,
        "dram": ttnn.get_memory_view(
            submesh, ttnn.BufferType.DRAM
        ).total_bytes_allocated_per_bank,
    }


def test_direct_neighbor_with_gapped_chip_ids_round_trips(
    kvcache_manager_topology_mesh,
):
    """A direct physical link is one hop even when its fabric chip IDs are non-consecutive."""
    import ttnn
    from kvcache_manager import Client, Server

    parent_mesh = kvcache_manager_topology_mesh
    client_coord, server_coord = max(
        _orthogonal_pairs(ttnn),
        key=lambda pair: abs(
            _chip_id(parent_mesh, pair[0]) - _chip_id(parent_mesh, pair[1])
        ),
    )
    chip_gap = abs(
        _chip_id(parent_mesh, client_coord) - _chip_id(parent_mesh, server_coord)
    )
    assert (
        chip_gap > 1
    ), f"T3K mapping has no non-consecutive direct pair; largest chip-id gap was {chip_gap}"

    client_submesh = parent_mesh.create_submesh(ttnn.MeshShape(1, 1), client_coord)
    server_submesh = parent_mesh.create_submesh(ttnn.MeshShape(1, 1), server_coord)
    try:
        with (
            Server(server_submesh) as server,
            Client(client_submesh, server_fabric_node(server_submesh)) as client,
        ):
            payload = b"gapped-direct-neighbor"
            assert client.ping(payload) == payload
    finally:
        parent_mesh.quiesce_devices()
        del client_submesh
        del server_submesh
        gc.collect()


def test_multihop_pair_fails_before_client_allocation(
    kvcache_manager_topology_mesh, expect_error
):
    """Opposite corners of the 2x4 T3K mesh are multiple hops apart, which a 1D wire cannot route.

    This fixture runs under FABRIC_1D, where the route word is a hop count pinned to one. On 2D the
    same pairing is legal, because the route word carries the absolute destination.
    """
    import ttnn
    from kvcache_manager import Client, Server

    parent_mesh = kvcache_manager_topology_mesh
    client_coord = ttnn.MeshCoordinate(0, 0)
    server_coord = ttnn.MeshCoordinate(1, 3)
    client_submesh = parent_mesh.create_submesh(ttnn.MeshShape(1, 1), client_coord)
    server_submesh = parent_mesh.create_submesh(ttnn.MeshShape(1, 1), server_coord)
    try:
        baseline = _allocated_bytes(client_submesh)
        with Server(server_submesh) as server:
            with expect_error(RuntimeError, "are not Fabric-reachable"):
                Client(client_submesh, server_fabric_node(server_submesh))
            assert _allocated_bytes(client_submesh) == baseline
    finally:
        parent_mesh.quiesce_devices()
        del client_submesh
        del server_submesh
        gc.collect()
