# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for kvcache-manager device tests.

Provides `kvcache_manager_mesh_and_submesh` for the kvcache_manager
lifecycle suite, and `kvcache_manager_client_setup` for the
Client/Server suite.

The kvcache_manager submesh-only restructure: caller now owns a parent
mesh and two (1,1) submeshes (client_submesh + server_submesh) — there is
no SubDevice manager involved. Each submesh has its own L1 allocator and
mesh_command_queue.
"""

from pathlib import Path

import pytest


@pytest.fixture
def expect_error():
    """Require the expected device error and its identifying message."""

    def expect_error_(error, message):
        return pytest.raises(error, match=message)

    return expect_error_


def pytest_collection_modifyitems(items):
    device = pytest.mark.device
    device_test_root = Path(__file__).parent.resolve()
    for item in items:
        if item.path.resolve().is_relative_to(device_test_root):
            item.add_marker(device)


def _require_visible_devices(fixture_name: str, needed: int) -> None:
    """Fail BEFORE opening the mesh when TT_VISIBLE_DEVICES cannot satisfy the fixture.

    A mesh-shape/device-count mismatch otherwise surfaces as a fabric router handshake
    timeout, which reads like a hardware fault rather than a configuration one.
    An unset TT_VISIBLE_DEVICES exposes the whole machine and is fine.
    """
    import os

    import pytest

    visible = os.environ.get("TT_VISIBLE_DEVICES", "").strip()
    if not visible:
        return
    have = len([d for d in visible.split(",") if d.strip()])
    if have != needed:
        pytest.fail(
            f"{fixture_name} opens a mesh of exactly {needed} chips but TT_VISIBLE_DEVICES "
            f"names {have}. Fix the device list; opening anyway would hang in the fabric "
            f"router handshake and masquerade as a hardware fault.",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def kvcache_manager_mesh_and_submesh():
    """Open a 1x1 parent mesh with FABRIC_1D, create a (1,1) server_submesh,
    yield `(parent_mesh, server_submesh)`. No SubDevice manager.

    Lifecycle tests use only Server (no Client), and a 1x1 parent has no
    reachable peers — the server kernel runs in idle mode (num_connections=0).
    """
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    try:
        _require_visible_devices("kvcache_manager_mesh_and_submesh", 1)
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        server_submesh = parent_mesh.create_submesh(
            ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0)
        )
        try:
            yield parent_mesh, server_submesh
        finally:
            parent_mesh.quiesce_devices()
            ttnn.close_mesh_device(server_submesh)
            ttnn.close_mesh_device(parent_mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _open_client_server_mesh(fixture_name: str, fabric: str):
    """Yield the common one-Client/one-Server topology under the selected fabric."""
    import ttnn

    fabric_config = {
        "1d": ttnn.FabricConfig.FABRIC_1D,
        "2d": ttnn.FabricConfig.FABRIC_2D,
    }[fabric]
    _require_visible_devices(fixture_name, 2)
    ttnn.set_fabric_config(fabric_config)
    try:
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 1))
        client_submesh = parent_mesh.create_submesh(
            ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0)
        )
        server_submesh = parent_mesh.create_submesh(
            ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, 0)
        )
        try:
            yield parent_mesh, client_submesh, server_submesh
        finally:
            parent_mesh.quiesce_devices()
            ttnn.close_mesh_device(client_submesh)
            ttnn.close_mesh_device(server_submesh)
            ttnn.close_mesh_device(parent_mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def kvcache_manager_client_setup():
    """Open the standard 2x1 direct-neighbor Client/Server topology under FABRIC_1D.

    Caller invariant: any L1 tensor on the client chip must be allocated via
    client_submesh; any L1 tensor on the server chip must be allocated via
    server_submesh. DRAM allocation is unaffected.
    """
    yield from _open_client_server_mesh("kvcache_manager_client_setup", "1d")


@pytest.fixture(scope="module")
def kvcache_manager_multiclient_setup():
    """A model mesh whose chips can each establish with one Server, plus that Server's chip.

    Opens a (2,4) parent under FABRIC_2D and yields
    ``(parent_mesh, model_submesh, server_submesh)`` with the model on the (1,4) row at (1,0).
    The Server uses an MMIO-capable coordinate selected from the unused top row, exercising the
    production GET completion path as well as absolute 2D routing.

    Needs EIGHT visible devices. The other kvcache fixtures open (1,1) or (2,1), and a mismatch
    between the fixture's mesh shape and TT_VISIBLE_DEVICES surfaces as a fabric router handshake
    timeout, which reads like a hardware fault rather than a configuration one.

    FABRIC_2D rather than 1D because 1D addresses a peer by hop count pinned to one, so every client
    chip would have to be a direct neighbour of the Server; 2D routes by absolute chip id and lets
    the row establish with a Server that is not adjacent to all of it.
    """
    import ttnn

    _require_visible_devices("kvcache_manager_multiclient_setup", 8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    try:
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
        model_submesh = None
        server_submesh = None
        try:
            model_submesh = parent_mesh.create_submesh(
                ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(1, 0)
            )
            parent_device_ids = [
                int(device_id) for device_id in parent_mesh.get_device_ids()
            ]
            pcie_device_ids = {
                int(device_id) for device_id in ttnn.get_pcie_device_ids()
            }
            server_columns = [
                column
                for column in range(4)
                if parent_device_ids[column] in pcie_device_ids
            ]
            if not server_columns:
                pytest.fail("the unused top row has no MMIO-capable Server coordinate")
            server_submesh = parent_mesh.create_submesh(
                ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, server_columns[0])
            )
            yield parent_mesh, model_submesh, server_submesh
        finally:
            parent_mesh.quiesce_devices()
            if model_submesh is not None:
                ttnn.close_mesh_device(model_submesh)
            if server_submesh is not None:
                ttnn.close_mesh_device(server_submesh)
            ttnn.close_mesh_device(parent_mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="module")
def kvcache_manager_gather_2d_setup():
    """A 2x3 model mesh plus a Server outside its top-row TP owner group.

    The model occupies ``(2,3)@(0,0)`` in a ``(2,4)`` FABRIC_2D parent and the Server occupies
    ``(1,1)@(0,3)``. Tests can gather the top-row TP shards onto model-local ``(1,2)``, proving
    that the selected wire chip need not own one of the gathered shards.
    """
    import ttnn

    _require_visible_devices("kvcache_manager_gather_2d_setup", 8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    try:
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
        model_submesh = parent_mesh.create_submesh(
            ttnn.MeshShape(2, 3), ttnn.MeshCoordinate(0, 0)
        )
        server_submesh = parent_mesh.create_submesh(
            ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 3)
        )
        try:
            yield parent_mesh, model_submesh, server_submesh
        finally:
            parent_mesh.quiesce_devices()
            ttnn.close_mesh_device(model_submesh)
            ttnn.close_mesh_device(server_submesh)
            ttnn.close_mesh_device(parent_mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


_IRAM_PROCESS_CLAIM = None


def pytest_runtest_setup(item):
    """Each IRAM-budget test owns its pytest process.

    The measurement parses the ELF the JIT just built, and the in-process program cache
    happily reuses a kernel the OTHER test compiled -- the number then describes the wrong
    variant while looking plausible. Enforced here (before fixtures) rather than in the test
    body so a fixture failure cannot mask the violation.
    """
    global _IRAM_PROCESS_CLAIM
    if "test_kvcache_manager_iram_budget" not in str(item.fspath):
        return
    if _IRAM_PROCESS_CLAIM is not None and _IRAM_PROCESS_CLAIM != item.name:
        import pytest

        pytest.fail(
            f"{item.name} must run in its own pytest process, but {_IRAM_PROCESS_CLAIM} "
            "already compiled kernels here. Invoke each IRAM test as a separate "
            "`pytest <file>::<test>` command.",
            pytrace=False,
        )
    _IRAM_PROCESS_CLAIM = item.name
