# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0

"""Device regression coverage for PUT-to-EXISTS ordering."""

import os
import secrets

import pytest

from .test_kvcache_manager_helper import make_dram_tensor, server_fabric_node


def _require_full_t3k() -> None:
    """Reject a partial TT_VISIBLE_DEVICES selection before opening the mesh."""
    visible = os.environ.get("TT_VISIBLE_DEVICES", "").strip()
    if not visible:
        return

    visible_devices = [device for device in visible.split(",") if device.strip()]
    if len(visible_devices) != 8:
        pytest.fail(
            "the EXISTS ordering test requires exactly 8 visible devices",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def kvcache_manager_full_mesh_client_setup():
    """Keep the full T3K active while selecting one adjacent Client/Server pair."""
    import ttnn

    _require_full_t3k()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    try:
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
        client_submesh = None
        server_submesh = None
        try:
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
                pytest.fail("the top row has no MMIO-capable Server coordinate")
            server_column = server_columns[0]
            client_submesh = parent_mesh.create_submesh(
                ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, server_column)
            )
            server_submesh = parent_mesh.create_submesh(
                ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, server_column)
            )
            yield parent_mesh, client_submesh, server_submesh
        finally:
            parent_mesh.quiesce_devices()
            if client_submesh is not None:
                ttnn.close_mesh_device(client_submesh)
            if server_submesh is not None:
                ttnn.close_mesh_device(server_submesh)
            ttnn.close_mesh_device(parent_mesh)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_exists_waits_for_delayed_async_put_commit(
    kvcache_manager_full_mesh_client_setup,
):
    """EXISTS returns only after the preceding asynchronous PUT is committed."""
    from kvcache_manager import Client
    from kvcache_manager import _native as native

    _parent_mesh, client_submesh, server_submesh = (
        kvcache_manager_full_mesh_client_setup
    )
    server = native.Server._test_create_delay_first_put_commit(server_submesh)
    client = None
    try:
        client = Client(client_submesh, server_fabric_node(server_submesh))
        assert client.exists(b"compile-exists-before-delayed-put") is False

        key = b"delayed-commit"
        value = secrets.token_bytes(256)
        value_tensor, value_len = make_dram_tensor(client_submesh, value)
        client.put(key, value_tensor, value_len, sync_commit=False)

        issued, committed = server._test_put_drain_progress()
        assert issued != committed, "the test hook did not expose an in-flight PUT"
        assert client.exists(key) is True
        assert server._test_put_drain_progress() == (issued, issued)
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            server.close()
