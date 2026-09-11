# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0

"""Client/Server ping echo for ttnn.kvcache_manager.

Builds a 2x1 parent mesh with FABRIC_1D + two (1,1) submeshes, brings up a
persistent server kernel as a Server on the server_submesh (chip (1,0)) and a
Client on the client_submesh (chip (0,0)). The Client ships a PING packet
over fabric; the Server's server kernel echoes it back. The test asserts
byte-equal round-trip.
"""

import secrets

import pytest

from .test_kvcache_manager_helper import server_fabric_node


def test_ping_round_trip(kvcache_manager_client_setup):
    """End-to-end ping at the max ping payload size (1024 bytes)."""
    from kvcache_manager import Server, Client

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup

    with (
        Server(server_submesh) as server,
        Client(client_submesh, server_fabric_node(server_submesh)) as client,
    ):
        assert server.is_ready()
        assert client.is_established()
        payload = secrets.token_bytes(
            1024
        )  # kMaxPingPayloadBytes (RT-args bound, not the wire MTU)
        echoed = client.ping(payload)
        assert echoed == payload


@pytest.mark.parametrize("size", [0, 1, 16, 1024])
def test_ping_various_sizes(kvcache_manager_client_setup, size):
    """Round-trip a range of payload sizes through the same Client/Server pair."""
    from kvcache_manager import Server, Client

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup

    with (
        Server(server_submesh) as server,
        Client(client_submesh, server_fabric_node(server_submesh)) as client,
    ):
        assert server.is_ready()
        assert client.is_established()
        payload = secrets.token_bytes(size)
        echoed = client.ping(payload)
        assert echoed == payload


def test_ping_rejects_oversized_payload_without_state_corruption(
    kvcache_manager_client_setup, expect_error
):
    """A 1025-byte payload fails before enqueue and a following PING still succeeds."""
    from kvcache_manager import Server, Client

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup

    with (
        Server(server_submesh) as server,
        Client(client_submesh, server_fabric_node(server_submesh)) as client,
    ):
        with expect_error(
            RuntimeError, "ping payload size 1025 exceeds kMaxPingPayloadBytes 1024"
        ):
            client.ping(b"x" * 1025)

        assert client.is_established()
        assert client.ping(b"after-oversized-ping") == b"after-oversized-ping"


def test_duplicate_client_establishment_fails(kvcache_manager_client_setup, expect_error):
    """A second live Client is rejected, while DETACH permits later establishment."""
    from kvcache_manager import Server, Client

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup

    with Server(server_submesh) as server:
        assert server.is_ready()
        server_node = server_fabric_node(server_submesh)
        client = Client(client_submesh, server_node)
        with expect_error(RuntimeError, "A Client is already active on mesh"):
            _client2 = Client(client_submesh, server_node)
        client.close()


@pytest.mark.parametrize("count", [1, 8, 65])
def test_ping_repeated(kvcache_manager_client_setup, count):
    """Verify ring-buffer wrap-around stays consistent across N > kRingNumSlots
    consecutive pings. count=65 exceeds kRingNumSlots=64 by one to force at
    least one wrap. Each iteration uses a fresh random payload of varying
    size so a mismatch surfaces a desync between Client's `next_ring_slot_idx_`
    and the server kernel's `cursor`.
    """
    from kvcache_manager import Server, Client

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup

    with (
        Server(server_submesh) as server,
        Client(client_submesh, server_fabric_node(server_submesh)) as client,
    ):
        assert server.is_ready()
        assert client.is_established()
        for i in range(count):
            # Vary size each iteration so any slot-reuse corruption surfaces
            # at a different offset; cap at kMaxPingPayloadBytes.
            payload = secrets.token_bytes(((i * 257) % 1025))
            echoed = client.ping(payload)
            assert echoed == payload, f"mismatch at iter {i} (size={len(payload)})"
