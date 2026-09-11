# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0

"""EXISTS op tests for ttnn.kvcache_manager.

Client.exists(key) issues a single-packet wire request and blocks for a
single-packet status response from the Server's persistent device kernels.
No value bytes are transferred.

Client kernel op_kind:
    2 (kOpKindExists): wait on kClientRecvSemOffset
"""

import secrets

from .test_kvcache_manager_helper import make_dram_tensor, server_fabric_node


def test_exists_returns_false_for_missing_key(kvcache_manager_client_setup):
    """EXISTS on a key never inserted: returns False."""
    from kvcache_manager import Server, Client

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup

    with (
        Server(server_submesh) as server,
        Client(client_submesh, server_fabric_node(server_submesh)) as client,
    ):
        assert client.exists(b"never-inserted") is False


def test_exists_returns_true_after_put(kvcache_manager_client_setup):
    """PUT then EXISTS returns True for the inserted key."""
    from kvcache_manager import Server, Client

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup

    with (
        Server(server_submesh) as server,
        Client(client_submesh, server_fabric_node(server_submesh)) as client,
    ):
        key = b"present"
        value = secrets.token_bytes(256)
        v_t, n = make_dram_tensor(client_submesh, value)
        client.put(key, v_t, n)
        assert client.exists(key) is True
        assert client.exists(b"other") is False


def test_exists_accepts_str_and_bytes_key(kvcache_manager_client_setup):
    """to_std_string accepts both bytes and str."""
    from kvcache_manager import Server, Client

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup

    with (
        Server(server_submesh) as server,
        Client(client_submesh, server_fabric_node(server_submesh)) as client,
    ):
        value = secrets.token_bytes(16)
        v_t, n = make_dram_tensor(client_submesh, value)
        client.put("k_str", v_t, n)
        assert client.exists("k_str") is True
        assert client.exists(b"k_str") is True


def test_exists_key_size_violation(kvcache_manager_client_setup, expect_error):
    """key bigger than kMaxPutKeyBytes (128) is rejected client-side."""
    from kvcache_manager import Server, Client

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup

    with (
        Server(server_submesh) as server,
        Client(client_submesh, server_fabric_node(server_submesh)) as client,
    ):
        with expect_error(RuntimeError, "exceeds Server max_key_bytes"):
            client.exists(b"x" * 129)


def test_exists_empty_key_rejected(kvcache_manager_client_setup, expect_error):
    """Empty key is rejected (key_len must be > 0)."""
    from kvcache_manager import Server, Client

    _parent_mesh, client_submesh, server_submesh = kvcache_manager_client_setup

    with (
        Server(server_submesh) as server,
        Client(client_submesh, server_fabric_node(server_submesh)) as client,
    ):
        with expect_error(RuntimeError, "key must be non-empty"):
            client.exists(b"")
