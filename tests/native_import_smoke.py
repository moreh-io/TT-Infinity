from __future__ import annotations

from importlib import import_module
from pathlib import Path


def main() -> None:
    import ttnn

    from kvcache_manager import Client, Server

    native = import_module("kvcache_manager._native")
    ttnn_native = import_module("ttnn._ttnn.operations.kvcache_manager")

    assert Path(native.__file__).name.startswith("_native")
    assert not hasattr(ttnn_native, "Server")
    assert not hasattr(ttnn_native, "Client")
    assert not hasattr(ttnn.kvcache_manager, "Server")
    assert not hasattr(ttnn.kvcache_manager, "Client")
    assert Server.__module__ == "kvcache_manager._facade"
    assert Client.__module__ == "kvcache_manager._facade"

    assert {name for name in dir(native.Server) if not name.startswith("_")} == {
        "is_ready",
        "close",
        "moves_completed",
        "move_profile",
        "allocator_profile",
    }
    assert {name for name in dir(native.Client) if not name.startswith("_")} == {
        "is_established",
        "close",
        "ping",
        "put",
        "put_batch",
        "put_profile",
        "flush",
        "get",
        "get_parallel",
        "exists",
        "delete",
        "move_to_host",
        "move_to_device",
    }

    for name in ("make_descriptor", "_test_create", "_cache_state"):
        assert not hasattr(native.Server, name)
    for name in ("_test_create", "_test_client_id", "_test_registry_holds_submesh"):
        assert not hasattr(native.Client, name)

    assert ttnn is not None


if __name__ == "__main__":
    main()
