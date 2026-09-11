from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kvcache_manager import Client, Server
from kvcache_manager import _facade


class RecordingNative:
    parallel_calls: list[tuple[list[RecordingNative], list[str | bytes], Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.constructor_args = args
        self.constructor_kwargs = kwargs
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.results: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return self.results.get(name)

        return record

    @staticmethod
    def get_parallel(
        clients: list[RecordingNative], keys: list[str | bytes], out_tensor: Any
    ) -> list[bool]:
        RecordingNative.parallel_calls.append((clients, keys, out_tensor))
        return [True] * len(clients)


@pytest.fixture
def native_types(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    RecordingNative.parallel_calls.clear()
    native_module = SimpleNamespace(Server=RecordingNative, Client=RecordingNative)
    monkeypatch.setattr(_facade, "_load_native_module", lambda: native_module)
    return native_module


def test_server_forwards_construction_and_public_methods(
    native_types: SimpleNamespace,
) -> None:
    submesh = object()
    server = Server(submesh, True, 7)
    native = server._native

    assert native.constructor_args == (submesh, True, 7, 0)

    calls = [
        ("is_ready", (), True),
        ("moves_completed", (), 4),
        ("move_profile", (), {"count": 1}),
        ("allocator_profile", (), {"allocation_calls": 3}),
    ]
    for method_name, args, result in calls:
        native.results[method_name] = result
        assert getattr(server, method_name)(*args) == result

    assert native.calls == [(name, args, {}) for name, args, _ in calls]


def test_server_context_manager_closes_native(native_types: SimpleNamespace) -> None:
    server = Server(object())

    with server as entered:
        assert entered is server

    assert server._native.calls == [("close", (), {})]


def test_client_preserves_server_node(native_types: SimpleNamespace) -> None:
    server_node = object()

    client = Client("client", server_node)

    assert client._native.constructor_args == (
        "client",
        server_node,
        None,
        False,
        0,
        0,
    )


def test_client_validates_connection_mode(
    native_types: SimpleNamespace,
) -> None:
    server_node = object()

    with pytest.raises(ValueError, match="requires server_node"):
        Client("client")
    with pytest.raises(ValueError, match="does not take server_node"):
        Client("client", server_node, static_intermesh_galaxy_to_t3k=True)

    client = Client("client", static_intermesh_galaxy_to_t3k=True)
    assert client._native.constructor_args == ("client", None, None, True, 0, 0)


def test_client_forwards_public_methods(native_types: SimpleNamespace) -> None:
    client = Client("client", object())
    native = client._native
    tensor = object()
    tensors = [object(), object()]

    calls = [
        ("is_established", (), True),
        ("ping", (b"ping",), b"ping"),
        ("put", ("key", tensor, 64, True), None),
        ("put_batch", (["a", b"b"], tensors, [32, 64], True), None),
        ("put_profile", (), {"count": 2}),
        ("flush", (), None),
        ("get", ("key", tensor), True),
        ("exists", (b"key",), False),
        ("delete", ("key",), True),
        ("move_to_host", ("key",), None),
        ("move_to_device", (b"key",), None),
    ]
    for method_name, args, result in calls:
        native.results[method_name] = result
        assert getattr(client, method_name)(*args) == result

    assert native.calls == [(name, args, {}) for name, args, _ in calls]


def test_client_context_manager_closes_native(native_types: SimpleNamespace) -> None:
    client = Client("client", object())

    with client as entered:
        assert entered is client

    assert client._native.calls == [("close", (), {})]


def test_client_parallel_get_forwards_native_clients(
    native_types: SimpleNamespace,
) -> None:
    first = Client("first", object())
    second = Client("second", object())
    tensor = object()

    assert Client.get_parallel([first, second], ["a", b"b"], tensor) == [
        True,
        True,
    ]
    assert RecordingNative.parallel_calls == [
        ([first._native, second._native], ["a", b"b"], tensor)
    ]


def test_test_only_native_methods_are_not_exposed(
    native_types: SimpleNamespace,
) -> None:
    server = Server(object())
    client = Client("client", object())

    for name in (
        "flush",
        "pool_stats",
        "get",
        "exists",
        "delete",
        "make_descriptor",
        "_test_create",
        "_cache_state",
    ):
        assert not hasattr(server, name)
    assert not hasattr(client, "_test_create")
    assert not hasattr(client, "_test_client_id")


def test_native_loader_imports_ttnn_before_manager_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    native_module = object()

    def import_recording(name: str) -> Any:
        calls.append(name)
        return native_module if name == _facade._NATIVE_MODULE_NAME else object()

    monkeypatch.setattr(_facade, "import_module", import_recording)

    assert _facade._load_native_module() is native_module
    assert calls == ["ttnn", "kvcache_manager._native"]


def test_missing_native_runtime_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(_facade, "import_module", missing_module)

    with pytest.raises(ImportError, match="requires its native extension"):
        _facade._load_native_module()
