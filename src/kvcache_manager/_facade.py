"""Stable product facades over the manager-owned native runtime."""

from __future__ import annotations

from importlib import import_module
from types import TracebackType
from typing import Any


_NATIVE_MODULE_NAME = "kvcache_manager._native"


def _load_native_module() -> Any:
    try:
        import_module("ttnn")
        return import_module(_NATIVE_MODULE_NAME)
    except ImportError as error:
        raise ImportError(
            "kvcache-manager requires its native extension and a matching "
            "tt-metal installation"
        ) from error


def _native_class(name: str) -> type[Any]:
    native_module = _load_native_module()
    try:
        return getattr(native_module, name)
    except AttributeError as error:
        raise ImportError(
            f"kvcache-manager native bindings do not provide {name}"
        ) from error


class Server:
    """Own a native KV cache server."""

    def __init__(
        self,
        server_submesh: Any,
        static_intermesh_t3k_server: bool = False,
        static_intermesh_remote_mesh_id: int = 0,
        worker_group: int = 0,
    ) -> None:
        self._native = _native_class("Server")(
            server_submesh,
            static_intermesh_t3k_server,
            static_intermesh_remote_mesh_id,
            worker_group,
        )

    def is_ready(self) -> bool:
        return self._native.is_ready()

    def close(self) -> None:
        self._native.close()

    def moves_completed(self) -> int:
        return self._native.moves_completed()

    def move_profile(self) -> dict[str, Any]:
        return self._native.move_profile()

    def allocator_profile(self) -> dict[str, Any]:
        return self._native.allocator_profile()

    def __enter__(self) -> Server:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class Client:
    """Own a native TTNN KV cache client session."""

    def __init__(
        self,
        client_submesh: Any,
        server_node: Any | None = None,
        client_coord: Any | None = None,
        static_intermesh_galaxy_to_t3k: bool = False,
        server_worker_group: int = 0,
        static_intermesh_lane: int = 0,
    ) -> None:
        if static_intermesh_galaxy_to_t3k and server_node is not None:
            raise ValueError("static Galaxy-to-T3K mode does not take server_node")
        if not static_intermesh_galaxy_to_t3k and server_node is None:
            raise ValueError("standard Fabric mode requires server_node")

        self._native = _native_class("Client")(
            client_submesh,
            server_node,
            client_coord,
            static_intermesh_galaxy_to_t3k,
            server_worker_group,
            static_intermesh_lane,
        )

    def is_established(self) -> bool:
        return self._native.is_established()

    def close(self) -> None:
        self._native.close()

    def ping(self, payload: bytes) -> bytes:
        return self._native.ping(payload)

    def put(
        self,
        key: str | bytes,
        value_tensor: Any,
        value_len_bytes: int,
        sync_commit: bool = False,
    ) -> None:
        self._native.put(key, value_tensor, value_len_bytes, sync_commit)

    def put_batch(
        self,
        keys: list[str | bytes],
        value_tensors: list[Any],
        value_len_bytes: list[int],
        sync_commit: bool = False,
    ) -> None:
        self._native.put_batch(keys, value_tensors, value_len_bytes, sync_commit)

    def put_profile(self) -> dict[str, Any]:
        return self._native.put_profile()

    def flush(self) -> None:
        self._native.flush()

    def get(self, key: str | bytes, out_tensor: Any) -> bool:
        return self._native.get(key, out_tensor)

    @staticmethod
    def get_parallel(
        clients: list[Client], keys: list[str | bytes], out_tensor: Any
    ) -> list[bool]:
        native_clients = [client._native for client in clients]
        return list(
            _native_class("Client").get_parallel(native_clients, keys, out_tensor)
        )

    def exists(self, key: str | bytes) -> bool:
        return self._native.exists(key)

    def delete(self, key: str | bytes) -> bool:
        return self._native.delete(key)

    def move_to_host(self, key: str | bytes) -> None:
        self._native.move_to_host(key)

    def move_to_device(self, key: str | bytes) -> None:
        self._native.move_to_device(key)

    def __enter__(self) -> Client:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
