# SPDX-FileCopyrightText: (c) 2026 Moreh
# SPDX-License-Identifier: Apache-2.0

"""Manual product-Python smoke for intranode and static inter-mesh KVM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import torch
import ttnn

from kvcache_manager import Client, Server


_CHUNK_BYTES = 4288
_GALAXY_ENDPOINT_CHIP = 24
_T3K_ENDPOINT_CHIP = 1
_DEFAULT_BENCHMARK_SIZES = (
    1024,
    4096,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PROC_SELF_MAPS = Path("/proc/self/maps")
_IMPLEMENTATION_LIBRARY_NAMES = (
    "libkvcache_manager.so",
    "_ttnncpp.so",
    "libtt_metal.so",
)


def _create_intranode_server(
    server_submesh,
    *,
    delay_first_response: bool,
    drop_first_response: bool,
):
    if not delay_first_response and not drop_first_response:
        return Server(server_submesh)
    native = importlib.import_module("kvcache_manager._native")
    if delay_first_response:
        return native.Server._test_create_delay_first_establish_response(
            server_submesh
        )
    return native.Server._test_create_drop_first_establish_response(
        server_submesh
    )


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(
            "timeout must be a finite positive number of seconds"
        )
    return seconds


def _positive_int(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return integer


def _non_negative_int(value: str) -> int:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return integer


def _static_link_index(value: str) -> int:
    index = int(value)
    if not 0 <= index <= 3:
        raise argparse.ArgumentTypeError("link index must be between 0 and 3")
    return index


def _benchmark_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(_positive_int(part.strip()) for part in value.split(","))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError(
            "benchmark sizes must be comma-separated positive integers"
        ) from error
    if not sizes or len(sizes) != len(set(sizes)):
        raise argparse.ArgumentTypeError("benchmark sizes must be non-empty and unique")
    return sizes


def _make_dram_tensor(mesh, payload: bytes):
    padded_bytes = ((len(payload) + _CHUNK_BYTES - 1) // _CHUNK_BYTES) * _CHUNK_BYTES
    host = torch.frombuffer(
        bytearray(payload) + bytearray(padded_bytes - len(payload)),
        dtype=torch.uint8,
    ).reshape(padded_bytes // _CHUNK_BYTES, _CHUNK_BYTES)
    return ttnn.from_torch(
        host,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_output_tensor(mesh, payload_bytes: int):
    padded_bytes = ((payload_bytes + _CHUNK_BYTES - 1) // _CHUNK_BYTES) * _CHUNK_BYTES
    host = torch.zeros((padded_bytes // _CHUNK_BYTES, _CHUNK_BYTES), dtype=torch.uint8)
    return ttnn.from_torch(
        host,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _read_payload(tensor, payload_bytes: int) -> bytes:
    return bytes(ttnn.to_torch(tensor).flatten()[:payload_bytes].tolist())


def _exercise_client(
    client: Client,
    client_mesh,
    *,
    iterations: int,
    payload_bytes: int,
    key_prefix: str,
) -> None:
    ping_payload = bytes(range(64))
    if client.ping(ping_payload) != ping_payload:
        raise RuntimeError("PING response did not match the request")
    print("PASS establish+ping", flush=True)

    for iteration in range(iterations):
        key = f"{key_prefix}-{iteration}".encode()
        payload = bytes((iteration + offset) % 256 for offset in range(payload_bytes))
        value = _make_dram_tensor(client_mesh, payload)
        output = _make_output_tensor(client_mesh, len(payload))

        sync_commit = iteration % 2 == 1
        client.put(key, value, len(payload), sync_commit=sync_commit)
        if not sync_commit:
            client.flush()
        if not client.exists(key):
            raise RuntimeError(
                f"iteration {iteration}: EXISTS missed the committed PUT"
            )
        if not client.get(key, output):
            raise RuntimeError(f"iteration {iteration}: GET missed the committed PUT")
        if _read_payload(output, len(payload)) != payload:
            raise RuntimeError(f"iteration {iteration}: device GET payload mismatch")

        client.move_to_host(key)
        if not client.get(key, output):
            raise RuntimeError(f"iteration {iteration}: GET missed after MOVE_TO_HOST")
        if _read_payload(output, len(payload)) != payload:
            raise RuntimeError(
                f"iteration {iteration}: host-backed GET payload mismatch"
            )

        client.move_to_device(key)
        if not client.get(key, output):
            raise RuntimeError(
                f"iteration {iteration}: GET missed after MOVE_TO_DEVICE"
            )
        if _read_payload(output, len(payload)) != payload:
            raise RuntimeError(f"iteration {iteration}: promoted GET payload mismatch")

        if not client.delete(key) or client.exists(key):
            raise RuntimeError(f"iteration {iteration}: REMOVE/EXISTS result mismatch")
        print(
            f"PASS iteration={iteration} bytes={len(payload)} "
            f"put={'committed' if sync_commit else 'landed+flush'} "
            "get=device+host+promoted remove=ok",
            flush=True,
        )


def _padded_value_bytes(payload_bytes: int) -> int:
    return ((payload_bytes + _CHUNK_BYTES - 1) // _CHUNK_BYTES) * _CHUNK_BYTES


def _gbps(payload_bytes: int, duration_ns: float) -> float:
    if duration_ns <= 0:
        raise ValueError("benchmark duration must be positive")
    return payload_bytes * 8.0 / duration_ns


def _benchmark_record(
    path: str,
    payload_bytes: int,
    samples_ns: list[int],
    *,
    warmups: int,
) -> dict[str, object]:
    if not samples_ns:
        raise ValueError("benchmark record requires at least one sample")
    ordered = sorted(samples_ns)
    p50_ns = float(statistics.median(samples_ns))
    return {
        "path": path,
        "scope": "client_host_wall",
        "size": payload_bytes,
        "padded_storage_bytes": _padded_value_bytes(payload_bytes),
        "warmups": warmups,
        "n": len(samples_ns),
        "min_ms": ordered[0] / 1e6,
        "p50_ms": p50_ns / 1e6,
        "max_ms": ordered[-1] / 1e6,
        "min_gbps": _gbps(payload_bytes, ordered[-1]),
        "p50_gbps": _gbps(payload_bytes, p50_ns),
        "max_gbps": _gbps(payload_bytes, ordered[0]),
        "iteration_samples_ms": [sample / 1e6 for sample in samples_ns],
    }


def _poison_output(client_mesh, poison_tensor, output_tensor) -> None:
    ttnn.copy(poison_tensor, output_tensor)
    ttnn.synchronize_device(client_mesh)


def _require_benchmark_payload(
    output_tensor,
    expected: bytes,
    *,
    path: str,
    iteration: int,
) -> None:
    actual = _read_payload(output_tensor, len(expected))
    if actual != expected:
        raise RuntimeError(
            f"{path} payload mismatch size={len(expected)} iteration={iteration}"
        )


def _delete_benchmark_key(client: Client, key: bytes) -> None:
    if not client.delete(key) or client.exists(key):
        raise RuntimeError(f"benchmark DELETE/EXISTS mismatch for key {key!r}")


def _measure_benchmark_path(
    client: Client,
    client_mesh,
    *,
    path: str,
    payload: bytes,
    value_tensor,
    output_tensor,
    poison_tensor,
    warmups: int,
    samples: int,
) -> dict[str, object]:
    payload_bytes = len(payload)
    samples_ns: list[int] = []
    for iteration in range(warmups + samples):
        key = (
            f"manager-bench-{os.getpid()}-{path}-{payload_bytes}-{iteration}"
        ).encode()

        if path == "put_commit":
            started_ns = time.perf_counter_ns()
            client.put(key, value_tensor, payload_bytes, sync_commit=True)
            elapsed_ns = time.perf_counter_ns() - started_ns
            _poison_output(client_mesh, poison_tensor, output_tensor)
            if not client.get(key, output_tensor):
                raise RuntimeError(
                    f"put_commit verification GET missed size={payload_bytes} "
                    f"iteration={iteration}"
                )
        else:
            client.put(key, value_tensor, payload_bytes, sync_commit=True)
            if path == "get_host":
                client.move_to_host(key)
            elif path != "get_device":
                raise ValueError(f"unknown benchmark path: {path}")
            _poison_output(client_mesh, poison_tensor, output_tensor)
            started_ns = time.perf_counter_ns()
            hit = client.get(key, output_tensor)
            elapsed_ns = time.perf_counter_ns() - started_ns
            if not hit:
                raise RuntimeError(
                    f"{path} missed size={payload_bytes} iteration={iteration}"
                )

        _require_benchmark_payload(
            output_tensor,
            payload,
            path=path,
            iteration=iteration,
        )
        _delete_benchmark_key(client, key)
        if iteration >= warmups:
            samples_ns.append(elapsed_ns)

    return _benchmark_record(
        path,
        payload_bytes,
        samples_ns,
        warmups=warmups,
    )


def _git_checkout_metadata(repository: Path) -> dict[str, object]:
    metadata: dict[str, object] = {"path": str(repository.resolve())}
    try:
        metadata["head"] = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        metadata["dirty"] = bool(status.strip())
    except (OSError, subprocess.SubprocessError) as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
    return metadata


def _read_manifest_value(name: str) -> str | None:
    try:
        return (
            (_REPOSITORY_ROOT / "third_party" / "tt-metal" / name)
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"loaded artifact path is not a file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _loaded_native_artifact(module_name: str) -> dict[str, object]:
    metadata: dict[str, object] = {"module": module_name}
    try:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError("loaded module does not expose __file__")
        metadata.update(_file_identity(Path(module_file)))
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
    return metadata


def _loaded_implementation_artifacts() -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": str(_PROC_SELF_MAPS),
        "artifacts": [],
    }
    try:
        paths: dict[str, set[Path]] = {
            name: set() for name in _IMPLEMENTATION_LIBRARY_NAMES
        }
        for line in _PROC_SELF_MAPS.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            path = Path(fields[5])
            name = path.name
            for expected_name in _IMPLEMENTATION_LIBRARY_NAMES:
                if name == expected_name or name.startswith(f"{expected_name}."):
                    paths[expected_name].add(path)

        artifacts = metadata["artifacts"]
        assert isinstance(artifacts, list)
        for expected_name in _IMPLEMENTATION_LIBRARY_NAMES:
            for path in sorted(paths[expected_name]):
                artifacts.append(
                    {"expected_name": expected_name, **_file_identity(path)}
                )
        metadata["missing"] = [
            name for name in _IMPLEMENTATION_LIBRARY_NAMES if not paths[name]
        ]
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
    return metadata


def _write_benchmark_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
            temporary = Path(output.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _print_benchmark_record(record: dict[str, object]) -> None:
    print(
        "KVM_HOST_WALL_BENCH "
        f"path={record['path']} size={record['size']} n={record['n']} "
        f"latency_ms={record['min_ms']:.3f}/{record['p50_ms']:.3f}/{record['max_ms']:.3f} "
        f"gbps={record['min_gbps']:.4f}/{record['p50_gbps']:.4f}/{record['max_gbps']:.4f}",
        flush=True,
    )


def _run_host_wall_benchmark(
    client: Client,
    client_mesh,
    args: argparse.Namespace,
    *,
    topology: str,
) -> tuple[Path, dict[str, object]] | None:
    if args.benchmark_output is None:
        return None

    output_path = Path(args.benchmark_output).expanduser().resolve()
    if output_path.exists() and not args.benchmark_overwrite:
        raise FileExistsError(
            f"benchmark output already exists; pass --benchmark-overwrite to replace it: {output_path}"
        )
    loaded_python_extensions = {
        "kvcache_manager": _loaded_native_artifact("kvcache_manager._native"),
        "ttnn": _loaded_native_artifact("ttnn._ttnn"),
    }
    loaded_implementation_shared_objects = _loaded_implementation_artifacts()
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "status_scope": (
            "client_process" if args.role == "client" else "complete_intranode_run"
        ),
        "measurement_status": "running",
        "cleanup_status": "pending",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "launcher_checkout": _git_checkout_metadata(_REPOSITORY_ROOT),
        "loaded_python_extensions": loaded_python_extensions,
        "loaded_implementation_shared_objects": loaded_implementation_shared_objects,
        "manifest_expectation": {
            "tt_metal_base_revision": _read_manifest_value("REVISION"),
            "tt_metal_target_tree": _read_manifest_value("TARGET_TREE"),
        },
        "provenance": {
            "build_revision_embedded": False,
            "artifact_identity": "selected_host_objects_path_size_sha256",
            "artifact_identity_scope": "benchmark_host_process",
            "device_kernel_build_identity_recorded": False,
            "server_process_artifacts_recorded": args.role != "client",
        },
        "topology": topology,
        "role": args.role,
        "fabric": getattr(args, "fabric", "2d-static-intermesh-vc0"),
        "link_index": getattr(args, "link_index", None),
        "source_chip": getattr(args, "source_chip", None),
        "server_chip": _T3K_ENDPOINT_CHIP if args.role == "client" else None,
        "sizes": list(args.benchmark_sizes),
        "warmups": args.benchmark_warmups,
        "samples": args.benchmark_samples,
        "bandwidth_basis": "logical_value_bytes",
        "timer": "time.perf_counter_ns",
        "byte_check": True,
        "delete_miss_check": True,
        "records": [],
    }
    _write_benchmark_result(output_path, result)
    artifact = (output_path, result)
    args._benchmark_artifact = artifact

    try:
        records = result["records"]
        assert isinstance(records, list)
        for payload_bytes in args.benchmark_sizes:
            payload = bytes(
                (payload_bytes + offset) % 251 for offset in range(payload_bytes)
            )
            value_tensor = _make_dram_tensor(client_mesh, payload)
            output_tensor = _make_output_tensor(client_mesh, payload_bytes)
            poison_tensor = _make_dram_tensor(
                client_mesh, bytes([0xA5]) * payload_bytes
            )
            for path in ("put_commit", "get_device", "get_host"):
                record = _measure_benchmark_path(
                    client,
                    client_mesh,
                    path=path,
                    payload=payload,
                    value_tensor=value_tensor,
                    output_tensor=output_tensor,
                    poison_tensor=poison_tensor,
                    warmups=args.benchmark_warmups,
                    samples=args.benchmark_samples,
                )
                records.append(record)
                _print_benchmark_record(record)
                _write_benchmark_result(output_path, result)

        result["measurement_status"] = "pass"
        result["status"] = "pending_cleanup"
        result["measurement_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_benchmark_result(output_path, result)
        print(f"PASS host-wall measurement output={output_path}", flush=True)
        return artifact
    except BaseException as error:
        result["status"] = "fail"
        result["measurement_status"] = "fail"
        result["failure_phase"] = "measurement"
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        _write_benchmark_result(output_path, result)
        raise


def _finalize_benchmark_after_cleanup(
    artifact: tuple[Path, dict[str, object]] | None,
    *,
    cleanup_error: BaseException | None,
    workload_error: BaseException | None,
) -> None:
    if artifact is None:
        return

    output_path, result = artifact
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    measurement_failed = result.get("measurement_status") == "fail"
    if cleanup_error is not None:
        result["status"] = "fail"
        result["cleanup_status"] = "fail"
        if measurement_failed:
            result["cleanup_error_type"] = type(cleanup_error).__name__
            result["cleanup_error"] = str(cleanup_error)
        else:
            result["failure_phase"] = "cleanup"
            result["error_type"] = type(cleanup_error).__name__
            result["error"] = str(cleanup_error)
    elif workload_error is not None:
        result["status"] = "fail"
        result["cleanup_status"] = "pass"
        if not measurement_failed:
            result["failure_phase"] = "workload"
            result["error_type"] = type(workload_error).__name__
            result["error"] = str(workload_error)
    else:
        result["status"] = "pass"
        result["cleanup_status"] = "pass"

    _write_benchmark_result(output_path, result)
    if result["status"] == "pass":
        print(f"PASS host-wall benchmark output={output_path}", flush=True)


def _close_mesh(mesh) -> None:
    if mesh is not None:
        ttnn.close_mesh_device(mesh)


def _cleanup_runtime(
    *, lifecycle_owners, parent_mesh=None, submeshes=()
) -> BaseException | None:
    first_error = None

    def attempt(label, action) -> None:
        nonlocal first_error
        try:
            action()
        except BaseException as error:
            error = error.with_traceback(None)
            if first_error is None:
                first_error = error
            print(
                f"CLEANUP_ERROR step={label}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

    # Consume the transferred references so native destructors run before meshes close.
    while lifecycle_owners:
        label, owner = lifecycle_owners.pop(0)
        if owner is not None:
            attempt(f"{label}.close", owner.close)
            del owner
    if parent_mesh is not None:
        attempt("parent_mesh.quiesce_devices", parent_mesh.quiesce_devices)
    for index, submesh in enumerate(submeshes):
        if submesh is not None:
            attempt(f"submesh[{index}].close", lambda mesh=submesh: _close_mesh(mesh))
    if parent_mesh is not None:
        attempt("parent_mesh.close", lambda: _close_mesh(parent_mesh))
    attempt(
        "fabric.disable",
        lambda: ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED),
    )
    return first_error


def _raise_cleanup_error_without_masking_workload(
    cleanup_error: BaseException | None, workload_error: BaseException | None
) -> None:
    if cleanup_error is not None and workload_error is None:
        raise cleanup_error


def _run_intranode(args: argparse.Namespace) -> None:
    if ttnn.get_num_devices() < 2:
        raise RuntimeError("intranode role requires at least two visible devices")

    parent_mesh = None
    client_submesh = None
    server_submesh = None
    server = None
    client = None
    benchmark_artifact = None
    try:
        fabric_config = {
            "1d": ttnn.FabricConfig.FABRIC_1D,
            "2d": ttnn.FabricConfig.FABRIC_2D,
        }[args.fabric]
        ttnn.set_fabric_config(fabric_config)
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 1))
        client_submesh = parent_mesh.create_submesh(
            ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0)
        )
        server_submesh = parent_mesh.create_submesh(
            ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, 0)
        )
        server = _create_intranode_server(
            server_submesh,
            delay_first_response=args.test_delay_first_establish_response,
            drop_first_response=args.test_drop_first_establish_response,
        )
        server_node = server_submesh.get_fabric_node_id(ttnn.MeshCoordinate(0, 0))
        client = Client(client_submesh, server_node)
        if not server.is_ready() or not client.is_established():
            raise RuntimeError("intranode Server/Client did not become ready")
        _exercise_client(
            client,
            client_submesh,
            iterations=args.iterations,
            payload_bytes=args.payload_bytes,
            key_prefix="manager-python-intranode",
        )
        benchmark_artifact = _run_host_wall_benchmark(
            client,
            client_submesh,
            args,
            topology="intranode",
        )
    finally:
        workload_error = sys.exc_info()[1]
        if workload_error is not None:
            traceback.clear_frames(workload_error.__traceback__)
        lifecycle_owners = [("client", client), ("server", server)]
        client = None
        server = None
        cleanup_error = _cleanup_runtime(
            lifecycle_owners=lifecycle_owners,
            parent_mesh=parent_mesh,
            submeshes=(client_submesh, server_submesh),
        )
        benchmark_artifact = benchmark_artifact or getattr(
            args, "_benchmark_artifact", None
        )
        _finalize_benchmark_after_cleanup(
            benchmark_artifact,
            cleanup_error=cleanup_error,
            workload_error=workload_error,
        )
        _raise_cleanup_error_without_masking_workload(cleanup_error, workload_error)


def _router_config(role: str, link_index: int):
    router = ttnn.FabricRouterConfig()
    router.intermesh_vc0_only = True
    router.static_intermesh_endpoint_link_index = link_index
    if role == "server":
        router.static_intermesh_endpoint_role = (
            ttnn.StaticIntermeshEndpointRole.T3K_ENDPOINT
        )
        router.static_intermesh_endpoint_physical_chip = _T3K_ENDPOINT_CHIP
        router.static_intermesh_peer_handshake_address = 0x11000
        router.static_intermesh_peer_receiver_base_address = 0x2FD60
        router.static_intermesh_peer_receiver_num_slots = 11
    else:
        router.static_intermesh_endpoint_role = (
            ttnn.StaticIntermeshEndpointRole.GALAXY_GATEWAY
        )
        router.static_intermesh_endpoint_physical_chip = _GALAXY_ENDPOINT_CHIP
        router.static_intermesh_peer_handshake_address = 0x18000
        router.static_intermesh_peer_receiver_base_address = 0x29CE0
        router.static_intermesh_peer_receiver_num_slots = 16
    return router


def _enable_static_fabric(role: str, link_index: int) -> None:
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        _router_config(role, link_index),
    )


def _endpoint_submesh(parent_mesh, physical_chip: int):
    device_ids = [int(device_id) for device_id in parent_mesh.get_device_ids()]
    if physical_chip not in device_ids:
        raise RuntimeError(
            f"physical chip D{physical_chip} is not in the opened mesh: {device_ids}"
        )
    index = device_ids.index(physical_chip)
    return parent_mesh.create_submesh(
        ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(index // 4, index % 4)
    )


def _run_server(args: argparse.Namespace) -> None:
    if ttnn.get_num_devices() != 8:
        raise RuntimeError("server role requires exactly eight visible T3K devices")

    parent_mesh = None
    server_submesh = None
    server = None
    try:
        _enable_static_fabric("server", args.link_index)
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
        server_submesh = _endpoint_submesh(parent_mesh, _T3K_ENDPOINT_CHIP)
        server = Server(
            server_submesh,
            static_intermesh_t3k_server=True,
            static_intermesh_remote_mesh_id=args.remote_mesh_id,
        )
        print("SERVER_READY", flush=True)

        stop = threading.Event()

        def request_stop(_signum, _frame):
            stop.set()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        deadline = time.monotonic() + args.run_timeout_seconds
        stop_path = Path(args.stop_file)
        while not stop.wait(0.5) and not stop_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "server stop signal was not delivered before timeout"
                )
    finally:
        workload_error = sys.exc_info()[1]
        if workload_error is not None:
            traceback.clear_frames(workload_error.__traceback__)
        lifecycle_owners = [("server", server)]
        server = None
        cleanup_error = _cleanup_runtime(
            lifecycle_owners=lifecycle_owners,
            parent_mesh=parent_mesh,
            submeshes=(server_submesh,),
        )
        _raise_cleanup_error_without_masking_workload(cleanup_error, workload_error)


def _run_client(args: argparse.Namespace) -> None:
    if ttnn.get_num_devices() != 32:
        raise RuntimeError("client role requires exactly 32 visible Galaxy devices")

    parent_mesh = None
    client_submesh = None
    client = None
    benchmark_artifact = None
    try:
        _enable_static_fabric("client", args.link_index)
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
        client_submesh = _endpoint_submesh(parent_mesh, args.source_chip)
        print("CLIENT_FABRIC_READY", flush=True)
        client = Client(
            client_submesh,
            static_intermesh_galaxy_to_t3k=True,
        )
        if not client.is_established():
            raise RuntimeError("static inter-mesh Client did not establish")
        _exercise_client(
            client,
            client_submesh,
            iterations=args.iterations,
            payload_bytes=args.payload_bytes,
            key_prefix="manager-python-intermesh",
        )
        benchmark_artifact = _run_host_wall_benchmark(
            client,
            client_submesh,
            args,
            topology="static-intermesh",
        )
    finally:
        workload_error = sys.exc_info()[1]
        if workload_error is not None:
            traceback.clear_frames(workload_error.__traceback__)
        lifecycle_owners = [("client", client)]
        client = None
        cleanup_error = _cleanup_runtime(
            lifecycle_owners=lifecycle_owners,
            parent_mesh=parent_mesh,
            submeshes=(client_submesh,),
        )
        benchmark_artifact = benchmark_artifact or getattr(
            args, "_benchmark_artifact", None
        )
        _finalize_benchmark_after_cleanup(
            benchmark_artifact,
            cleanup_error=cleanup_error,
            workload_error=workload_error,
        )
        _raise_cleanup_error_without_masking_workload(cleanup_error, workload_error)


def _add_workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=_positive_int, default=2)
    parser.add_argument("--payload-bytes", type=_positive_int, default=16 * 1024)


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        help="write host-wall benchmark samples and summary to this JSON file",
    )
    parser.add_argument(
        "--benchmark-sizes",
        type=_benchmark_sizes,
        default=_DEFAULT_BENCHMARK_SIZES,
        metavar="BYTES,...",
    )
    parser.add_argument(
        "--benchmark-samples", type=_positive_int, default=20, metavar="N"
    )
    parser.add_argument(
        "--benchmark-warmups", type=_non_negative_int, default=1, metavar="N"
    )
    parser.add_argument(
        "--benchmark-overwrite",
        action="store_true",
        help="replace an existing benchmark output file",
    )


def _validate_benchmark_arguments(args: argparse.Namespace) -> None:
    if getattr(args, "benchmark_output", None) is None:
        return
    output_path = Path(args.benchmark_output).expanduser().resolve()
    if output_path.exists() and not args.benchmark_overwrite:
        raise FileExistsError(
            f"benchmark output already exists; pass --benchmark-overwrite to replace it: {output_path}"
        )

def _require_fresh_path(path: Path, label: str) -> None:
    if path.exists():
        raise RuntimeError(f"{label} already exists; use a run-specific path: {path}")
    if not path.parent.is_dir():
        raise RuntimeError(f"{label} parent directory does not exist: {path.parent}")


def _validate_run_paths(args: argparse.Namespace) -> None:
    if args.role == "server":
        stop_path = Path(args.stop_file)
        _require_fresh_path(stop_path, "server stop file")

    _validate_benchmark_arguments(args)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)

    intranode = subparsers.add_parser("intranode")
    intranode.add_argument("--fabric", choices=("1d", "2d"), default="1d")
    establish_fault = intranode.add_mutually_exclusive_group()
    establish_fault.add_argument(
        "--test-delay-first-establish-response",
        action="store_true",
        help="delay the first accepted ESTABLISH response beyond one retry",
    )
    establish_fault.add_argument(
        "--test-drop-first-establish-response",
        action="store_true",
        help="drop the first accepted ESTABLISH response to exercise retry state",
    )
    _add_workload_arguments(intranode)
    _add_benchmark_arguments(intranode)

    server = subparsers.add_parser("server")
    server.add_argument("--stop-file", default="/tmp/kvm-server-stop")
    server.add_argument("--link-index", type=_static_link_index, default=1)
    server.add_argument("--remote-mesh-id", type=_non_negative_int, default=0)
    server.add_argument(
        "--run-timeout-seconds", type=_positive_seconds, default=30 * 60
    )

    client = subparsers.add_parser("client")
    client.add_argument("--link-index", type=_static_link_index, default=1)
    client.add_argument(
        "--source-chip", type=_non_negative_int, default=_GALAXY_ENDPOINT_CHIP
    )
    _add_workload_arguments(client)
    _add_benchmark_arguments(client)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_run_paths(args)
    if args.role == "intranode":
        _run_intranode(args)
    elif args.role == "server":
        _run_server(args)
    else:
        _run_client(args)


if __name__ == "__main__":
    main()
