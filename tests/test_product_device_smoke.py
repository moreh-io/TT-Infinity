from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.fixture
def product_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, SimpleNamespace]:
    fake_ttnn = SimpleNamespace(
        FabricConfig=SimpleNamespace(DISABLED="disabled"),
        set_fabric_config=lambda _config: None,
    )
    monkeypatch.setitem(sys.modules, "ttnn", fake_ttnn)

    path = Path(__file__).with_name("product_device_smoke.py")
    spec = importlib.util.spec_from_file_location("product_device_smoke_tested", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fake_ttnn


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_non_positive_values(
    product_smoke: tuple[ModuleType, SimpleNamespace], value: str
) -> None:
    module, _ = product_smoke

    with pytest.raises(module.argparse.ArgumentTypeError, match="positive integer"):
        module._positive_int(value)


@pytest.mark.parametrize("value", ["", "1,", "0,4096", "4096,4096"])
def test_benchmark_sizes_reject_invalid_values(
    product_smoke: tuple[ModuleType, SimpleNamespace], value: str
) -> None:
    module, _ = product_smoke

    with pytest.raises(module.argparse.ArgumentTypeError):
        module._benchmark_sizes(value)


@pytest.mark.parametrize("value", ["-1", "4"])
def test_static_link_index_rejects_out_of_range_values(
    product_smoke: tuple[ModuleType, SimpleNamespace], value: str
) -> None:
    module, _ = product_smoke

    with pytest.raises(module.argparse.ArgumentTypeError, match="between 0 and 3"):
        module._static_link_index(value)


def test_cli_defaults_to_validated_static_link_and_optional_2d_intranode(
    product_smoke: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, _ = product_smoke

    client = module._parse_args(["client"])
    intranode = module._parse_args(["intranode", "--fabric", "2d"])

    assert client.link_index == 1
    assert client.benchmark_sizes == module._DEFAULT_BENCHMARK_SIZES
    assert client.benchmark_samples == 20
    assert client.benchmark_warmups == 1
    assert intranode.fabric == "2d"
    assert intranode.test_delay_first_establish_response is False
    assert intranode.test_drop_first_establish_response is False


def test_intranode_retry_fault_uses_private_native_factory(
    product_smoke: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = product_smoke
    submesh = object()
    created = object()

    class NativeServer:
        @staticmethod
        def _test_create_drop_first_establish_response(value):
            assert value is submesh
            return created

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: SimpleNamespace(Server=NativeServer)
        if name == "kvcache_manager._native"
        else None,
    )

    assert (
        module._create_intranode_server(
            submesh,
            delay_first_response=False,
            drop_first_response=True,
        )
        is created
    )


def test_intranode_delayed_retry_fault_uses_private_native_factory(
    product_smoke: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = product_smoke
    submesh = object()
    created = object()

    class NativeServer:
        @staticmethod
        def _test_create_delay_first_establish_response(value):
            assert value is submesh
            return created

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: SimpleNamespace(Server=NativeServer)
        if name == "kvcache_manager._native"
        else None,
    )

    assert (
        module._create_intranode_server(
            submesh,
            delay_first_response=True,
            drop_first_response=False,
        )
        is created
    )


def test_benchmark_record_uses_logical_payload_gbps(
    product_smoke: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, _ = product_smoke

    record = module._benchmark_record("get_device", 1024, [1000, 2000, 3000], warmups=1)

    assert record["p50_ms"] == pytest.approx(0.002)
    assert record["p50_gbps"] == pytest.approx(4.096)
    assert record["padded_storage_bytes"] == 4288
    assert record["n"] == 3


@pytest.mark.parametrize(
    ("path", "expected_first_iteration", "expected_moves"),
    [
        (
            "put_commit",
            ["clock", "put", "clock", "poison", "get", "verify", "delete", "exists"],
            0,
        ),
        (
            "get_device",
            ["put", "poison", "clock", "get", "clock", "verify", "delete", "exists"],
            0,
        ),
        (
            "get_host",
            [
                "put",
                "move",
                "poison",
                "clock",
                "get",
                "clock",
                "verify",
                "delete",
                "exists",
            ],
            3,
        ),
    ],
)
def test_measure_benchmark_path_excludes_setup_and_verifies_every_iteration(
    product_smoke: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    expected_first_iteration: list[str],
    expected_moves: int,
) -> None:
    module, _ = product_smoke
    clock = iter((0, 10, 20, 40, 50, 80))
    verified: list[int] = []
    events: list[str] = []

    class Client:
        def __init__(self) -> None:
            self.keys: set[bytes] = set()
            self.moves = 0

        def put(self, key, _value, _length, *, sync_commit):
            assert sync_commit is True
            events.append("put")
            self.keys.add(key)

        def move_to_host(self, key):
            assert key in self.keys
            events.append("move")
            self.moves += 1

        def get(self, key, _output):
            events.append("get")
            return key in self.keys

        def delete(self, key):
            events.append("delete")
            self.keys.remove(key)
            return True

        def exists(self, key):
            events.append("exists")
            return key in self.keys

    client = Client()

    def read_clock():
        events.append("clock")
        return next(clock)

    monkeypatch.setattr(module.time, "perf_counter_ns", read_clock)
    monkeypatch.setattr(
        module, "_poison_output", lambda *_args: events.append("poison")
    )

    def verify(_output, _expected, *, path, iteration):
        events.append("verify")
        verified.append(iteration)

    monkeypatch.setattr(module, "_require_benchmark_payload", verify)

    record = module._measure_benchmark_path(
        client,
        object(),
        path=path,
        payload=b"payload",
        value_tensor=object(),
        output_tensor=object(),
        poison_tensor=object(),
        warmups=1,
        samples=2,
    )

    assert record["iteration_samples_ms"] == [20 / 1e6, 30 / 1e6]
    assert verified == [0, 1, 2]
    assert events[: len(expected_first_iteration)] == expected_first_iteration
    assert client.moves == expected_moves
    assert client.keys == set()


def test_benchmark_result_is_atomic(
    product_smoke: tuple[ModuleType, SimpleNamespace], tmp_path: Path
) -> None:
    module, _ = product_smoke
    output = tmp_path / "results" / "benchmark.json"
    result = {"status": "pass", "records": [{"p50_gbps": 1.0}]}

    module._write_benchmark_result(output, result)

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_loaded_native_artifact_records_actual_file_identity(
    product_smoke: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _ = product_smoke
    artifact = tmp_path / "_native.so"
    artifact.write_bytes(b"native-artifact")
    monkeypatch.setitem(
        sys.modules, "test_fake_native", SimpleNamespace(__file__=str(artifact))
    )

    metadata = module._loaded_native_artifact("test_fake_native")

    assert metadata["path"] == str(artifact.resolve())
    assert metadata["size_bytes"] == len(b"native-artifact")
    assert metadata["sha256"] == module.hashlib.sha256(b"native-artifact").hexdigest()


def test_loaded_implementation_artifacts_come_from_process_maps(
    product_smoke: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _ = product_smoke
    libraries = []
    for index, name in enumerate(module._IMPLEMENTATION_LIBRARY_NAMES):
        path = tmp_path / name
        path.write_bytes(f"artifact-{index}".encode())
        libraries.append(path)
    maps = tmp_path / "maps"
    maps.write_text(
        "\n".join(
            f"7f{index:010x}-7f{index + 1:010x} r-xp 00000000 00:00 0 {path}"
            for index, path in enumerate(libraries)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_PROC_SELF_MAPS", maps)

    metadata = module._loaded_implementation_artifacts()

    assert metadata["missing"] == []
    assert {artifact["expected_name"] for artifact in metadata["artifacts"]} == set(
        module._IMPLEMENTATION_LIBRARY_NAMES
    )
    assert all(len(artifact["sha256"]) == 64 for artifact in metadata["artifacts"])


def test_host_wall_benchmark_writes_all_paths(
    product_smoke: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _ = product_smoke
    output = tmp_path / "benchmark.json"
    args = SimpleNamespace(
        benchmark_output=output,
        benchmark_overwrite=False,
        benchmark_sizes=(1024,),
        benchmark_samples=2,
        benchmark_warmups=1,
        role="intranode",
        fabric="2d",
    )
    measured_paths: list[str] = []

    monkeypatch.setattr(module, "_make_dram_tensor", lambda *_args: object())
    monkeypatch.setattr(module, "_make_output_tensor", lambda *_args: object())

    def measure(_client, _mesh, *, path, payload, warmups, samples, **_kwargs):
        measured_paths.append(path)
        assert len(payload) == 1024
        assert warmups == 1
        assert samples == 2
        return module._benchmark_record(path, 1024, [1000, 2000], warmups=1)

    monkeypatch.setattr(module, "_measure_benchmark_path", measure)

    artifact = module._run_host_wall_benchmark(
        object(), object(), args, topology="intranode"
    )

    pending = json.loads(output.read_text(encoding="utf-8"))
    assert pending["status"] == "pending_cleanup"
    assert pending["measurement_status"] == "pass"

    module._finalize_benchmark_after_cleanup(
        artifact,
        cleanup_error=None,
        workload_error=None,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "pass"
    assert result["cleanup_status"] == "pass"
    assert result["fabric"] == "2d"
    assert [record["path"] for record in result["records"]] == [
        "put_commit",
        "get_device",
        "get_host",
    ]
    assert measured_paths == ["put_commit", "get_device", "get_host"]


def test_cleanup_failure_prevents_benchmark_pass(
    product_smoke: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _ = product_smoke
    output = tmp_path / "benchmark.json"
    args = SimpleNamespace(
        benchmark_output=output,
        benchmark_overwrite=False,
        benchmark_sizes=(1024,),
        benchmark_samples=1,
        benchmark_warmups=0,
        role="intranode",
        fabric="2d",
    )

    monkeypatch.setattr(module, "_make_dram_tensor", lambda *_args: object())
    monkeypatch.setattr(module, "_make_output_tensor", lambda *_args: object())
    monkeypatch.setattr(
        module,
        "_measure_benchmark_path",
        lambda _client, _mesh, *, path, **_kwargs: module._benchmark_record(
            path, 1024, [1000], warmups=0
        ),
    )

    artifact = module._run_host_wall_benchmark(
        object(), object(), args, topology="intranode"
    )
    module._finalize_benchmark_after_cleanup(
        artifact,
        cleanup_error=RuntimeError("injected cleanup failure"),
        workload_error=None,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "fail"
    assert result["measurement_status"] == "pass"
    assert result["cleanup_status"] == "fail"
    assert result["failure_phase"] == "cleanup"
    assert result["error"] == "injected cleanup failure"


def test_host_wall_benchmark_preserves_failure_in_result(
    product_smoke: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _ = product_smoke
    output = tmp_path / "benchmark.json"
    args = SimpleNamespace(
        benchmark_output=output,
        benchmark_overwrite=False,
        benchmark_sizes=(1024,),
        benchmark_samples=1,
        benchmark_warmups=0,
        role="intranode",
        fabric="2d",
    )

    monkeypatch.setattr(module, "_make_dram_tensor", lambda *_args: object())
    monkeypatch.setattr(module, "_make_output_tensor", lambda *_args: object())

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected benchmark failure")

    monkeypatch.setattr(module, "_measure_benchmark_path", fail)

    with pytest.raises(
        RuntimeError, match="injected benchmark failure"
    ) as raised_error:
        module._run_host_wall_benchmark(object(), object(), args, topology="intranode")
    module._finalize_benchmark_after_cleanup(
        args._benchmark_artifact,
        cleanup_error=None,
        workload_error=raised_error.value,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "fail"
    assert result["measurement_status"] == "fail"
    assert result["cleanup_status"] == "pass"
    assert result["failure_phase"] == "measurement"
    assert result["error_type"] == "RuntimeError"
    assert result["error"] == "injected benchmark failure"


def test_benchmark_argument_validation_refuses_stale_output(
    product_smoke: tuple[ModuleType, SimpleNamespace], tmp_path: Path
) -> None:
    module, _ = product_smoke
    output = tmp_path / "benchmark.json"
    output.write_text("{}", encoding="utf-8")
    intranode = module._parse_args(["intranode", "--benchmark-output", str(output)])

    with pytest.raises(FileExistsError, match="benchmark output already exists"):
        module._validate_benchmark_arguments(intranode)

    overwrite = module._parse_args(
        [
            "intranode",
            "--benchmark-output",
            str(output),
            "--benchmark-overwrite",
        ]
    )
    module._validate_benchmark_arguments(overwrite)

def test_run_path_validation_refuses_stale_control_files(
    product_smoke: tuple[ModuleType, SimpleNamespace], tmp_path: Path
) -> None:
    module, _ = product_smoke
    stop = tmp_path / "stop"
    stop.write_text("", encoding="utf-8")

    server = module._parse_args(["server", "--stop-file", str(stop)])
    with pytest.raises(RuntimeError, match="server stop file already exists"):
        module._validate_run_paths(server)


def test_cleanup_attempts_every_step_and_keeps_first_error(
    product_smoke: tuple[ModuleType, SimpleNamespace],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, fake_ttnn = product_smoke
    events: list[str] = []
    client_error = RuntimeError("client close failed")

    class Owner:
        def __init__(
            self, label: str, close_error: BaseException | None = None
        ) -> None:
            self.label = label
            self.close_error = close_error

        def close(self) -> None:
            events.append(self.label)
            if self.close_error is not None:
                raise self.close_error

        def __del__(self) -> None:
            events.append(f"destroy:{self.label}")

    def quiesce() -> None:
        events.append("quiesce")

    parent = SimpleNamespace(quiesce_devices=quiesce)

    def close_mesh(mesh) -> None:
        label = "parent" if mesh is parent else mesh
        events.append(f"close:{label}")
        if mesh == "submesh-0":
            raise RuntimeError("submesh close failed")

    def disable_fabric(config: str) -> None:
        events.append(f"fabric:{config}")

    monkeypatch.setattr(module, "_close_mesh", close_mesh)
    fake_ttnn.set_fabric_config = disable_fabric

    lifecycle_owners = [
        ("client", Owner("client", client_error)),
        ("server", Owner("server")),
    ]
    error = module._cleanup_runtime(
        lifecycle_owners=lifecycle_owners,
        parent_mesh=parent,
        submeshes=("submesh-0", "submesh-1"),
    )

    assert error is client_error
    assert lifecycle_owners == []
    assert events == [
        "client",
        "destroy:client",
        "server",
        "destroy:server",
        "quiesce",
        "close:submesh-0",
        "close:submesh-1",
        "close:parent",
        "fabric:disabled",
    ]


def test_cleanup_error_does_not_replace_workload_error(
    product_smoke: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, _ = product_smoke
    cleanup_error = RuntimeError("cleanup failed")

    module._raise_cleanup_error_without_masking_workload(
        cleanup_error, ValueError("workload failed")
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        module._raise_cleanup_error_without_masking_workload(cleanup_error, None)
