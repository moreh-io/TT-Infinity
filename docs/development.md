# Development, Build, and Test

This document is for developers who modify TT-Infinity and build its `kvcache-manager` C++ library, Python bindings, wheel, and test executables. To run only the Qwen3-32B or multi-turn prefix-reuse workloads, follow the [README Quick Start](../README.md#quick-start) and the [usage guide](usage.md).

## Development tt-metal Tree

First prepare a built tt-metal tree at the matching revision with the patch series applied, as described in the [README Quick Start](../README.md#quick-start). Development builds and tests must use the same tt-metal source and TTNN installation.

## Building kvcache-manager

Configure the C++ library, Python bindings, and optional device-test executables against the matching TTNN installation.

```bash
cd /path/to/TT-Infinity
uv pip install --python "$VIRTUAL_ENV/bin/python" "nanobind==2.12.0" "pytest>=8"
cmake -S . -B build -G Ninja \
    -DCMAKE_PREFIX_PATH=/path/to/tt-metal/build_Release \
    -DTT-NN_DIR=/path/to/tt-metal/build_Release/lib/cmake/tt-nn \
    -DTT-Metalium_DIR=/path/to/tt-metal/build_Release/lib/cmake/tt-metalium \
    -DKVCACHE_MANAGER_BUILD_PYTHON_BINDINGS=ON \
    -DKVCACHE_MANAGER_BUILD_DEVICE_TESTS=ON
cmake --build build
```

Build the `kvcache-manager` Python wheel against the same TTNN build.

```bash
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/path/to/tt-metal/build_Release \
-DTT-NN_DIR=/path/to/tt-metal/build_Release/lib/cmake/tt-nn \
-DTT-Metalium_DIR=/path/to/tt-metal/build_Release/lib/cmake/tt-metalium" \
    python -m build --wheel
```

The wheel contains the native extension and private `libkvcache_manager` shared library. The deployment environment supplies TTNN and the tt-metal runtime.

## Host-only Tests

### CTest

After the build, run all host-side tests except tests labeled `device`:

```bash
ctest --test-dir build -LE device --output-on-failure
```

### pytest

The default pytest run uses repository-provided test doubles and requires neither a TTNN installation nor a device.

```bash
python -m pytest
```

The project configuration excludes tests marked `device`.

### External Patch Bundles and Multi-Turn Fixture

Host tests validate the revision, patch order, checksums, and target-tree contract for the `third_party/tt-metal`, `third_party/vllm-tt-plugin`, and `third_party/aiperf` bundles. The AIPerf bundle pins the SemiAnalysisAI fork used by InferenceX and adds only offline local-tokenizer directory resolution; it does not change the official scenario or result calculations.

When changing the multi-turn fixture, run the generator in the prepared AIPerf environment and verify that the checked-in trace matches the regenerated output.

```bash
"$AIPERF_VENV/bin/python" \
    benchmarks/multiturn/build_multiturn_trace.py \
    --output-dir /tmp/multiturn-trace-check
cmp \
    benchmarks/multiturn/datasets/000-qwen3-32b-tt-prefix-reuse-5turn-4k-7k.json \
    /tmp/multiturn-trace-check/000-qwen3-32b-tt-prefix-reuse-5turn-4k-7k.json
```

## Device Tests

### Device pytest

Device pytest covers PING, EXISTS, tensor validation, topology, IRAM budget, gather/scatter, and per-chip pack/unpack. On a host with devices, run selected tests against the built native package.

```bash
PYTHONPATH=/path/to/TT-Infinity/build/python \
    python -m pytest -o pythonpath= -m device \
    tests/device/test_kvcache_manager_ping.py
```

The `pythonpath=` override is required in a source checkout because the default host-test configuration exposes `src`, which would shadow the package containing the built `_native` extension. When testing an installed wheel, omit `PYTHONPATH` but retain the override.

### Intranode End-to-End Tests

C++ device tests cover ESTABLISH, PUT, GET, PING, EXISTS, and REMOVE.

```bash
ctest --test-dir build -L device --output-on-failure
```

The Python product runner exercises both PUT completion modes, GET, MOVE, byte equality, and removal through the Python bindings.

```bash
PYTHONPATH=/path/to/TT-Infinity/build/python TT_METAL_INSPECTOR=0 \
    python tests/product_device_smoke.py intranode \
    --iterations 2 --payload-bytes 16384
```

When using a TTNN build tree instead of an installed package, set `TT_METAL_RUNTIME_ROOT` to that tt-metal checkout. CTest disables TT-Metal Inspector for device targets to match the canonical cache-test environment and avoid affecting short Fabric control workloads.

### Internode End-to-End Tests

Internode validation runs a Galaxy `Client` and T3K `Server` connected through a supported intermesh configuration. See [Using TT-Infinity with vLLM and LMCache](usage.md) for environment setup and the validated Qwen3-32B procedure.
