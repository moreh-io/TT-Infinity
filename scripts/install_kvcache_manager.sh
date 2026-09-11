#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(realpath "$SCRIPT_DIR/..")
# shellcheck source=lib/patch_bundle.sh
source "$SCRIPT_DIR/lib/patch_bundle.sh"

PATCH_BUNDLE_PROGRAM=install_kvcache_manager.sh
patch_bundle_reject_git_path_overrides

: "${TT_METAL_ROOT:?set TT_METAL_ROOT to the prepared tt-metal checkout}"
command -v uv >/dev/null || {
    printf 'install_kvcache_manager.sh: uv is required\n' >&2
    exit 1
}

TT_METAL_ROOT=$(realpath "$TT_METAL_ROOT")
patch_bundle_validate_prepared "$TT_METAL_ROOT" "$REPOSITORY_ROOT/third_party/tt-metal"
TTNN_CMAKE_DIR=$TT_METAL_ROOT/build_Release/lib/cmake/tt-nn
METALIUM_CMAKE_DIR=$TT_METAL_ROOT/build_Release/lib/cmake/tt-metalium
PYTHON=$TT_METAL_ROOT/python_env/bin/python
LMCACHE_VERSION=$(tr -d '[:space:]' <"$REPOSITORY_ROOT/third_party/lmcache/VERSION")

[[ -f "$TTNN_CMAKE_DIR/tt-nn-config.cmake" ]] || {
    printf 'install_kvcache_manager.sh: missing TTNN CMake package: %s\n' "$TTNN_CMAKE_DIR" >&2
    exit 1
}
[[ -f "$METALIUM_CMAKE_DIR/tt-metalium-config.cmake" ]] || {
    printf 'install_kvcache_manager.sh: missing Metalium CMake package: %s\n' "$METALIUM_CMAKE_DIR" >&2
    exit 1
}
[[ -x "$PYTHON" ]] || {
    printf 'install_kvcache_manager.sh: missing tt-metal Python: %s\n' "$PYTHON" >&2
    exit 1
}
[[ "$LMCACHE_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
    printf 'install_kvcache_manager.sh: invalid pinned LMCache version: %s\n' "$LMCACHE_VERSION" >&2
    exit 1
}
[[ -f "$TT_METAL_ROOT/lmcache-t3knic/pyproject.toml" ]] || {
    printf 'install_kvcache_manager.sh: patched lmcache-t3knic source is missing: %s\n' "$TT_METAL_ROOT/lmcache-t3knic" >&2
    exit 1
}

CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$TT_METAL_ROOT/build_Release -DTT-NN_DIR=$TTNN_CMAKE_DIR -DTT-Metalium_DIR=$METALIUM_CMAKE_DIR" \
    uv pip install \
    --python "$PYTHON" \
    --no-deps \
    --reinstall \
    "$REPOSITORY_ROOT"

# The standalone T3K server imports LMCache for logging and the connector package for its
# intermesh configuration. Install them without their generic dependency sets: those include
# CUDA packages that are neither needed nor valid in the TT-metal Python environment.
uv pip install \
    --python "$PYTHON" \
    --no-deps \
    "lmcache==$LMCACHE_VERSION" \
    -e "$TT_METAL_ROOT/lmcache-t3knic"
uv pip install \
    --python "$PYTHON" \
    --no-deps \
    sortedcontainers \
    aiofile \
    caio \
    msgspec

TT_METAL_HOME="$TT_METAL_ROOT" \
PYTHONPATH="$TT_METAL_ROOT:$TT_METAL_ROOT/lmcache-t3knic${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -c \
    'import kvcache_manager, lmcache; from lmcache_t3knic.t3knic_lmcache import t3k_kvcache_server'
