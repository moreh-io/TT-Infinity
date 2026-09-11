#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: setup_vllm_tt_plugin.sh \
    [--tt-metal-dir PATH] \
    [--plugin-dir PATH] \
    [--venv-dir PATH]

Prepare the pinned vLLM TT plugin and install the Galaxy Qwen3/LMCache environment against an
already built, canonically patched tt-metal checkout. All paths are explicit. The script installs
only into VENV_DIR and does not modify tt-metal's Python environment. Set TT_METAL_ROOT, PLUGIN,
and PLUGIN_VENV, or pass the matching command-line options explicitly.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(realpath "$SCRIPT_DIR/..")
# shellcheck source=lib/patch_bundle.sh
source "$SCRIPT_DIR/lib/patch_bundle.sh"

PATCH_BUNDLE_PROGRAM=setup_vllm_tt_plugin.sh
TT_METAL_DIR=${TT_METAL_ROOT:-}
PLUGIN_DIR=${PLUGIN:-}
VENV_DIR=${PLUGIN_VENV:-}

validate_plugin_venv() {
    local venv_dir=$1
    local validation_result

    [[ -f "$venv_dir/pyvenv.cfg" ]] ||
        patch_bundle_fail "Python environment is missing pyvenv.cfg: $venv_dir"
    if ! validation_result=$("$venv_dir/bin/python" - "$venv_dir" <<'PY'
from pathlib import Path
import sys

expected_prefix = Path(sys.argv[1]).resolve()
actual_prefix = Path(sys.prefix).resolve()
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, got {sys.version.split()[0]}")
if actual_prefix != expected_prefix:
    raise SystemExit(
        f"interpreter prefix {actual_prefix} does not match environment {expected_prefix}"
    )
print("VENV_VALIDATED")
PY
    ); then
        patch_bundle_fail "refusing to install into an invalid Python environment: $venv_dir"
    fi
    [[ "$validation_result" == VENV_VALIDATED ]] ||
        patch_bundle_fail "Python environment validation returned unexpected output: $venv_dir"
}

patch_bundle_reject_git_path_overrides

while (($#)); do
    case "$1" in
        --tt-metal-dir)
            (($# >= 2)) || patch_bundle_fail "--tt-metal-dir requires a path"
            TT_METAL_DIR=$2
            shift 2
            ;;
        --plugin-dir)
            (($# >= 2)) || patch_bundle_fail "--plugin-dir requires a path"
            PLUGIN_DIR=$2
            shift 2
            ;;
        --venv-dir)
            (($# >= 2)) || patch_bundle_fail "--venv-dir requires a path"
            VENV_DIR=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            patch_bundle_fail "unknown argument: $1"
            ;;
    esac
done

[[ -n "$TT_METAL_DIR" ]] || patch_bundle_fail "TT_METAL_ROOT or --tt-metal-dir is required"
[[ -n "$PLUGIN_DIR" ]] || patch_bundle_fail "PLUGIN or --plugin-dir is required"
[[ -n "$VENV_DIR" ]] || patch_bundle_fail "PLUGIN_VENV or --venv-dir is required"
command -v uv >/dev/null || patch_bundle_fail "uv is required"

TT_METAL_DIR=$(patch_bundle_resolve_checkout "$TT_METAL_DIR")
PLUGIN_DIR=$(realpath -m "$PLUGIN_DIR")
VENV_DIR=$(realpath -m "$VENV_DIR")
[[ "$PLUGIN_DIR" != "$VENV_DIR" ]] || patch_bundle_fail "plugin and venv paths must differ"
case "$PLUGIN_DIR/" in
    "$TT_METAL_DIR/"*|"$REPOSITORY_ROOT/"*)
        patch_bundle_fail "plugin checkout must be outside the source repositories: $PLUGIN_DIR"
        ;;
esac
case "$VENV_DIR/" in
    "$TT_METAL_DIR/"*|"$REPOSITORY_ROOT/"*|"$PLUGIN_DIR/"*)
        patch_bundle_fail "venv must be outside the source and plugin repositories: $VENV_DIR"
        ;;
esac
case "$PLUGIN_DIR/" in
    "$VENV_DIR/"*)
        patch_bundle_fail "plugin checkout must not be inside the venv: $PLUGIN_DIR"
        ;;
esac
mkdir -p "$(dirname -- "$PLUGIN_DIR")" "$(dirname -- "$VENV_DIR")"
if [[ -e "$VENV_DIR" && ! -x "$VENV_DIR/bin/python" ]]; then
    patch_bundle_fail "existing venv path is not a Python environment: $VENV_DIR"
fi
if [[ -e "$VENV_DIR" ]]; then
    validate_plugin_venv "$VENV_DIR"
fi

patch_bundle_validate_prepared "$TT_METAL_DIR" "$REPOSITORY_ROOT/third_party/tt-metal"

TTNN_CMAKE_DIR="$TT_METAL_DIR/build_Release/lib/cmake/tt-nn"
METALIUM_CMAKE_DIR="$TT_METAL_DIR/build_Release/lib/cmake/tt-metalium"
[[ -f "$TTNN_CMAKE_DIR/tt-nn-config.cmake" ]] ||
    patch_bundle_fail "missing TTNN CMake package: $TTNN_CMAKE_DIR/tt-nn-config.cmake"
[[ -f "$METALIUM_CMAKE_DIR/tt-metalium-config.cmake" ]] ||
    patch_bundle_fail "missing Metalium CMake package: $METALIUM_CMAKE_DIR/tt-metalium-config.cmake"
[[ -f "$TT_METAL_DIR/lmcache-t3knic/pyproject.toml" ]] ||
    patch_bundle_fail "patched lmcache-t3knic source is missing from tt-metal"

VLLM_VERSION=$(tr -d '[:space:]' <"$REPOSITORY_ROOT/third_party/vllm-tt-plugin/VLLM_VERSION")
LMCACHE_VERSION=$(tr -d '[:space:]' <"$REPOSITORY_ROOT/third_party/lmcache/VERSION")
[[ "$VLLM_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
    patch_bundle_fail "invalid pinned vLLM version: $VLLM_VERSION"
[[ "$LMCACHE_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
    patch_bundle_fail "invalid pinned LMCache version: $LMCACHE_VERSION"

"$SCRIPT_DIR/prepare_vllm_tt_plugin.sh" --plugin-dir "$PLUGIN_DIR"

if [[ ! -e "$VENV_DIR" ]]; then
    uv venv --python 3.10 "$VENV_DIR"
    validate_plugin_venv "$VENV_DIR"
fi

VLLM_TARGET_DEVICE=empty uv pip install \
    --python "$VENV_DIR/bin/python" \
    --no-binary vllm \
    "vllm==$VLLM_VERSION" \
    "transformers==5.5.3"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    -e "$TT_METAL_DIR" \
    -e "$PLUGIN_DIR" \
    "transformers==5.5.3"

CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$TT_METAL_DIR/build_Release -DTT-NN_DIR=$TTNN_CMAKE_DIR -DTT-Metalium_DIR=$METALIUM_CMAKE_DIR" \
    uv pip install \
    --python "$VENV_DIR/bin/python" \
    --no-deps \
    --reinstall \
    "$REPOSITORY_ROOT"

uv pip install \
    --python "$VENV_DIR/bin/python" \
    --no-deps \
    "lmcache==$LMCACHE_VERSION" \
    -e "$TT_METAL_DIR/lmcache-t3knic"
uv pip install \
    --python "$VENV_DIR/bin/python" \
    --no-deps \
    sortedcontainers \
    aiofile \
    caio
uv pip install \
    --python "$VENV_DIR/bin/python" \
    "pytest>=8"

TT_METAL_HOME="$TT_METAL_DIR" \
PYTHONPATH="$TT_METAL_DIR:$TT_METAL_DIR/lmcache-t3knic${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_DIR/bin/python" -c \
    'import kvcache_manager, lmcache, lmcache_t3knic, ttnn, vllm, vllm_tt_plugin; import models.tt_transformers.tt.generator_vllm; import lmcache_t3knic.t3knic_lmcache.t3knic_kv_connector'

printf 'TT Metal: %s\nKV manager: %s\nPlugin: %s\nEnvironment: %s\n' \
    "$TT_METAL_DIR" "$REPOSITORY_ROOT" "$PLUGIN_DIR" "$VENV_DIR"
