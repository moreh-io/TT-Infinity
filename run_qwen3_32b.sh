#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage_path=docs/usage.md

fail() {
    printf 'ERROR: %s\nSee %s in this checkout.\n' "$1" "$usage_path" >&2
    exit 2
}

require_env() {
    local name=$1
    [[ -n "${!name-}" ]] || fail "$name is not set"
}

# Quick Start exports the deployment-specific values. See docs/usage.md.
for name in TT_METAL_ROOT PLUGIN PLUGIN_VENV T3K_HOST MODEL; do
    require_env "$name"
done

TT_REPO=${TT_REPO:-$TT_METAL_ROOT}
PLUGIN_PY=${PLUGIN_PY:-${PLUGIN_VENV}/bin/python}
T3K_REPO=${T3K_REPO:-$TT_REPO}
LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE:-$SCRIPT_DIR/configs/lmcache_qwen3_32b.yaml}
LINK=1

T3KNIC_INTERMESH_LINK_COUNT=${T3KNIC_INTERMESH_LINK_COUNT:-2}
T3KNIC_SERVER_SHARDS=${T3KNIC_SERVER_SHARDS:-8}
T3KNIC_WIRE_MODE=${T3KNIC_WIRE_MODE:-per_chip}
REQUESTS=${REQUESTS:-3}
OSL=${OSL:-16}
PROMPT_TOKENS=${PROMPT_TOKENS:-}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
OUT=${OUT:-$PWD/kvm-run}

[[ -d "$TT_REPO" ]] || fail "TT_REPO does not exist: $TT_REPO"
[[ -d "$PLUGIN" ]] || fail "PLUGIN does not exist: $PLUGIN"
[[ -x "$PLUGIN_PY" ]] || fail "PLUGIN_PY is not executable: $PLUGIN_PY"
[[ -e "$MODEL" ]] || fail "MODEL does not exist: $MODEL"
[[ -s "$LMCACHE_CONFIG_FILE" ]] || fail "LMCACHE_CONFIG_FILE is missing or empty: $LMCACHE_CONFIG_FILE"

runner=$TT_REPO/lmcache-t3knic/benchmarks/run_cross_node_ttft.sh
[[ -x "$runner" ]] || fail "cross-node runner is missing or not executable: $runner"

exec env \
    TT_REPO="$TT_REPO" \
    PLUGIN="$PLUGIN" \
    PLUGIN_PY="$PLUGIN_PY" \
    T3K_HOST="$T3K_HOST" \
    T3K_REPO="$T3K_REPO" \
    MODEL="$MODEL" \
    LINK="$LINK" \
    T3KNIC_INTERMESH_LINK_COUNT="$T3KNIC_INTERMESH_LINK_COUNT" \
    T3KNIC_SERVER_SHARDS="$T3KNIC_SERVER_SHARDS" \
    T3KNIC_WIRE_MODE="$T3KNIC_WIRE_MODE" \
    REQUESTS="$REQUESTS" \
    OSL="$OSL" \
    PROMPT_TOKENS="$PROMPT_TOKENS" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    LMCACHE_CONFIG_FILE="$LMCACHE_CONFIG_FILE" \
    OUT="$OUT" \
    "$runner"
