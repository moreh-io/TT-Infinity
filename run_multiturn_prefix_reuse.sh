#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage_path=docs/usage.md

fail() {
    printf 'ERROR: %s\nSee %s in this checkout.\n' "$1" "$usage_path" >&2
    exit 2
}

missing=()
for name in TT_METAL_ROOT PLUGIN PLUGIN_VENV AIPERF_VENV HF_HOME MODEL OUT_MULTITURN; do
    [[ -n "${!name-}" ]] || missing+=("$name")
done

KVM_DISABLE=${KVM_DISABLE:-0}
if [[ "$KVM_DISABLE" != 1 ]]; then
    for name in T3K_HOST; do
        [[ -n "${!name-}" ]] || missing+=("$name")
    done
fi
if ((${#missing[@]})); then
    printf 'ERROR: missing required configuration:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    printf 'See %s in this checkout.\n' "$usage_path" >&2
    exit 2
fi

TT_REPO=${TT_REPO:-$TT_METAL_ROOT}
PLUGIN_PY=${PLUGIN_PY:-$PLUGIN_VENV/bin/python}
AIPERF_CLI=${AIPERF_CLI:-$AIPERF_VENV/bin/aiperf}
T3K_REPO=${T3K_REPO:-$TT_REPO}
T3K_PY=${T3K_PY:-$T3K_REPO/python_env/bin/python3}
SHARED=${SHARED:-$HOME/t3knic}
LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE:-$SCRIPT_DIR/configs/lmcache_qwen3_32b.yaml}
MULTITURN_INPUT_DIR=${MULTITURN_INPUT_DIR:-$SCRIPT_DIR/benchmarks/multiturn/datasets}
LINK=1

[[ -d "$TT_REPO" ]] || fail "TT_REPO does not exist: $TT_REPO"
[[ -d "$PLUGIN" ]] || fail "PLUGIN does not exist: $PLUGIN"
[[ -x "$PLUGIN_PY" ]] || fail "PLUGIN_PY is not executable: $PLUGIN_PY"
[[ -x "$AIPERF_CLI" ]] || fail "AIPERF_CLI is not executable: $AIPERF_CLI"
[[ -e "$MODEL" ]] || fail "MODEL does not exist: $MODEL"
[[ -d "$MULTITURN_INPUT_DIR" ]] || fail "MULTITURN_INPUT_DIR does not exist: $MULTITURN_INPUT_DIR"
compgen -G "$MULTITURN_INPUT_DIR/*.json" >/dev/null || fail "MULTITURN_INPUT_DIR contains no JSON traces: $MULTITURN_INPUT_DIR"
if [[ "$KVM_DISABLE" != 1 ]]; then
    [[ -s "$LMCACHE_CONFIG_FILE" ]] || fail "LMCACHE_CONFIG_FILE is missing or empty: $LMCACHE_CONFIG_FILE"
fi
[[ ! -e "$OUT_MULTITURN" || -d "$OUT_MULTITURN" ]] || fail "OUT_MULTITURN exists and is not a directory: $OUT_MULTITURN"
if [[ -e "$OUT_MULTITURN" ]] && [[ -n "$(find "$OUT_MULTITURN" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "OUT_MULTITURN already contains results; choose a new directory: $OUT_MULTITURN"
fi

runner=$TT_REPO/lmcache-t3knic/benchmarks/run_cross_node_ttft.sh
[[ -x "$runner" ]] || fail "cross-node runner is missing or not executable: $runner"

# BENCHMARK_DRIVER and AGENTX_* are the delegated tt-metal runner's external interface.
set +e
env \
    TT_REPO="$TT_REPO" \
    PLUGIN="$PLUGIN" \
    PLUGIN_PY="$PLUGIN_PY" \
    T3K_HOST="${T3K_HOST:-}" \
    T3K_REPO="$T3K_REPO" \
    T3K_PY="$T3K_PY" \
    MODEL="$MODEL" \
    SHARED="$SHARED" \
    LINK="$LINK" \
    T3KNIC_INTERMESH_LINK_COUNT="${T3KNIC_INTERMESH_LINK_COUNT:-2}" \
    T3KNIC_SERVER_SHARDS="${T3KNIC_SERVER_SHARDS:-8}" \
    T3KNIC_WIRE_MODE="${T3KNIC_WIRE_MODE:-per_chip}" \
    LMCACHE_CONFIG_FILE="$LMCACHE_CONFIG_FILE" \
    KVM_DISABLE="$KVM_DISABLE" \
    BENCHMARK_DRIVER=agentx \
    AIPERF_CLI="$AIPERF_CLI" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
    PYTHONNOUSERSITE=1 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" \
    PREWARM_RESUMED_PREFILL="${PREWARM_RESUMED_PREFILL:-1}" \
    PREWARM_RESUMED_PREFIX_TOKENS="${PREWARM_RESUMED_PREFIX_TOKENS:-4096}" \
    PREWARM_RESUMED_TAIL_TOKENS="${PREWARM_RESUMED_TAIL_TOKENS:-1024}" \
    PREWARM_RESUMED_DELAY="${PREWARM_RESUMED_DELAY:-1}" \
    CONC="${CONC:-1}" \
    DURATION="${DURATION:-90}" \
    AGENTX_REQUEST_COUNT="${MULTITURN_REQUEST_COUNT:-5}" \
    AGENTX_DATASET_ENTRIES="${MULTITURN_DATASET_ENTRIES:-1}" \
    AGENTX_WARMUP_REQUESTS_PER_LANE="${MULTITURN_WARMUP_REQUESTS_PER_LANE:-0}" \
    AGENTX_TRAJECTORY_START_MIN_RATIO="${MULTITURN_TRAJECTORY_START_MIN_RATIO:-0}" \
    AGENTX_TRAJECTORY_START_MAX_RATIO="${MULTITURN_TRAJECTORY_START_MAX_RATIO:-0}" \
    AGENTX_INPUT_DIR="$MULTITURN_INPUT_DIR" \
    OUT="$OUT_MULTITURN" \
    "$runner"
runner_rc=$?
set -e

records=$OUT_MULTITURN/aiperf_artifacts/profile_export.jsonl
if [[ -s "$records" ]]; then
    "$AIPERF_VENV/bin/python" - "$records" <<'PY' || printf 'WARNING: failed to summarize multi-turn results\n' >&2
import json
import sys

print("\nMulti-turn prefix-reuse timings")
print(f"{'Turn':>4}  {'ISL':>6}  {'TTFT (s)':>10}")
for line in open(sys.argv[1], encoding="utf-8"):
    record = json.loads(line)
    turn = record["metadata"]["turn_index"] + 1
    isl = record["metrics"]["input_sequence_length"]["value"]
    ttft_ms = record["metrics"]["time_to_first_token"]["value"]
    print(f"{turn:>4}  {isl:>6}  {ttft_ms / 1000:>10.3f}")
PY
fi

exit "$runner_rc"
