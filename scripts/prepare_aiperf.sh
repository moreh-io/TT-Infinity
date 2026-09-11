#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: prepare_aiperf.sh --aiperf-dir PATH

Clone the pinned SemiAnalysisAI AIPerf fork when PATH does not exist, then apply the canonical
local-tokenizer patch bundle. An existing checkout must be either the exact clean base or the
already prepared target tree. The script never resets, cleans, or overwrites an existing checkout.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/patch_bundle.sh
source "$SCRIPT_DIR/lib/patch_bundle.sh"

PATCH_BUNDLE_PROGRAM=prepare_aiperf.sh
PATCH_BUNDLE_SUCCESS_MARKER=AIPERF_PATCH_BOOTSTRAP_PASS
PATCH_BUNDLE_ALLOW_PREPARED=1
AIPERF_DIR=
MANIFEST_DIR="$SCRIPT_DIR/../third_party/aiperf"

patch_bundle_reject_git_path_overrides

while (($#)); do
    case "$1" in
        --aiperf-dir)
            (($# >= 2)) || patch_bundle_fail "--aiperf-dir requires a path"
            AIPERF_DIR=$2
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

[[ -n "$AIPERF_DIR" ]] || patch_bundle_fail "--aiperf-dir is required"

REPOSITORY=$(tr -d '[:space:]' <"$MANIFEST_DIR/REPOSITORY")
[[ -n "$REPOSITORY" ]] || patch_bundle_fail "REPOSITORY must not be empty"
REVISION=$(patch_bundle_read_object_id "$MANIFEST_DIR/REVISION" REVISION)
AIPERF_DIR=$(realpath -m "$AIPERF_DIR")

if [[ ! -e "$AIPERF_DIR" ]]; then
    [[ -d "$(dirname -- "$AIPERF_DIR")" ]] ||
        patch_bundle_fail "AIPerf checkout parent does not exist: $(dirname -- "$AIPERF_DIR")"
    git clone --no-checkout "$REPOSITORY" "$AIPERF_DIR"
    git -C "$AIPERF_DIR" checkout --detach "$REVISION"
elif [[ ! -d "$AIPERF_DIR" ]]; then
    patch_bundle_fail "AIPerf checkout path is not a directory: $AIPERF_DIR"
fi

patch_bundle_apply "$AIPERF_DIR" "$MANIFEST_DIR"
