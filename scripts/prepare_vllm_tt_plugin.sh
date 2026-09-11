#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: prepare_vllm_tt_plugin.sh --plugin-dir PATH

Clone the pinned vLLM TT plugin when PATH does not exist, then apply the canonical patch bundle.
An existing checkout must be either the exact clean base or the already prepared target tree.
The script never resets, cleans, or overwrites an existing checkout.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/patch_bundle.sh
source "$SCRIPT_DIR/lib/patch_bundle.sh"

PATCH_BUNDLE_PROGRAM=prepare_vllm_tt_plugin.sh
PATCH_BUNDLE_SUCCESS_MARKER=PLUGIN_PATCH_BOOTSTRAP_PASS
PATCH_BUNDLE_ALLOW_PREPARED=1
PLUGIN_DIR=
MANIFEST_DIR="$SCRIPT_DIR/../third_party/vllm-tt-plugin"

patch_bundle_reject_git_path_overrides

while (($#)); do
    case "$1" in
        --plugin-dir)
            (($# >= 2)) || patch_bundle_fail "--plugin-dir requires a path"
            PLUGIN_DIR=$2
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

[[ -n "$PLUGIN_DIR" ]] || patch_bundle_fail "--plugin-dir is required"

REPOSITORY=$(tr -d '[:space:]' <"$MANIFEST_DIR/REPOSITORY")
[[ -n "$REPOSITORY" ]] || patch_bundle_fail "REPOSITORY must not be empty"
REVISION=$(patch_bundle_read_object_id "$MANIFEST_DIR/REVISION" REVISION)
PLUGIN_DIR=$(realpath -m "$PLUGIN_DIR")

if [[ ! -e "$PLUGIN_DIR" ]]; then
    [[ -d "$(dirname -- "$PLUGIN_DIR")" ]] ||
        patch_bundle_fail "plugin checkout parent does not exist: $(dirname -- "$PLUGIN_DIR")"
    git clone --no-checkout "$REPOSITORY" "$PLUGIN_DIR"
    git -C "$PLUGIN_DIR" checkout --detach "$REVISION"
elif [[ ! -d "$PLUGIN_DIR" ]]; then
    patch_bundle_fail "plugin checkout path is not a directory: $PLUGIN_DIR"
fi

patch_bundle_apply "$PLUGIN_DIR" "$MANIFEST_DIR"
