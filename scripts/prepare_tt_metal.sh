#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: prepare_tt_metal.sh --tt-metal-dir PATH

Apply the pinned tt-metal patch series to an exact, clean checkout.
The script does not clone, reset, clean, build, or delete the checkout.
If patching fails or is interrupted, the checkout may be partially patched.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/patch_bundle.sh
source "$SCRIPT_DIR/lib/patch_bundle.sh"

PATCH_BUNDLE_PROGRAM=prepare_tt_metal.sh
PATCH_BUNDLE_SUCCESS_MARKER=PATCH_BOOTSTRAP_PASS
TT_METAL_DIR=

patch_bundle_reject_git_path_overrides

while (($#)); do
    case "$1" in
        --tt-metal-dir)
            (($# >= 2)) || patch_bundle_fail "--tt-metal-dir requires a path"
            TT_METAL_DIR=$2
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

[[ -n "$TT_METAL_DIR" ]] || patch_bundle_fail "--tt-metal-dir is required"

patch_bundle_apply "$TT_METAL_DIR" "$SCRIPT_DIR/../third_party/tt-metal"
