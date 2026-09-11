#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: generate_tt_metal_patches.sh --tt-metal-dir PATH

Regenerate the canonical functional patch bundle from a prepared tt-metal source index.
The source checkout must have no unstaged or untracked files. Its index is compared directly with
third_party/tt-metal/REVISION, so the script works before or after the source changes are committed.
EOF
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/patch_bundle.sh
source "$SCRIPT_DIR/lib/patch_bundle.sh"

PATCH_BUNDLE_PROGRAM=generate_tt_metal_patches.sh
REPOSITORY_ROOT=$(realpath "$SCRIPT_DIR/..")
MANIFEST_DIR="$REPOSITORY_ROOT/third_party/tt-metal"
TT_METAL_DIR=

fail() {
    patch_bundle_fail "$@"
}

patch_bundle_reject_git_path_overrides

while (($#)); do
    case "$1" in
        --tt-metal-dir)
            (($# >= 2)) || fail "--tt-metal-dir requires a path"
            TT_METAL_DIR=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[[ -n "$TT_METAL_DIR" ]] || fail "--tt-metal-dir is required"
[[ -d "$TT_METAL_DIR" ]] || fail "checkout does not exist: $TT_METAL_DIR"
TT_METAL_DIR=$(realpath "$TT_METAL_DIR")
TOP_LEVEL=$(git -C "$TT_METAL_DIR" rev-parse --show-toplevel 2>/dev/null) ||
    fail "not a Git checkout: $TT_METAL_DIR"
[[ "$(realpath "$TOP_LEVEL")" == "$TT_METAL_DIR" ]] || fail "path must name the checkout root"

REVISION=$(tr -d '[:space:]' <"$MANIFEST_DIR/REVISION")
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || fail "REVISION must contain one full Git object ID"
git -C "$TT_METAL_DIR" cat-file -e "$REVISION^{commit}" || fail "base revision is unavailable"
git -C "$TT_METAL_DIR" merge-base --is-ancestor "$REVISION" HEAD ||
    fail "base revision is not an ancestor of the source checkout"
SOURCE_HEAD=$(git -C "$TT_METAL_DIR" rev-parse --verify HEAD) ||
    fail "failed to inspect source HEAD"
git -C "$TT_METAL_DIR" diff --quiet --no-ext-diff -- || fail "source checkout has unstaged changes"
UNTRACKED_FILES=$(git -C "$TT_METAL_DIR" ls-files --others --exclude-standard) ||
    fail "failed to inspect source checkout"
[[ -z "$UNTRACKED_FILES" ]] || fail "source checkout has untracked files: $UNTRACKED_FILES"

PATCH_NAMES=(
    0001-static-intermesh-fabric.patch
    0002-cross-node-data-path-operations.patch
    0003-kvcache-protocol-device-runtime.patch
    0004-qwen3-runtime.patch
    0005-lmcache-t3knic-connector.patch
)
PRESERVED_MANIFEST_FILES=(
    LICENSE
    LICENSE_understanding.txt
    NOTICE
    README.md
    REVISION
)
declare -a GROUP_0=()
declare -a GROUP_1=()
declare -a GROUP_2=()
declare -a GROUP_3=()
declare -a GROUP_4=()

while IFS= read -r -d '' path; do
    case "$path" in
        tests/tt_metal/tt_fabric/custom_mesh_descriptors/*|\
        tt_metal/api/tt-metalium/experimental/fabric/*|\
        tt_metal/fabric/*|\
        tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h|\
        tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp|\
        ttnn/cpp/ttnn-nanobind/fabric.cpp)
            GROUP_0+=("$path")
            ;;
        ttnn/cpp/ttnn/operations/experimental/CMakeLists.txt|\
        ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp|\
        ttnn/cpp/ttnn/operations/experimental/moreh_paged_fill_cache_varlen/*|\
        ttnn/cpp/ttnn/operations/experimental/moreh_raw_byte_copy/*|\
        ttnn/sources.cmake)
            GROUP_1+=("$path")
            ;;
        tests/ttnn/unit_tests/gtests/sources.cmake|\
        tests/ttnn/unit_tests/gtests/test_kvcache_manager_*.cpp|\
        tt_metal/api/tt-metalium/distributed.hpp|\
        tt_metal/api/tt-metalium/tt_metal.hpp|\
        tt_metal/api/tt-metalium/tt_metal_profiler.hpp|\
        tt_metal/distributed/distributed.cpp|\
        tt_metal/impl/host_api/tt_metal.cpp|\
        tt_metal/impl/profiler/profiler.cpp|\
        tt_metal/llrt/tt_cluster.cpp|\
        tt_metal/llrt/tt_cluster.hpp|\
        ttnn/cpp/ttnn-nanobind/__init__.cpp|\
        ttnn/cpp/ttnn/operations/CMakeLists.txt|\
        ttnn/cpp/ttnn/operations/kvcache_manager/*|\
        ttnn/ttnn/__init__.py|\
        ttnn/ttnn/kvcache_manager/*)
            GROUP_2+=("$path")
            ;;
        models/common/models/qwen3_32b/*|\
        models/common/rmsnorm.py|\
        models/tt_transformers/tt/attention.py|\
        models/tt_transformers/tt/ccl.py|\
        models/tt_transformers/tt/distributed_norm.py|\
        models/tt_transformers/tt/generator.py|\
        models/tt_transformers/tt/model.py|\
        models/tt_transformers/tt/model_config.py)
            GROUP_3+=("$path")
            ;;
        lmcache-t3knic/*)
            GROUP_4+=("$path")
            ;;
        *)
            fail "changed path has no functional patch owner: $path"
            ;;
    esac
done < <(git -C "$TT_METAL_DIR" diff --cached --name-only --no-renames -z "$REVISION")

TARGET_TREE=$(git -C "$TT_METAL_DIR" write-tree)
BASE_TREE=$(git -C "$TT_METAL_DIR" rev-parse "$REVISION^{tree}")
[[ "$TARGET_TREE" != "$BASE_TREE" ]] || fail "source index contains no changes from the base"

validate_source_snapshot() {
    local actual_head
    local actual_tree
    local untracked_files

    actual_head=$(git -C "$TT_METAL_DIR" rev-parse --verify HEAD) ||
        fail "failed to revalidate source HEAD"
    [[ "$actual_head" == "$SOURCE_HEAD" ]] || fail "source HEAD changed during patch generation"
    git -C "$TT_METAL_DIR" merge-base --is-ancestor "$REVISION" "$actual_head" ||
        fail "base revision is no longer an ancestor of the source checkout"
    git -C "$TT_METAL_DIR" diff --quiet --no-ext-diff -- ||
        fail "source worktree changed during patch generation"
    untracked_files=$(git -C "$TT_METAL_DIR" ls-files --others --exclude-standard) ||
        fail "failed to revalidate source untracked files"
    [[ -z "$untracked_files" ]] ||
        fail "source checkout gained untracked files during patch generation: $untracked_files"
    actual_tree=$(git -C "$TT_METAL_DIR" write-tree) ||
        fail "failed to revalidate source index"
    [[ "$actual_tree" == "$TARGET_TREE" ]] ||
        fail "source index changed during patch generation: expected $TARGET_TREE, got $actual_tree"
}

THIRD_PARTY_DIR=$(dirname -- "$MANIFEST_DIR")
WORK_DIR=$(mktemp -d "$THIRD_PARTY_DIR/.tt-metal.generate.XXXXXX")
STAGED_MANIFEST="$WORK_DIR/tt-metal"
STAGED_PATCH_DIR="$STAGED_MANIFEST/patches"
VALIDATION_INDEX="$WORK_DIR/validation.index"
BACKUP_DIR=
BACKUP_MANIFEST=
SWAP_ACTIVE=0

cleanup_tree() {
    local directory=$1

    [[ -n "$directory" && -d "$directory" ]] || return
    find "$directory" -type f -delete 2>/dev/null || true
    find "$directory" -type l -delete 2>/dev/null || true
    find "$directory" -depth -type d -empty -delete 2>/dev/null || true
}

cleanup() {
    local exit_status=$?
    local failed_manifest
    local rollback_failed=0

    if [[ "$SWAP_ACTIVE" -eq 1 && -n "$BACKUP_MANIFEST" && -d "$BACKUP_MANIFEST" ]]; then
        if [[ -e "$MANIFEST_DIR" ]]; then
            failed_manifest="$BACKUP_DIR/failed-tt-metal"
            if ! mv -- "$MANIFEST_DIR" "$failed_manifest"; then
                echo "generate_tt_metal_patches.sh: failed to move incomplete manifest during rollback" >&2
                rollback_failed=1
            fi
        fi
        if [[ ! -e "$MANIFEST_DIR" ]] && ! mv -- "$BACKUP_MANIFEST" "$MANIFEST_DIR"; then
            echo "generate_tt_metal_patches.sh: failed to restore manifest from $BACKUP_MANIFEST" >&2
            rollback_failed=1
        fi
    fi

    cleanup_tree "$WORK_DIR"
    if [[ "$rollback_failed" -eq 0 ]]; then
        cleanup_tree "$BACKUP_DIR"
    elif [[ -n "$BACKUP_DIR" ]]; then
        echo "generate_tt_metal_patches.sh: recovery files remain at $BACKUP_DIR" >&2
        exit_status=1
    fi
    return "$exit_status"
}
trap cleanup EXIT

install -d -m 0755 "$STAGED_PATCH_DIR"
for manifest_file in "${PRESERVED_MANIFEST_FILES[@]}"; do
    install -m 0644 "$MANIFEST_DIR/$manifest_file" "$STAGED_MANIFEST/$manifest_file"
done

for index in "${!PATCH_NAMES[@]}"; do
    declare -n group="GROUP_$index"
    [[ "${#group[@]}" -gt 0 ]] || fail "functional patch group is empty: ${PATCH_NAMES[$index]}"
    git -C "$TT_METAL_DIR" diff \
        --cached \
        --binary \
        --full-index \
        --no-renames \
        "$REVISION" \
        -- "${group[@]}" >"$STAGED_PATCH_DIR/${PATCH_NAMES[$index]}"
    [[ -s "$STAGED_PATCH_DIR/${PATCH_NAMES[$index]}" ]] ||
        fail "generated patch is empty: ${PATCH_NAMES[$index]}"
done

printf '%s\n' "${PATCH_NAMES[@]}" >"$STAGED_PATCH_DIR/series"
(
    cd "$STAGED_PATCH_DIR"
    sha256sum "${PATCH_NAMES[@]}" >SHA256SUMS
)
printf '%s\n' "$TARGET_TREE" >"$STAGED_MANIFEST/TARGET_TREE"

GIT_INDEX_FILE="$VALIDATION_INDEX" git -C "$TT_METAL_DIR" read-tree "$REVISION"
for patch_name in "${PATCH_NAMES[@]}"; do
    GIT_INDEX_FILE="$VALIDATION_INDEX" git -C "$TT_METAL_DIR" apply \
        --cached "$STAGED_PATCH_DIR/$patch_name"
done
GENERATED_TREE=$(GIT_INDEX_FILE="$VALIDATION_INDEX" git -C "$TT_METAL_DIR" write-tree)
[[ "$GENERATED_TREE" == "$TARGET_TREE" ]] ||
    fail "generated series produced $GENERATED_TREE, expected $TARGET_TREE"

validate_source_snapshot
BACKUP_DIR=$(mktemp -d "$THIRD_PARTY_DIR/.tt-metal.backup.XXXXXX")
BACKUP_MANIFEST="$BACKUP_DIR/tt-metal"
SWAP_ACTIVE=1
mv -- "$MANIFEST_DIR" "$BACKUP_MANIFEST"
mv -- "$STAGED_MANIFEST" "$MANIFEST_DIR"
validate_source_snapshot
SWAP_ACTIVE=0

printf 'TT_METAL_PATCH_GENERATION_PASS patches=%d target_tree=%s\n' \
    "${#PATCH_NAMES[@]}" "$TARGET_TREE"
