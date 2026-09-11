#!/usr/bin/env bash

patch_bundle_fail() {
    echo "${PATCH_BUNDLE_PROGRAM:-patch bundle}: $*" >&2
    exit 1
}

patch_bundle_reject_git_path_overrides() {
    local variable_name

    for variable_name in \
        GIT_DIR \
        GIT_WORK_TREE \
        GIT_INDEX_FILE \
        GIT_COMMON_DIR \
        GIT_OBJECT_DIRECTORY \
        GIT_ALTERNATE_OBJECT_DIRECTORIES; do
        [[ -z "${!variable_name+x}" ]] ||
            patch_bundle_fail "Git path override must be unset: $variable_name"
    done
}

patch_bundle_read_object_id() {
    local manifest_file=$1
    local label=$2
    local object_id

    [[ -f "$manifest_file" ]] || patch_bundle_fail "missing manifest file: $manifest_file"
    object_id=$(tr -d '[:space:]' <"$manifest_file")
    [[ "$object_id" =~ ^[0-9a-f]{40}$ ]] ||
        patch_bundle_fail "$label must contain one full Git object ID"
    printf '%s\n' "$object_id"
}

patch_bundle_resolve_checkout() {
    local checkout=$1
    local top_level

    [[ -d "$checkout" ]] || patch_bundle_fail "checkout does not exist: $checkout"
    checkout=$(realpath "$checkout")
    top_level=$(git -C "$checkout" rev-parse --show-toplevel 2>/dev/null) ||
        patch_bundle_fail "not a Git checkout: $checkout"
    top_level=$(realpath "$top_level")
    [[ "$top_level" == "$checkout" ]] || patch_bundle_fail "path must name the checkout root: $checkout"
    printf '%s\n' "$checkout"
}

patch_bundle_validate_prepared() {
    local checkout=$1
    local manifest_dir=$2
    local revision
    local target_tree
    local actual_revision
    local actual_tree
    local untracked_files

    checkout=$(patch_bundle_resolve_checkout "$checkout")
    revision=$(patch_bundle_read_object_id "$manifest_dir/REVISION" REVISION)
    target_tree=$(patch_bundle_read_object_id "$manifest_dir/TARGET_TREE" TARGET_TREE)

    actual_revision=$(git -C "$checkout" rev-parse --verify HEAD) ||
        patch_bundle_fail "failed to inspect checkout revision"
    [[ "$actual_revision" == "$revision" ]] ||
        patch_bundle_fail "expected base $revision, got $actual_revision"

    actual_tree=$(git -C "$checkout" write-tree) ||
        patch_bundle_fail "failed to inspect checkout index"
    [[ "$actual_tree" == "$target_tree" ]] ||
        patch_bundle_fail "prepared tree mismatch: expected $target_tree, got $actual_tree"
    git -C "$checkout" diff --quiet --no-ext-diff -- ||
        patch_bundle_fail "worktree differs from the prepared index"
    untracked_files=$(git -C "$checkout" ls-files --others --exclude-standard) ||
        patch_bundle_fail "failed to inspect untracked files"
    [[ -z "$untracked_files" ]] ||
        patch_bundle_fail "prepared checkout contains untracked files: $untracked_files"
}

patch_bundle_apply() {
    local checkout=$1
    local manifest_dir=$2
    local patch_dir="$manifest_dir/patches"
    local revision
    local target_tree
    local checkout_status
    local actual_revision
    local actual_tree
    local patch_name
    local patch_path
    local digest
    local checksum_name
    local extra
    local index
    local preflight_dir
    local preflight_index
    local preflight_tree
    local untracked_files
    local apply_started=0
    local -a patch_names=()
    local -a patches=()
    local -a checksum_names=()
    local -a manifest_patch_files=()
    local -A seen_patches=()

    checkout=$(patch_bundle_resolve_checkout "$checkout")
    [[ -d "$patch_dir" ]] || patch_bundle_fail "missing patch directory: $patch_dir"
    [[ -f "$patch_dir/series" ]] || patch_bundle_fail "missing patch series: $patch_dir/series"
    [[ -f "$patch_dir/SHA256SUMS" ]] ||
        patch_bundle_fail "missing patch checksums: $patch_dir/SHA256SUMS"
    revision=$(patch_bundle_read_object_id "$manifest_dir/REVISION" REVISION)
    target_tree=$(patch_bundle_read_object_id "$manifest_dir/TARGET_TREE" TARGET_TREE)

    while IFS= read -r patch_name || [[ -n "$patch_name" ]]; do
        [[ -z "$patch_name" || "$patch_name" == \#* ]] && continue
        [[ "$patch_name" != */* && "$patch_name" == *.patch ]] ||
            patch_bundle_fail "invalid patch name in series: $patch_name"
        [[ -z "${seen_patches[$patch_name]:-}" ]] ||
            patch_bundle_fail "duplicate patch in series: $patch_name"
        [[ -f "$patch_dir/$patch_name" ]] || patch_bundle_fail "missing patch: $patch_name"
        seen_patches[$patch_name]=1
        patch_names+=("$patch_name")
        patches+=("$patch_dir/$patch_name")
    done <"$patch_dir/series"
    [[ "${#patches[@]}" -gt 0 ]] || patch_bundle_fail "patch series is empty"

    shopt -s nullglob
    manifest_patch_files=("$patch_dir"/*.patch)
    shopt -u nullglob
    [[ "${#manifest_patch_files[@]}" -eq "${#patches[@]}" ]] ||
        patch_bundle_fail "patch directory contains files not listed in series"
    for patch_path in "${manifest_patch_files[@]}"; do
        patch_name=$(basename "$patch_path")
        [[ -n "${seen_patches[$patch_name]:-}" ]] ||
            patch_bundle_fail "patch is not listed in series: $patch_name"
    done

    while read -r digest checksum_name extra || [[ -n "${digest:-}${checksum_name:-}${extra:-}" ]]; do
        [[ -z "${digest:-}" || "$digest" == \#* ]] && continue
        [[ "$digest" =~ ^[0-9a-f]{64}$ && -n "${checksum_name:-}" && -z "${extra:-}" ]] ||
            patch_bundle_fail "invalid SHA256SUMS entry"
        [[ "$checksum_name" != */* && "$checksum_name" == *.patch ]] ||
            patch_bundle_fail "invalid patch name in SHA256SUMS: $checksum_name"
        checksum_names+=("$checksum_name")
    done <"$patch_dir/SHA256SUMS"
    [[ "${#checksum_names[@]}" -eq "${#patch_names[@]}" ]] ||
        patch_bundle_fail "series and SHA256SUMS list different patch counts"
    for index in "${!patch_names[@]}"; do
        [[ "${patch_names[$index]}" == "${checksum_names[$index]}" ]] ||
            patch_bundle_fail "series and SHA256SUMS differ at entry $((index + 1))"
    done

    (
        cd "$patch_dir"
        sha256sum --check SHA256SUMS
    ) || patch_bundle_fail "patch checksum verification failed"

    checkout_status=$(git -C "$checkout" status --porcelain=v1 --untracked-files=all) ||
        patch_bundle_fail "failed to inspect checkout status before patching"
    if [[ -n "$checkout_status" ]]; then
        if [[ "${PATCH_BUNDLE_ALLOW_PREPARED:-0}" == 1 ]]; then
            patch_bundle_validate_prepared "$checkout" "$manifest_dir"
            echo "${PATCH_BUNDLE_SUCCESS_MARKER:-PATCH_BUNDLE_PASS} revision=$revision target_tree=$target_tree checkout=$checkout already_prepared=1"
            return
        fi
        patch_bundle_fail "checkout must be clean before patching"
    fi

    actual_revision=$(git -C "$checkout" rev-parse --verify HEAD) ||
        patch_bundle_fail "failed to inspect checkout revision"
    [[ "$actual_revision" == "$revision" ]] ||
        patch_bundle_fail "expected base $revision, got $actual_revision"

    preflight_dir=$(mktemp -d)
    preflight_index="$preflight_dir/index"
    patch_bundle_cleanup() {
        local exit_status=$?

        rm -f -- "$preflight_index"
        rmdir -- "$preflight_dir" 2>/dev/null || true
        if [[ "$exit_status" -ne 0 && "$apply_started" -eq 1 ]]; then
            echo "${PATCH_BUNDLE_PROGRAM:-patch bundle}: patching did not complete; checkout may be partially patched" >&2
        fi
        return "$exit_status"
    }
    trap patch_bundle_cleanup EXIT

    GIT_INDEX_FILE="$preflight_index" git -C "$checkout" read-tree "$revision"
    for patch_path in "${patches[@]}"; do
        GIT_INDEX_FILE="$preflight_index" git -C "$checkout" apply --cached --check "$patch_path"
        GIT_INDEX_FILE="$preflight_index" git -C "$checkout" apply --cached "$patch_path"
    done
    preflight_tree=$(GIT_INDEX_FILE="$preflight_index" git -C "$checkout" write-tree)
    [[ "$preflight_tree" == "$target_tree" ]] ||
        patch_bundle_fail "preflight tree mismatch: expected $target_tree, got $preflight_tree"

    checkout_status=$(git -C "$checkout" status --porcelain=v1 --untracked-files=all) ||
        patch_bundle_fail "failed to inspect checkout status after preflight"
    [[ -z "$checkout_status" ]] || patch_bundle_fail "checkout changed during preflight"
    [[ "$(git -C "$checkout" rev-parse --verify HEAD)" == "$revision" ]] ||
        patch_bundle_fail "checkout revision changed during preflight"

    apply_started=1
    for patch_path in "${patches[@]}"; do
        git -C "$checkout" apply --index "$patch_path"
    done
    [[ "$(git -C "$checkout" rev-parse --verify HEAD)" == "$revision" ]] ||
        patch_bundle_fail "checkout revision changed during patching"
    actual_tree=$(git -C "$checkout" write-tree)
    [[ "$actual_tree" == "$target_tree" ]] ||
        patch_bundle_fail "applied tree mismatch: expected $target_tree, got $actual_tree"
    git -C "$checkout" diff --quiet --no-ext-diff -- ||
        patch_bundle_fail "worktree differs from the applied index"
    untracked_files=$(git -C "$checkout" ls-files --others --exclude-standard) ||
        patch_bundle_fail "failed to inspect untracked files after patching"
    [[ -z "$untracked_files" ]] ||
        patch_bundle_fail "unexpected untracked files appeared during patching"
    apply_started=0

    trap - EXIT
    patch_bundle_cleanup
    echo "${PATCH_BUNDLE_SUCCESS_MARKER:-PATCH_BUNDLE_PASS} revision=$revision target_tree=$target_tree checkout=$checkout"
}
