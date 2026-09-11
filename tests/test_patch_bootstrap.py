# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess


_REPOSITORY = Path(__file__).resolve().parents[1]
_MANIFEST = _REPOSITORY / "third_party" / "tt-metal"
_PATCHES = _MANIFEST / "patches"
_SCRIPT = _REPOSITORY / "scripts" / "prepare_tt_metal.sh"
_PATCH_BUNDLE_LIBRARY = _REPOSITORY / "scripts" / "lib" / "patch_bundle.sh"
_PLUGIN_MANIFEST = _REPOSITORY / "third_party" / "vllm-tt-plugin"
_PLUGIN_PATCHES = _PLUGIN_MANIFEST / "patches"
_PLUGIN_SCRIPT = _REPOSITORY / "scripts" / "prepare_vllm_tt_plugin.sh"
_AIPERF_MANIFEST = _REPOSITORY / "third_party" / "aiperf"
_AIPERF_PATCHES = _AIPERF_MANIFEST / "patches"
_AIPERF_SCRIPT = _REPOSITORY / "scripts" / "prepare_aiperf.sh"
_SETUP_SCRIPT = _REPOSITORY / "scripts" / "setup_vllm_tt_plugin.sh"
_GENERATOR_SCRIPT = _REPOSITORY / "scripts" / "generate_tt_metal_patches.sh"
_TT_METAL_LEGAL_FILES = ("LICENSE", "LICENSE_understanding.txt", "NOTICE")
_GIT_PATH_OVERRIDE_NAMES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable_name in _GIT_PATH_OVERRIDE_NAMES:
        environment.pop(variable_name, None)
    return environment


def _run_script(
    script: Path, *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=_clean_environment() if env is None else env,
    )


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return _run_script(_SCRIPT, *arguments)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _new_repository(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Patch Test")
    _git(path, "config", "user.email", "patch-test@example.invalid")
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "--quiet", "-m", "base")


def _synthetic_bootstrap(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repository = tmp_path / "checkout"
    _new_repository(repository)
    revision = _git_output(repository, "rev-parse", "HEAD")

    (repository / "tracked.txt").write_text("first\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "--quiet", "-m", "first")
    first_revision = _git_output(repository, "rev-parse", "HEAD")

    (repository / "tracked.txt").write_text("second\n", encoding="utf-8")
    (repository / "added.txt").write_text("added\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt", "added.txt")
    _git(repository, "commit", "--quiet", "-m", "second")
    final_revision = _git_output(repository, "rev-parse", "HEAD")
    target_tree = _git_output(repository, "rev-parse", "HEAD^{tree}")

    patch_arguments = ("diff", "--binary", "--full-index", "--no-renames")
    first_patch = subprocess.run(
        ["git", "-C", str(repository), *patch_arguments, revision, first_revision],
        check=True,
        capture_output=True,
    ).stdout
    second_patch = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            *patch_arguments,
            first_revision,
            final_revision,
        ],
        check=True,
        capture_output=True,
    ).stdout
    _git(repository, "checkout", "--detach", "--quiet", revision)

    bundle = tmp_path / "bundle"
    script = bundle / "scripts" / "prepare_tt_metal.sh"
    patch_bundle_library = bundle / "scripts" / "lib" / "patch_bundle.sh"
    patches = bundle / "third_party" / "tt-metal" / "patches"
    patch_bundle_library.parent.mkdir(parents=True)
    patches.mkdir(parents=True)
    shutil.copy2(_SCRIPT, script)
    shutil.copy2(_PATCH_BUNDLE_LIBRARY, patch_bundle_library)
    (patches.parent / "REVISION").write_text(f"{revision}\n", encoding="utf-8")
    (patches.parent / "TARGET_TREE").write_text(f"{target_tree}\n", encoding="utf-8")

    patch_payloads = {
        "0001-first.patch": first_patch,
        "0002-second.patch": second_patch,
    }
    for filename, payload in patch_payloads.items():
        (patches / filename).write_bytes(payload)
    (patches / "series").write_text(
        "".join(f"{filename}\n" for filename in patch_payloads), encoding="utf-8"
    )
    (patches / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(payload).hexdigest()}  {filename}\n"
            for filename, payload in patch_payloads.items()
        ),
        encoding="utf-8",
    )
    return script, repository, revision, target_tree


def _synthetic_plugin_bootstrap(
    tmp_path: Path,
) -> tuple[Path, Path, str, str]:
    upstream = tmp_path / "plugin-upstream"
    _new_repository(upstream)
    revision = _git_output(upstream, "rev-parse", "HEAD")

    (upstream / "tracked.txt").write_text("patched plugin\n", encoding="utf-8")
    _git(upstream, "add", "tracked.txt")
    _git(upstream, "commit", "--quiet", "-m", "plugin integration")
    target_tree = _git_output(upstream, "rev-parse", "HEAD^{tree}")
    patch_payload = subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "diff",
            "--binary",
            "--full-index",
            "--no-renames",
            revision,
            "HEAD",
        ],
        check=True,
        capture_output=True,
    ).stdout

    bundle = tmp_path / "plugin-bundle"
    script = bundle / "scripts" / "prepare_vllm_tt_plugin.sh"
    patch_bundle_library = bundle / "scripts" / "lib" / "patch_bundle.sh"
    manifest = bundle / "third_party" / "vllm-tt-plugin"
    patches = manifest / "patches"
    patch_bundle_library.parent.mkdir(parents=True)
    patches.mkdir(parents=True)
    shutil.copy2(_PLUGIN_SCRIPT, script)
    shutil.copy2(_PATCH_BUNDLE_LIBRARY, patch_bundle_library)

    patch_name = "0001-plugin.patch"
    (manifest / "REPOSITORY").write_text(f"{upstream}\n", encoding="utf-8")
    (manifest / "REVISION").write_text(f"{revision}\n", encoding="utf-8")
    (manifest / "TARGET_TREE").write_text(f"{target_tree}\n", encoding="utf-8")
    (patches / patch_name).write_bytes(patch_payload)
    (patches / "series").write_text(f"{patch_name}\n", encoding="utf-8")
    (patches / "SHA256SUMS").write_text(
        f"{hashlib.sha256(patch_payload).hexdigest()}  {patch_name}\n",
        encoding="utf-8",
    )
    return script, tmp_path / "plugin-checkout", revision, target_tree


def _synthetic_patch_generator(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str]:
    source = tmp_path / "tt-metal-source"
    _new_repository(source)
    base_files = (
        "tt_metal/fabric/fabric.cpp",
        "ttnn/sources.cmake",
        "ttnn/cpp/ttnn/operations/kvcache_manager/common/config.hpp",
        "models/common/rmsnorm.py",
        "lmcache-t3knic/README.md",
    )
    for relative_path in base_files:
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("base\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "--quiet", "-m", "functional patch base")
    revision = _git_output(source, "rev-parse", "HEAD")

    for relative_path in base_files:
        if relative_path == "models/common/rmsnorm.py":
            continue
        (source / relative_path).write_text("target\n", encoding="utf-8")
    renamed_path = "models/common/models/qwen3_32b/renamed_rmsnorm.py"
    (source / renamed_path).parent.mkdir(parents=True)
    _git(source, "mv", "models/common/rmsnorm.py", renamed_path)
    _git(source, "add", ".")
    _git(source, "commit", "--quiet", "-m", "functional patch target")
    target_tree = _git_output(source, "rev-parse", "HEAD^{tree}")

    bundle = tmp_path / "generator-bundle"
    script = bundle / "scripts" / "generate_tt_metal_patches.sh"
    patch_bundle_library = bundle / "scripts" / "lib" / "patch_bundle.sh"
    manifest = bundle / "third_party" / "tt-metal"
    patch_bundle_library.parent.mkdir(parents=True)
    (manifest / "patches").mkdir(parents=True)
    shutil.copy2(_GENERATOR_SCRIPT, script)
    shutil.copy2(_PATCH_BUNDLE_LIBRARY, patch_bundle_library)
    (manifest / "REVISION").write_text(f"{revision}\n", encoding="utf-8")
    (manifest / "README.md").write_text("fixture manifest\n", encoding="utf-8")
    for filename in _TT_METAL_LEGAL_FILES:
        (manifest / filename).write_text(
            f"fixture {filename}\n", encoding="utf-8"
        )
    return script, source, manifest, target_tree


def _git_wrapper(path: Path) -> Path:
    wrapper = path / "git"
    wrapper.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

actual_apply=0
if [[ -z "${GIT_INDEX_FILE:-}" ]]; then
    for argument in "$@"; do
        [[ "$argument" == apply ]] && actual_apply=1
    done
fi

if [[ "$actual_apply" -eq 1 ]]; then
    count=0
    [[ ! -f "$APPLY_COUNT_FILE" ]] || read -r count <"$APPLY_COUNT_FILE"
    count=$((count + 1))
    printf '%s\n' "$count" >"$APPLY_COUNT_FILE"
    if [[ "${BOOTSTRAP_TEST_MODE:-}" == fail-second && "$count" -eq 2 ]]; then
        exit 86
    fi
fi

if [[ "${BOOTSTRAP_TEST_MODE:-}" == fail-final-untracked ]]; then
    for argument in "$@"; do
        [[ "$argument" != --others ]] || exit 87
    done
fi

"$REAL_GIT" "$@"
status=$?
if [[ "$actual_apply" -eq 1 && "${BOOTSTRAP_TEST_MODE:-}" == dirty-after-second && "$count" -eq 2 ]]; then
    printf 'concurrent change\n' >"$MUTATE_REPOSITORY/tracked.txt"
fi
exit "$status"
""",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def _wrapper_environment(tmp_path: Path, repository: Path, mode: str) -> dict[str, str]:
    wrapper_directory = tmp_path / "bin"
    wrapper_directory.mkdir()
    _git_wrapper(wrapper_directory)
    real_git = shutil.which("git")
    assert real_git is not None

    environment = _clean_environment()
    environment.update(
        {
            "APPLY_COUNT_FILE": str(tmp_path / "apply-count"),
            "BOOTSTRAP_TEST_MODE": mode,
            "MUTATE_REPOSITORY": str(repository),
            "PATH": f"{wrapper_directory}:{environment['PATH']}",
            "REAL_GIT": real_git,
        }
    )
    return environment


def test_patch_manifest_hashes_and_order() -> None:
    revision = (_MANIFEST / "REVISION").read_text(encoding="utf-8").strip()
    target_tree = (_MANIFEST / "TARGET_TREE").read_text(encoding="utf-8").strip()
    assert len(revision) == 40 and int(revision, 16) >= 0
    assert len(target_tree) == 40 and int(target_tree, 16) >= 0

    series = [
        line
        for line in (_PATCHES / "series").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    expected = {}
    for line in (_PATCHES / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split()
        expected[filename] = digest

    assert series == list(expected)
    assert len(series) == len(set(series)) > 0
    assert set(series) == {patch.name for patch in _PATCHES.glob("*.patch")}
    for filename in series:
        patch = _PATCHES / filename
        assert patch.is_file()
        assert hashlib.sha256(patch.read_bytes()).hexdigest() == expected[filename]


def test_plugin_patch_manifest_hashes_and_order() -> None:
    repository = (_PLUGIN_MANIFEST / "REPOSITORY").read_text(encoding="utf-8").strip()
    revision = (_PLUGIN_MANIFEST / "REVISION").read_text(encoding="utf-8").strip()
    target_tree = (_PLUGIN_MANIFEST / "TARGET_TREE").read_text(encoding="utf-8").strip()
    vllm_version = (
        (_PLUGIN_MANIFEST / "VLLM_VERSION").read_text(encoding="utf-8").strip()
    )
    assert repository == "https://github.com/tenstorrent/vllm-tt-plugin.git"
    assert len(revision) == 40 and int(revision, 16) >= 0
    assert len(target_tree) == 40 and int(target_tree, 16) >= 0
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", vllm_version)

    series = [
        line
        for line in (_PLUGIN_PATCHES / "series")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    ]
    expected = {}
    for line in (
        (_PLUGIN_PATCHES / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    ):
        digest, filename = line.split()
        expected[filename] = digest

    assert series == list(expected)
    assert len(series) == len(set(series)) > 0
    assert set(series) == {patch.name for patch in _PLUGIN_PATCHES.glob("*.patch")}
    for filename in series:
        patch = _PLUGIN_PATCHES / filename
        assert hashlib.sha256(patch.read_bytes()).hexdigest() == expected[filename]


def test_aiperf_patch_manifest_hashes_and_order() -> None:
    repository = (_AIPERF_MANIFEST / "REPOSITORY").read_text(encoding="utf-8").strip()
    revision = (_AIPERF_MANIFEST / "REVISION").read_text(encoding="utf-8").strip()
    target_tree = (_AIPERF_MANIFEST / "TARGET_TREE").read_text(encoding="utf-8").strip()
    assert repository == "https://github.com/SemiAnalysisAI/aiperf.git"
    assert len(revision) == 40 and int(revision, 16) >= 0
    assert len(target_tree) == 40 and int(target_tree, 16) >= 0

    series = [
        line
        for line in (_AIPERF_PATCHES / "series").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    expected = {}
    for line in (_AIPERF_PATCHES / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split()
        expected[filename] = digest

    assert series == list(expected)
    assert len(series) == len(set(series)) > 0
    assert set(series) == {patch.name for patch in _AIPERF_PATCHES.glob("*.patch")}
    for filename in series:
        patch = _AIPERF_PATCHES / filename
        assert hashlib.sha256(patch.read_bytes()).hexdigest() == expected[filename]


def test_patch_payloads_exclude_deployment_specific_markers() -> None:
    safe_wrapper_pattern = re.compile(rb"(?<![A-Za-z0-9_])_safe(?![A-Za-z0-9_])")
    forbidden_markers = (
        b"T3K_LINK_" + b"SAFE_ACTIVE",
        b"moreh-" + b"lock",
        b"ttdev" + b"26",
        b"ttdev" + b"34",
        b"/root/" + b"tt-metal",
        b"/work" + b"space/",
    )

    for patch in _PATCHES.glob("*.patch"):
        payload = patch.read_bytes()
        error = f"{patch.name} contains a deployment-specific marker"
        assert safe_wrapper_pattern.search(payload) is None, error
        for marker in forbidden_markers:
            assert marker not in payload, error
        assert b"tt-metal-memory-extension" not in payload, error


def test_qwen_runner_plugin_python_guard_parses_independently() -> None:
    connector_patch = (
        _PATCHES / "0005-lmcache-t3knic-connector.patch"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"^\+PLUGIN_PY=\$\{PLUGIN_PY:\?.+\}$",
        connector_patch,
        flags=re.MULTILINE,
    )
    assert match is not None

    environment = _clean_environment()
    environment["PLUGIN_PY"] = "/fixture/plugin-env/bin/python"
    assignment = match.group(0)[1:]
    result = subprocess.run(
        ["bash", "-c", f'{assignment}\nprintf "%s\\n" "$PLUGIN_PY"'],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == environment["PLUGIN_PY"]


def test_qwen_runner_patch_length_and_shell_syntax() -> None:
    connector_patch = (
        _PATCHES / "0005-lmcache-t3knic-connector.patch"
    ).read_text(encoding="utf-8")
    runner_patch = connector_patch.split(
        "diff --git a/lmcache-t3knic/benchmarks/run_cross_node_ttft.sh ", 1
    )[1].split("diff --git ", 1)[0]
    match = re.search(
        r"@@ -0,0 \+1,(\d+) @@\n((?:\+.*\n)+)",
        runner_patch,
    )
    assert match is not None
    declared_lines = int(match.group(1))
    script = "".join(line[1:] for line in match.group(2).splitlines(keepends=True))
    assert len(script.splitlines()) == declared_lines

    result = subprocess.run(
        ["bash", "-n"], input=script, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_prepare_plugin_clones_applies_and_accepts_prepared_tree(
    tmp_path: Path,
) -> None:
    script, checkout, revision, target_tree = _synthetic_plugin_bootstrap(tmp_path)

    first = _run_script(script, "--plugin-dir", str(checkout))
    assert first.returncode == 0, first.stderr
    assert "PLUGIN_PATCH_BOOTSTRAP_PASS" in first.stdout
    assert _git_output(checkout, "rev-parse", "HEAD") == revision
    assert _git_output(checkout, "write-tree") == target_tree
    assert _git_output(checkout, "diff", "--name-only") == ""

    second = _run_script(script, "--plugin-dir", str(checkout))
    assert second.returncode == 0, second.stderr
    assert "already_prepared=1" in second.stdout
    assert _git_output(checkout, "write-tree") == target_tree


def test_patch_generator_handles_renames_without_losing_the_source_path(
    tmp_path: Path,
) -> None:
    script, source, manifest, target_tree = _synthetic_patch_generator(tmp_path)

    result = _run_script(script, "--tt-metal-dir", str(source))

    assert result.returncode == 0, result.stderr
    assert "TT_METAL_PATCH_GENERATION_PASS" in result.stdout
    assert (manifest / "TARGET_TREE").read_text(encoding="utf-8").strip() == target_tree
    qwen_patch = (manifest / "patches" / "0004-qwen3-runtime.patch").read_text(
        encoding="utf-8"
    )
    assert "a/models/common/rmsnorm.py" in qwen_patch
    assert "b/models/common/models/qwen3_32b/renamed_rmsnorm.py" in qwen_patch
    for filename in _TT_METAL_LEGAL_FILES:
        assert (manifest / filename).read_text(encoding="utf-8") == (
            f"fixture {filename}\n"
        )
    assert not list(manifest.parent.glob(".tt-metal.*"))


def test_patch_generator_rejects_inherited_git_path_override(tmp_path: Path) -> None:
    script, source, manifest, _ = _synthetic_patch_generator(tmp_path)
    environment = _clean_environment()
    environment["GIT_INDEX_FILE"] = str(tmp_path / "alternate-index")

    result = _run_script(
        script,
        "--tt-metal-dir",
        str(source),
        env=environment,
    )

    assert result.returncode != 0
    assert "Git path override must be unset: GIT_INDEX_FILE" in result.stderr
    assert not (manifest / "TARGET_TREE").exists()
    assert not list(manifest.parent.glob(".tt-metal.*"))


def test_patch_generator_restores_previous_manifest_when_swap_fails(
    tmp_path: Path,
) -> None:
    script, source, manifest, _ = _synthetic_patch_generator(tmp_path)
    old_patch = b"previous patch bundle\n"
    (manifest / "TARGET_TREE").write_text("0" * 40 + "\n", encoding="utf-8")
    (manifest / "patches" / "0001-previous.patch").write_bytes(old_patch)
    (manifest / "patches" / "series").write_text(
        "0001-previous.patch\n", encoding="utf-8"
    )
    (manifest / "patches" / "SHA256SUMS").write_text(
        f"{hashlib.sha256(old_patch).hexdigest()}  0001-previous.patch\n",
        encoding="utf-8",
    )
    before = {
        path.relative_to(manifest): path.read_bytes()
        for path in manifest.rglob("*")
        if path.is_file()
    }

    fake_bin = tmp_path / "swap-bin"
    fake_bin.mkdir()
    real_mv = shutil.which("mv")
    assert real_mv is not None
    fake_mv = fake_bin / "mv"
    fake_mv.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
count=0
[[ ! -f "$MV_COUNT_FILE" ]] || read -r count <"$MV_COUNT_FILE"
count=$((count + 1))
printf '%s\n' "$count" >"$MV_COUNT_FILE"
if [[ "$count" -eq 2 ]]; then
    exit 86
fi
exec {real_mv!r} "$@"
""",
        encoding="utf-8",
    )
    fake_mv.chmod(0o755)
    environment = _clean_environment()
    environment["MV_COUNT_FILE"] = str(tmp_path / "mv-count")
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    result = _run_script(
        script,
        "--tt-metal-dir",
        str(source),
        env=environment,
    )

    assert result.returncode != 0
    after = {
        path.relative_to(manifest): path.read_bytes()
        for path in manifest.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(manifest.parent.glob(".tt-metal.*"))


def test_patch_generator_rejects_source_index_change_before_swap(
    tmp_path: Path,
) -> None:
    script, source, manifest, _ = _synthetic_patch_generator(tmp_path)
    before = {
        path.relative_to(manifest): path.read_bytes()
        for path in manifest.rglob("*")
        if path.is_file()
    }

    fake_bin = tmp_path / "checksum-bin"
    fake_bin.mkdir()
    real_sha256sum = shutil.which("sha256sum")
    assert real_sha256sum is not None
    fake_sha256sum = fake_bin / "sha256sum"
    fake_sha256sum.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' concurrent >"$GENERATOR_SOURCE/tt_metal/fabric/fabric.cpp"
git -C "$GENERATOR_SOURCE" add tt_metal/fabric/fabric.cpp
exec {real_sha256sum!r} "$@"
""",
        encoding="utf-8",
    )
    fake_sha256sum.chmod(0o755)
    environment = _clean_environment()
    environment["GENERATOR_SOURCE"] = str(source)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    result = _run_script(
        script,
        "--tt-metal-dir",
        str(source),
        env=environment,
    )

    assert result.returncode != 0
    assert "source index changed during patch generation" in result.stderr
    after = {
        path.relative_to(manifest): path.read_bytes()
        for path in manifest.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(manifest.parent.glob(".tt-metal.*"))


def test_setup_script_requires_explicit_paths() -> None:
    result = _run_script(_SETUP_SCRIPT)

    assert result.returncode != 0
    assert "--tt-metal-dir is required" in result.stderr


def test_prepare_aiperf_requires_explicit_path() -> None:
    result = _run_script(_AIPERF_SCRIPT)

    assert result.returncode != 0
    assert "--aiperf-dir is required" in result.stderr


def test_setup_rejects_python_from_another_prefix(tmp_path: Path) -> None:
    tt_metal = tmp_path / "tt-metal"
    _new_repository(tt_metal)
    plugin = tmp_path / "plugin"
    venv = tmp_path / "fake-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /not-this-environment\n", encoding="utf-8")
    system_python = shutil.which("python3")
    assert system_python is not None
    fake_python = venv / "bin" / "python"
    fake_python.write_text(
        f'#!/usr/bin/env bash\nexec {system_python!r} "$@"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    environment = _clean_environment()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    result = _run_script(
        _SETUP_SCRIPT,
        "--tt-metal-dir",
        str(tt_metal),
        "--plugin-dir",
        str(plugin),
        "--venv-dir",
        str(venv),
        env=environment,
    )

    assert result.returncode != 0
    assert "interpreter prefix" in result.stderr
    assert "refusing to install into an invalid Python environment" in result.stderr
    assert not plugin.exists()


def test_prepare_rejects_non_repository(tmp_path: Path) -> None:
    result = _run("--tt-metal-dir", str(tmp_path))
    assert result.returncode != 0
    assert "not a Git checkout" in result.stderr


def test_prepare_rejects_dirty_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "dirty"
    _new_repository(repository)
    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    result = _run("--tt-metal-dir", str(repository))
    assert result.returncode != 0
    assert "checkout must be clean before patching" in result.stderr


def test_prepare_rejects_wrong_revision(tmp_path: Path) -> None:
    repository = tmp_path / "wrong-revision"
    _new_repository(repository)

    result = _run("--tt-metal-dir", str(repository))
    assert result.returncode != 0
    assert "expected base" in result.stderr


def test_prepare_applies_series_with_matching_index_and_worktree(
    tmp_path: Path,
) -> None:
    script, repository, revision, target_tree = _synthetic_bootstrap(tmp_path)

    result = _run_script(script, "--tt-metal-dir", str(repository))

    assert result.returncode == 0, result.stderr
    assert "PATCH_BOOTSTRAP_PASS" in result.stdout
    assert _git_output(repository, "rev-parse", "HEAD") == revision
    assert _git_output(repository, "write-tree") == target_tree
    assert _git_output(repository, "diff", "--name-only") == ""
    assert _git_output(repository, "ls-files", "--others", "--exclude-standard") == ""


def test_prepare_rejects_inherited_alternate_git_index(tmp_path: Path) -> None:
    script, repository, revision, _ = _synthetic_bootstrap(tmp_path)
    alternate_index = tmp_path / "alternate-index"
    environment = _clean_environment()
    environment["GIT_INDEX_FILE"] = str(alternate_index)
    subprocess.run(
        ["git", "-C", str(repository), "read-tree", revision],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    base_tree = _git_output(repository, "rev-parse", f"{revision}^{{tree}}")

    result = _run_script(
        script,
        "--tt-metal-dir",
        str(repository),
        env=environment,
    )

    assert result.returncode != 0
    assert "Git path override must be unset: GIT_INDEX_FILE" in result.stderr
    assert _git_output(repository, "write-tree") == base_tree
    assert _git_output(repository, "status", "--porcelain=v1") == ""
    alternate_tree = subprocess.run(
        ["git", "-C", str(repository), "write-tree"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    assert alternate_tree == base_tree


def test_prepare_rejects_other_inherited_git_path_overrides(tmp_path: Path) -> None:
    script, repository, _, _ = _synthetic_bootstrap(tmp_path)
    for override_name in _GIT_PATH_OVERRIDE_NAMES:
        if override_name == "GIT_INDEX_FILE":
            continue
        environment = _clean_environment()
        environment[override_name] = str(tmp_path / "override")
        result = _run_script(
            script,
            "--tt-metal-dir",
            str(repository),
            env=environment,
        )

        assert result.returncode != 0
        assert f"Git path override must be unset: {override_name}" in result.stderr
        assert _git_output(repository, "status", "--porcelain=v1") == ""


def test_prepare_rejects_checksum_manifest_order_mismatch(tmp_path: Path) -> None:
    script, repository, _, _ = _synthetic_bootstrap(tmp_path)
    checksum_file = (
        script.parents[1] / "third_party" / "tt-metal" / "patches" / "SHA256SUMS"
    )
    checksum_lines = checksum_file.read_text(encoding="utf-8").splitlines()
    checksum_file.write_text(
        "\n".join(reversed(checksum_lines)) + "\n", encoding="utf-8"
    )

    result = _run_script(script, "--tt-metal-dir", str(repository))

    assert result.returncode != 0
    assert "series and SHA256SUMS differ at entry 1" in result.stderr
    assert _git_output(repository, "status", "--porcelain=v1") == ""


def test_prepare_warns_when_actual_application_partially_fails(tmp_path: Path) -> None:
    script, repository, _, _ = _synthetic_bootstrap(tmp_path)
    environment = _wrapper_environment(tmp_path, repository, "fail-second")

    result = _run_script(
        script,
        "--tt-metal-dir",
        str(repository),
        env=environment,
    )

    assert result.returncode != 0
    assert "checkout may be partially patched" in result.stderr
    assert _git_output(repository, "status", "--porcelain=v1")


def test_prepare_rejects_worktree_change_during_application(tmp_path: Path) -> None:
    script, repository, _, _ = _synthetic_bootstrap(tmp_path)
    environment = _wrapper_environment(tmp_path, repository, "dirty-after-second")

    result = _run_script(
        script,
        "--tt-metal-dir",
        str(repository),
        env=environment,
    )

    assert result.returncode != 0
    assert "worktree differs from the applied index" in result.stderr
    assert "checkout may be partially patched" in result.stderr


def test_prepare_fails_when_final_untracked_inspection_fails(tmp_path: Path) -> None:
    script, repository, _, _ = _synthetic_bootstrap(tmp_path)
    environment = _wrapper_environment(tmp_path, repository, "fail-final-untracked")

    result = _run_script(
        script,
        "--tt-metal-dir",
        str(repository),
        env=environment,
    )

    assert result.returncode != 0
    assert "failed to inspect untracked files after patching" in result.stderr
    assert "checkout may be partially patched" in result.stderr
    assert "PATCH_BOOTSTRAP_PASS" not in result.stdout
