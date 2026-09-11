# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import subprocess


_REPOSITORY = Path(__file__).resolve().parents[1]
_RUNNER = _REPOSITORY / "run_multiturn_prefix_reuse.sh"


def test_multiturn_runner_reports_the_first_missing_required_value() -> None:
    result = subprocess.run(
        ["bash", str(_RUNNER)],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},
    )

    assert result.returncode == 2
    assert "missing required configuration" in result.stderr
    assert "TT_METAL_ROOT" in result.stderr
    assert "AIPERF_VENV" in result.stderr
    assert "T3K_HOST" in result.stderr
    assert "docs/usage.md" in result.stderr


def test_multiturn_runner_builds_a_kvm_disabled_command(tmp_path: Path) -> None:
    tt_metal = tmp_path / "tt-metal"
    plugin = tmp_path / "plugin"
    plugin_venv = tmp_path / "plugin-venv"
    aiperf_venv = tmp_path / "aiperf-venv"
    model = tmp_path / "model"
    hf_home = tmp_path / "hf-home"
    output = tmp_path / "results"
    for directory in (tt_metal, plugin, plugin_venv / "bin", aiperf_venv / "bin", model, hf_home):
        directory.mkdir(parents=True)

    for executable in (plugin_venv / "bin" / "python", aiperf_venv / "bin" / "aiperf"):
        executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)

    delegated_runner = tt_metal / "lmcache-t3knic" / "benchmarks" / "run_cross_node_ttft.sh"
    delegated_runner.parent.mkdir(parents=True)
    delegated_runner.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$KVM_DISABLE $BENCHMARK_DRIVER $MAX_MODEL_LEN $AGENTX_REQUEST_COUNT\"\n",
        encoding="utf-8",
    )
    delegated_runner.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "TT_METAL_ROOT": str(tt_metal),
            "PLUGIN": str(plugin),
            "PLUGIN_VENV": str(plugin_venv),
            "AIPERF_VENV": str(aiperf_venv),
            "HF_HOME": str(hf_home),
            "MODEL": str(model),
            "OUT_MULTITURN": str(output),
            "MULTITURN_REQUEST_COUNT": "7",
            "KVM_DISABLE": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(_RUNNER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1 agentx 8192 7"
