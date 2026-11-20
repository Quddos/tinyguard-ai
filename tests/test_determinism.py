"""
Determinism and reproducibility tests.

These assert that the committed research artifacts can be regenerated from the
committed sources, which is the property a reader needs in order to trust any
reported figure.

Each test runs the real generator/exporter in a temporary working copy and
compares the output byte for byte against what is committed, so nothing in the
repository is modified by the test run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def run_script(script: Path, cwd: Path) -> subprocess.CompletedProcess:
    """
    Run a pipeline script in a subprocess with a 5 minute cap.

    Uses -s (no user site-packages) and -E (ignore PYTHON* env vars) rather
    than -I, because full isolation also removes the script's own directory
    from sys.path and run_evaluation.py imports header_params from there.
    """
    result = subprocess.run(
        [sys.executable, "-s", "-E", str(script)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result


@pytest.fixture
def workspace(tmp_path, repo_dir):
    """A disposable copy of ml/ plus firmware/ so exports can write freely."""
    shutil.copytree(repo_dir / "ml", tmp_path / "ml")
    (tmp_path / "firmware").mkdir()
    for name in ("model_params.h", "model_int8_params.h"):
        shutil.copy2(repo_dir / "firmware" / name, tmp_path / "firmware" / name)
    return tmp_path


def normalised(path: Path) -> bytes:
    """Compare content independent of checkout line-ending policy."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def test_dataset_generation_is_reproducible(workspace, repo_dir):
    """
    generate_dataset.py must reproduce the committed CSV exactly.

    This is also the regression test for the dead-code removal in
    generate_dataset.py: the refactor had to preserve the RNG draw order, and
    any change to it shows up here as a different dataset.
    """
    ml = workspace / "ml"
    committed = normalised(repo_dir / "ml" / "data" / "tinyguard_vibration_dataset.csv")

    run_script(ml / "generate_dataset.py", ml)

    regenerated = normalised(ml / "data" / "tinyguard_vibration_dataset.csv")

    assert regenerated == committed, (
        "generate_dataset.py no longer reproduces the committed dataset. "
        "The RNG draw order changed."
    )


def test_dataset_generation_is_stable_across_runs(workspace):
    """Two consecutive runs must agree, i.e. no unseeded randomness remains."""
    ml = workspace / "ml"

    run_script(ml / "generate_dataset.py", ml)
    first = normalised(ml / "data" / "tinyguard_vibration_dataset.csv")

    run_script(ml / "generate_dataset.py", ml)
    second = normalised(ml / "data" / "tinyguard_vibration_dataset.csv")

    assert first == second, "generate_dataset.py is not run-to-run stable"


def test_fp32_header_export_is_reproducible(workspace, repo_dir):
    """export_model.py must reproduce the committed FP32 C header."""
    ml = workspace / "ml"
    committed = normalised(repo_dir / "firmware" / "model_params.h")

    run_script(ml / "export_model.py", ml)

    assert normalised(workspace / "firmware" / "model_params.h") == committed
    assert normalised(ml / "models" / "model_params.h") == committed, (
        "the ml/models and firmware copies of model_params.h diverged"
    )


def test_int8_header_export_is_reproducible(workspace, repo_dir):
    """export_int8_model.py must reproduce the committed INT8 C header."""
    ml = workspace / "ml"
    committed = normalised(repo_dir / "firmware" / "model_int8_params.h")

    run_script(ml / "export_int8_model.py", ml)

    regenerated = normalised(workspace / "firmware" / "model_int8_params.h")

    assert regenerated == committed, (
        "export_int8_model.py no longer reproduces the committed INT8 header"
    )


def test_int8_header_export_is_stable_across_runs(workspace):
    ml = workspace / "ml"
    target = workspace / "firmware" / "model_int8_params.h"

    run_script(ml / "export_int8_model.py", ml)
    first = normalised(target)

    run_script(ml / "export_int8_model.py", ml)
    second = normalised(target)

    assert first == second


def test_exports_write_lf_line_endings(workspace):
    """
    Generated headers must use LF regardless of host OS, otherwise the
    byte-comparison above would depend on the platform running the tests.
    """
    ml = workspace / "ml"
    run_script(ml / "export_int8_model.py", ml)
    run_script(ml / "export_model.py", ml)

    for path in (
        workspace / "firmware" / "model_int8_params.h",
        workspace / "firmware" / "model_params.h",
    ):
        assert b"\r\n" not in path.read_bytes(), f"{path.name} contains CRLF"


def test_evaluation_metrics_are_reproducible(workspace, repo_dir):
    """
    run_evaluation.py must produce identical JSON content on repeated runs,
    apart from the deliberately varying timestamp and hashes-of-outputs.
    """
    import json

    ml = workspace / "ml"

    run_script(ml / "run_evaluation.py", ml)
    first = {
        p.name: json.loads(p.read_text(encoding="utf-8"))
        for p in (workspace / "results").glob("*.json")
    }

    run_script(ml / "run_evaluation.py", ml)
    second = {
        p.name: json.loads(p.read_text(encoding="utf-8"))
        for p in (workspace / "results").glob("*.json")
    }

    assert set(first) == set(second)

    for name in first:
        a, b = first[name], second[name]
        for volatile in ("generated_utc", "files"):
            a.pop(volatile, None)
            b.pop(volatile, None)
        assert a == b, f"{name} is not reproducible across runs"


def test_no_hardcoded_accuracy_in_evaluation(ml_dir):
    """
    Guard against a future edit that pastes a literal score into the metrics
    export. Every reported figure must be computed.
    """
    source = (ml_dir / "run_evaluation.py").read_text(encoding="utf-8")

    banned = ['"accuracy": 1.0', "'accuracy': 1.0", '"accuracy": 100', "= 1.0  # accuracy"]
    for pattern in banned:
        assert pattern not in source, (
            f"run_evaluation.py appears to hardcode a metric: {pattern!r}"
        )
