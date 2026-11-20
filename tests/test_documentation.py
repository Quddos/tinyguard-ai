"""
Documentation guard tests.

The synthetic-data and QEMU-emulation limitations are a standing requirement of
this project, and the accuracy figures are only honest when presented beside
the baselines. These tests make removing any of that a failing change rather
than a silent edit.
"""

from __future__ import annotations

import re

import pytest


@pytest.fixture(scope="module")
def readme(repo_dir):
    path = repo_dir / "README.md"
    assert path.exists(), "README.md is missing"
    return path.read_text(encoding="utf-8")


def test_readme_states_data_is_synthetic(readme):
    assert re.search(r"\bsynthetic\b", readme, flags=re.I), (
        "README must state that the dataset is synthetic"
    )


def test_readme_states_arm_is_emulated(readme):
    assert re.search(r"\bQEMU\b", readme), (
        "README must state that ARM execution is QEMU emulation"
    )


def test_readme_disclaims_physical_hardware_testing(readme):
    """
    An explicit disclaimer must be present, not merely an absence of claims.
    """
    lowered = readme.lower()
    assert "does not claim physical sensor validation" in lowered, (
        "README must explicitly disclaim physical sensor / hardware testing"
    )


def test_readme_makes_no_physical_hardware_claim(readme):
    """
    Catch phrasings that would assert real hardware or sensor measurement.
    """
    forbidden = [
        r"\bon real hardware\b",
        r"\bon physical hardware\b",
        r"\breal[- ]world sensor data\b",
        r"\bmeasured on (?:a |the )?(?:device|board|microcontroller)\b",
        r"\bdeployed (?:to|on) (?:a |the )?physical\b",
        r"\breal accelerometer\b",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, readme, flags=re.I), (
            f"README contains a physical-hardware claim matching {pattern!r}"
        )


def test_readme_presents_baselines_with_accuracy(readme):
    """
    A bare accuracy figure is misleading for this dataset, so the baseline
    comparison must stay in the README.
    """
    for needle in ("Logistic regression", "Dummy", "Decision tree"):
        assert needle in readme, f"README baseline table lost {needle!r}"


def test_readme_explains_the_separability_caveat(readme):
    lowered = readme.lower()
    assert "separable by construction" in lowered or (
        "property of the dataset generator" in lowered
    ), "README must explain that the task is separable by construction"


def test_readme_documents_saturation_limitation(readme):
    assert re.search(r"saturat", readme, flags=re.I), (
        "README must document the INT8 input saturation limitation"
    )


def test_readme_does_not_claim_latency_measurement(readme):
    """
    No wall-clock or energy figure can be obtained from emulation here, so none
    may be reported.
    """
    forbidden = [
        r"\d+\s*(?:ms|us|µs|microseconds|milliseconds)\s+(?:per\s+)?inference",
        r"inference (?:time|latency)\s*[:=]\s*\d",
        r"\d+\s*(?:mW|mJ|uJ)\b",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, readme, flags=re.I), (
            f"README appears to report an unmeasured timing/energy figure "
            f"({pattern!r})"
        )


def test_license_exists_and_is_recognisable(repo_dir):
    path = repo_dir / "LICENSE"
    assert path.exists(), "LICENSE is missing; a public repo without one is all-rights-reserved"

    text = path.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Copyright (c)" in text
    assert "WITHOUT WARRANTY OF ANY KIND" in text.upper()


def test_requirements_pins_sklearn_to_training_version(repo_dir):
    """
    The committed .pkl was written by scikit-learn 1.9.1; the pin must match so
    header regeneration is reproducible.
    """
    text = (repo_dir / "requirements.txt").read_text(encoding="utf-8")
    assert "scikit-learn==1.9.1" in text, (
        "requirements.txt must pin the scikit-learn version that produced the "
        "committed model artifacts"
    )

    for package in ("numpy", "pandas", "joblib", "streamlit", "pytest"):
        assert re.search(rf"^{package}==", text, flags=re.M), (
            f"{package} is not pinned in requirements.txt"
        )


def test_gitignore_excludes_build_output_but_keeps_research_assets(repo_dir):
    text = (repo_dir / ".gitignore").read_text(encoding="utf-8")

    for pattern in ("firmware/build/", "__pycache__/", "verify_int8_host"):
        assert pattern in text, f".gitignore should exclude {pattern}"

    # Research assets must remain tracked.
    for asset in ("ml/data", "ml/models", "results"):
        assert not re.search(rf"^{re.escape(asset)}/?$", text, flags=re.M), (
            f".gitignore must not exclude the research asset {asset}"
        )


def test_gitattributes_protects_binary_model_artifacts(repo_dir):
    """
    `* text=auto` relies on a heuristic to leave .pkl files alone. The explicit
    rule must stay, or a bad checkout can corrupt the committed model.
    """
    text = (repo_dir / ".gitattributes").read_text(encoding="utf-8")
    assert re.search(r"^\*\.pkl\s+binary", text, flags=re.M), (
        ".gitattributes must mark *.pkl as binary"
    )
