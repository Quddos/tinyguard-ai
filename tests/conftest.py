"""
Shared fixtures for the TinyGuard-AI test suite.

The ml/ directory is not a package, so it is put on sys.path here rather than
restructuring the existing research scripts.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
ML_DIR = REPO_DIR / "ml"
FIRMWARE_DIR = REPO_DIR / "firmware"
RESULTS_DIR = REPO_DIR / "results"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


def pytest_configure(config):
    # The committed .pkl was written by scikit-learn 1.9.1. Running under a
    # different patch release warns on every load; the version mismatch is
    # asserted explicitly in test_provenance.py instead of drowning the output.
    warnings.filterwarnings(
        "ignore",
        message=".*Trying to unpickle estimator.*",
    )


@pytest.fixture(scope="session")
def repo_dir() -> Path:
    return REPO_DIR


@pytest.fixture(scope="session")
def ml_dir() -> Path:
    return ML_DIR


@pytest.fixture(scope="session")
def firmware_dir() -> Path:
    return FIRMWARE_DIR


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return RESULTS_DIR


@pytest.fixture(scope="session")
def dataset():
    import pandas as pd

    return pd.read_csv(ML_DIR / "data" / "tinyguard_vibration_dataset.csv")


@pytest.fixture(scope="session")
def trained():
    import joblib

    model = joblib.load(ML_DIR / "models" / "tinyguard_mlp.pkl")
    scaler = joblib.load(ML_DIR / "models" / "tinyguard_scaler.pkl")
    return model, scaler


@pytest.fixture(scope="session")
def int8_header():
    from header_params import load_int8_header

    return load_int8_header()


@pytest.fixture(scope="session")
def fp32_header():
    from header_params import load_fp32_header

    return load_fp32_header()
