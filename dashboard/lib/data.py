"""
Cached data access for the dashboard.

Everything the dashboard shows comes from committed repository artifacts:

  * ml/data/tinyguard_vibration_dataset.csv  - the actual synthetic dataset
  * results/*.json                           - metrics from ml/run_evaluation.py
  * firmware/model_params.h                  - the FP32 constants the firmware compiles
  * firmware/model_int8_params.h             - the INT8 constants the firmware compiles

The dashboard deliberately reads the C HEADERS rather than the .pkl for live
inference, so what it displays is exactly what the firmware computes, and no
figure can drift from the deployed model. The .pkl is never loaded, which also
avoids the scikit-learn version warning.

No metric is computed here that run_evaluation.py already computes. Nothing is
hardcoded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_DIR = Path(__file__).resolve().parent.parent.parent
ML_DIR = REPO_DIR / "ml"
RESULTS_DIR = REPO_DIR / "results"
DATA_FILE = ML_DIR / "data" / "tinyguard_vibration_dataset.csv"

FEATURES = ["mean_absolute", "rms", "peak"]
SAMPLE_COLUMNS = [f"sample_{i}" for i in range(16)]

RESULT_FILES = (
    "index.json",
    "dataset_summary.json",
    "model_metrics.json",
    "quantization.json",
    "provenance.json",
)

# ml/ holds header_params.py and is not an installed package.
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))


class MissingArtifacts(RuntimeError):
    """Raised when the dashboard is opened before run_evaluation.py has run."""


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise MissingArtifacts(
            f"Dataset not found at {DATA_FILE}. "
            "Run `python ml/generate_dataset.py` first."
        )
    df = pd.read_csv(DATA_FILE)
    df["class_name"] = df["label"].map({0: "Healthy", 1: "Abnormal"})
    return df


@st.cache_data(show_spinner=False)
def load_results() -> dict:
    missing = [n for n in RESULT_FILES if not (RESULTS_DIR / n).exists()]
    if missing:
        raise MissingArtifacts(
            "Metrics artifacts are missing from results/: "
            + ", ".join(missing)
            + ". Run `python ml/run_evaluation.py` to generate them."
        )
    return {
        name.removesuffix(".json"): json.loads(
            (RESULTS_DIR / name).read_text(encoding="utf-8")
        )
        for name in RESULT_FILES
    }


@st.cache_resource(show_spinner=False)
def load_headers() -> tuple[dict, dict]:
    """The INT8 and FP32 constants as compiled into the firmware."""
    from header_params import load_fp32_header, load_int8_header

    return load_int8_header(), load_fp32_header()


def windows(df: pd.DataFrame):
    """The raw 16-sample windows as an integer array."""
    return df[SAMPLE_COLUMNS].to_numpy(dtype="int64")


def require_artifacts():
    """
    Load everything a page needs, or render an actionable error and stop.

    Returns (dataset, results, int8_header, fp32_header).
    """
    try:
        df = load_dataset()
        results = load_results()
        int8_header, fp32_header = load_headers()
    except MissingArtifacts as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001 - surface anything else plainly
        st.error(f"Could not load repository artifacts: {exc}")
        st.stop()
    return df, results, int8_header, fp32_header
