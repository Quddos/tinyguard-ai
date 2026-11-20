"""Provenance: exactly which artifacts produced the figures on every other page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import components as C
from lib.data import RESULTS_DIR, require_artifacts
from lib.theme import active_palette

palette = active_palette()
df, results, int8_header, fp32_header = require_artifacts()

prov = results["provenance"]
index = results["index"]
dataset = results["dataset_summary"]

st.title("Provenance")
st.caption(
    "Every figure in this dashboard is traceable to a committed artifact and a "
    "recorded environment."
)

C.limitation_banner(prov["limitations"])

# ----------------------------------------------------------------------
# Generation record
# ----------------------------------------------------------------------

st.subheader("Metrics generation")

sklearn_ok = prov["sklearn_version_matches_training"]

st.markdown(
    C.status_chip(
        f"generated {prov['generated_utc']}", "good", palette
    )
    + C.status_chip(
        f"scikit-learn {prov['environment']['scikit_learn']}"
        + ("" if sklearn_ok else f" ≠ training {prov['model_trained_with_sklearn']}"),
        "good" if sklearn_ok else "warning",
        palette,
    ),
    unsafe_allow_html=True,
)

if not sklearn_ok:
    st.warning(
        f"The committed model artifacts were serialised by scikit-learn "
        f"**{prov['model_trained_with_sklearn']}**, but these metrics were "
        f"generated under **{prov['environment']['scikit_learn']}**. The "
        "figures shown are still internally consistent — the dashboard reads "
        "the C headers, not the pickle — but byte-exact header regeneration is "
        "only guaranteed on the pinned version. See `requirements.txt`.",
        icon=":material/warning:",
    )

C.kv_table(
    [
        ("Generator", prov["generator"]),
        ("Generated (UTC)", prov["generated_utc"]),
        ("Git commit", prov["git_commit"] or "not available"),
        ("Model trained with", f"scikit-learn {prov['model_trained_with_sklearn']}"),
    ],
    palette,
)

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

st.subheader("Environment")

env = prov["environment"]
C.kv_table(
    [
        ("Python", env["python"]),
        ("Platform", env["platform"]),
        ("numpy", env["numpy"]),
        ("pandas", env["pandas"]),
        ("scikit-learn", env["scikit_learn"]),
        ("joblib", env["joblib"]),
    ],
    palette,
)

# ----------------------------------------------------------------------
# Artifact hashes
# ----------------------------------------------------------------------

st.subheader("Artifact hashes")

C.note(
    "SHA-256 of every input the metrics were computed from. If any of these "
    "change, <code>ml/run_evaluation.py</code> must be re-run before the "
    "figures can be quoted."
)

labels = {
    "dataset": "ml/data/tinyguard_vibration_dataset.csv",
    "model_pkl": "ml/models/tinyguard_mlp.pkl",
    "scaler_pkl": "ml/models/tinyguard_scaler.pkl",
    "int8_header": "firmware/model_int8_params.h",
    "fp32_header": "firmware/model_params.h",
}

st.dataframe(
    pd.DataFrame(
        {
            "Artifact": [labels.get(k, k) for k in prov["artifact_hashes"]],
            "SHA-256": list(prov["artifact_hashes"].values()),
        }
    ),
    width="stretch",
    hide_index=True,
)

# ----------------------------------------------------------------------
# The metrics files themselves
# ----------------------------------------------------------------------

st.subheader("Metrics files")

files = index["files"]
st.dataframe(
    pd.DataFrame(
        {
            "File": [f"results/{name}" for name in files],
            "Bytes": [meta["bytes"] for meta in files.values()],
            "SHA-256": [meta["sha256"] for meta in files.values()],
        }
    ),
    width="stretch",
    hide_index=True,
)

C.kv_table(
    [
        ("Schema version", str(index["schema_version"])),
        ("Results directory", str(RESULTS_DIR)),
        ("Dataset origin", dataset["data_origin"]),
        ("Physical sensor used", str(dataset["physical_sensor_used"])),
    ],
    palette,
)

# ----------------------------------------------------------------------
# Reproduction
# ----------------------------------------------------------------------

st.subheader("Reproducing these figures")

st.code(
    "python -m pip install -r requirements.txt\n"
    "python ml/generate_dataset.py      # regenerates the dataset byte-for-byte\n"
    "python ml/export_model.py          # FP32 C header\n"
    "python ml/export_int8_model.py     # INT8 C header\n"
    "python ml/run_evaluation.py        # rewrites results/*.json\n"
    "python -m pytest                   # 72 tests incl. determinism checks\n"
    "make -C firmware verify            # Python vs compiled C (needs a C toolchain)",
    language="bash",
)

st.caption(
    "The determinism tests assert that the dataset and both C headers "
    "regenerate byte-for-byte from the committed sources, so any drift in these "
    "figures is caught rather than inherited."
)

# ----------------------------------------------------------------------
# Raw artifacts
# ----------------------------------------------------------------------

st.subheader("Raw artifacts")

tabs = st.tabs(["index", "dataset_summary", "model_metrics", "quantization", "provenance"])
for tab, key in zip(
    tabs, ["index", "dataset_summary", "model_metrics", "quantization", "provenance"]
):
    with tab:
        st.json(results[key], expanded=False)
