"""Overview: what the project demonstrates, and what it does not."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import components as C
from lib.data import require_artifacts
from lib.theme import active_palette

palette = active_palette()
df, results, int8_header, fp32_header = require_artifacts()

head = results["index"]["headline"]
metrics = results["model_metrics"]
quant = results["quantization"]
dataset = results["dataset_summary"]
prov = results["provenance"]

st.title("TinyGuard-AI")
st.caption(
    "ARM Cortex-M4 TinyML predictive maintenance — INT8 quantization pipeline "
    "and FP32/INT8 equivalence on a synthetic vibration dataset."
)

C.limitation_banner(prov["limitations"])

# ----------------------------------------------------------------------
# The headline: the quantization result, not the accuracy score.
# ----------------------------------------------------------------------

fidelity = quant["fidelity_whole_dataset"]

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown("##### FP32 → INT8 prediction agreement")
    st.markdown(
        f'<div class="tg-hero">{fidelity["prediction_agreement"] * 100:.2f}%</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="tg-kpi-sub">{fidelity["prediction_mismatches"]} mismatches '
        f'across all {fidelity["n"]:,} windows</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown("##### What this establishes")
    st.markdown(
        f"""
- The quantized integer model reproduces the float model's **every** class
  decision, with a mean absolute logit error of
  **{fidelity['logit_abs_error_mean']:.3f}** and a worst case of
  **{fidelity['logit_abs_error_max']:.3f}**.
- The integer tensors compiled into the firmware are **bit-exact** against the
  trained model: `integer_tensors_bit_exact = {quant['integer_tensors_bit_exact']}`.
- Dataset and both C headers regenerate **byte-for-byte** from committed sources.
        """
    )
    st.caption(
        "Python-vs-compiled-C agreement is checked separately by "
        "`make -C firmware verify`, which needs a C toolchain and so is not "
        "run from this dashboard."
    )

st.divider()

# ----------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------

footprint = quant["parameter_footprint_bytes"]

C.kpi_row(
    [
        ("Dataset windows", f"{dataset['total_windows']:,}", "synthetic, balanced"),
        ("Mean logit error", f"{fidelity['logit_abs_error_mean']:.3f}", "FP32 vs INT8"),
        ("INT8 weight bytes", f"{footprint['int8_weights_only']}", f"from {footprint['fp32_weights_only']} B FP32"),
        ("Weight reduction", f"{footprint['reduction_ratio_weights_only']:.2f}×", f"{footprint['reduction_ratio_total']:.2f}× whole model"),
        ("Hidden units", f"{metrics['shipped_model']['architecture']['hidden_units']}", "3 → 8 → 1, ReLU"),
    ]
)

# ----------------------------------------------------------------------
# The accuracy caveat, stated up front rather than buried.
# ----------------------------------------------------------------------

st.subheader("Accuracy, in context")

C.note(
    "The model scores 100%. So does logistic regression, and so does a depth-4 "
    "decision tree. The figure describes how separable the synthetic generator "
    "made the two classes — it is <strong>not</strong> evidence that the neural "
    "network is a capable classifier. Baselines are shown beside it everywhere "
    "in this dashboard for exactly this reason."
)

baselines = metrics["baselines"]
display_order = [
    ("dummy_most_frequent", "Dummy (most frequent)"),
    ("hand_threshold_rms50_or_peak100", "Hand threshold"),
    ("decision_tree_depth4", "Decision tree (depth 4)"),
    ("logistic_regression", "Logistic regression"),
    ("mlp_8_shipped_architecture", "MLP 3→8→1 (shipped)"),
]

names = [label for key, label in display_order]
accs = [baselines[key]["accuracy"] for key, _ in display_order]

fig = C.ranked_bars(
    names,
    accs,
    palette,
    value_fmt="{:.3f}",
    emphasis="MLP 3→8→1 (shipped)",
    x_title="Accuracy on held-out test split (n = 600)",
    x_range=(0, 1.08),
)
C.chart(
    fig,
    pd.DataFrame(
        {
            "Model": names,
            "Accuracy": accs,
            "Precision": [baselines[k]["precision"] for k, _ in display_order],
            "Recall": [baselines[k]["recall"] for k, _ in display_order],
            "F1": [baselines[k]["f1"] for k, _ in display_order],
        }
    ),
    label="baseline comparison",
)

st.markdown(
    C.status_chip(
        f"Leakage-free grouped split also {metrics['leakage_free_grouped_split']['test']['accuracy']:.3f}",
        "warning",
        palette,
    )
    + C.status_chip(
        f"{dataset['redundancy']['unique_feature_vectors']:,} unique feature vectors "
        f"of {dataset['total_windows']:,} rows",
        "warning",
        palette,
    ),
    unsafe_allow_html=True,
)

C.note(
    "The naive split leaks: "
    f"{dataset['naive_split_leakage']['fraction'] * 100:.1f}% of test rows share an exact "
    "feature vector with training data. But a leakage-free grouped split scores the "
    "same, so leakage is not what produces the perfect score — the generator's "
    "class separation is. See the Dataset page."
)

st.divider()

st.subheader("Pipeline")

st.markdown(
    """
| Stage | Artifact | Verified by |
|---|---|---|
| Synthetic signal generation | `ml/data/tinyguard_vibration_dataset.csv` | regenerates byte-for-byte |
| Integer feature extraction | `mean_absolute`, `rms`, `peak` | Python ↔ C parity tests |
| FP32 training | `ml/models/tinyguard_mlp.pkl` | deterministic (`random_state=42`) |
| FP32 export | `firmware/model_params.h` | byte-for-byte reproducible |
| INT8 quantization | `firmware/model_int8_params.h` | integer tensors bit-exact |
| Bare-metal inference | `firmware/tinyguard_int8.c` | 0 mismatches vs Python, all 3000 windows |
| ARM execution | QEMU `mps2-an386` | emulation only — never silicon |
"""
)
