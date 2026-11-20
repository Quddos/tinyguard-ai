"""Model performance: the shipped network against baselines and held-out splits."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import components as C
from lib.data import require_artifacts
from lib.theme import active_palette

palette = active_palette()
df, results, int8_header, fp32_header = require_artifacts()

metrics = results["model_metrics"]
dataset = results["dataset_summary"]
prov = results["provenance"]

shipped = metrics["shipped_model"]
arch = shipped["architecture"]

st.title("Model performance")
st.caption(
    f"MLP {arch['inputs']}→{arch['hidden_units']}→{arch['outputs']}, "
    f"{arch['hidden_activation']} hidden activation, {arch['solver']} solver, "
    f"random_state={arch['random_state']}."
)

C.limitation_banner(prov["limitations"])

# ----------------------------------------------------------------------
# Baselines first, so the headline number is never read alone
# ----------------------------------------------------------------------

st.subheader("Against baselines")

C.note(
    "Presented first deliberately. The shipped model's score is only "
    "interpretable next to what trivial models achieve on the same split."
)

baselines = metrics["baselines"]
order = [
    ("dummy_most_frequent", "Dummy (most frequent)"),
    ("hand_threshold_rms50_or_peak100", "Hand threshold"),
    ("decision_tree_depth4", "Decision tree (depth 4)"),
    ("logistic_regression", "Logistic regression"),
    ("mlp_8_shipped_architecture", "MLP 3→8→1 (shipped)"),
]

table = pd.DataFrame(
    {
        "Model": [label for _, label in order],
        "Accuracy": [baselines[k]["accuracy"] for k, _ in order],
        "Precision": [baselines[k]["precision"] for k, _ in order],
        "Recall": [baselines[k]["recall"] for k, _ in order],
        "F1": [baselines[k]["f1"] for k, _ in order],
    }
)

fig = C.ranked_bars(
    list(table["Model"]),
    list(table["Accuracy"]),
    palette,
    value_fmt="{:.3f}",
    emphasis="MLP 3→8→1 (shipped)",
    x_title=f"Accuracy on held-out test split (n = {shipped['naive_split_test']['n']})",
    x_range=(0, 1.08),
)
C.chart(fig, table.round(4), label="baseline metrics")

st.info(
    metrics["baseline_interpretation"],
    icon=":material/science:",
)

st.divider()

# ----------------------------------------------------------------------
# The two splits side by side
# ----------------------------------------------------------------------

st.subheader("Held-out performance, two ways")

naive = shipped["naive_split_test"]
grouped = metrics["leakage_free_grouped_split"]
leak = dataset["naive_split_leakage"]

left, right = st.columns(2, gap="large")

with left:
    st.markdown("###### Committed split (leaky)")
    st.caption(
        f"`train_test_split(test_size={metrics['split']['test_size']}, "
        f"random_state={metrics['split']['random_state']}, stratify=y)` — "
        f"n = {naive['n']}"
    )
    C.confusion_grid(naive["confusion_matrix"], palette)
    C.kv_table(
        [
            ("Accuracy", f"{naive['accuracy']:.4f}"),
            ("Precision", f"{naive['precision']:.4f}"),
            ("Recall", f"{naive['recall']:.4f}"),
            ("F1", f"{naive['f1']:.4f}"),
            ("ROC AUC", f"{naive.get('roc_auc', float('nan')):.4f}"),
            ("Rows also seen in training",
             f"{leak['test_rows_whose_vector_appears_in_train']} "
             f"({leak['fraction'] * 100:.1f}%)"),
        ],
        palette,
    )

with right:
    st.markdown("###### Grouped split (leakage-free)")
    st.caption(
        "`GroupShuffleSplit` grouped by unique feature vector — "
        f"n = {grouped['test_n']}, retrained from scratch"
    )
    C.confusion_grid(grouped["test"]["confusion_matrix"], palette)
    C.kv_table(
        [
            ("Accuracy", f"{grouped['test']['accuracy']:.4f}"),
            ("Precision", f"{grouped['test']['precision']:.4f}"),
            ("Recall", f"{grouped['test']['recall']:.4f}"),
            ("F1", f"{grouped['test']['f1']:.4f}"),
            ("ROC AUC", f"{grouped['test'].get('roc_auc', float('nan')):.4f}"),
            ("Shared vectors across split",
             str(grouped["shared_feature_vectors_across_split"])),
        ],
        palette,
    )

C.note(
    "Removing the leakage does not move the score. That is the evidence that the "
    "generator's separability, not the split, produces the perfect result."
)

st.divider()

# ----------------------------------------------------------------------
# Per-scenario and leave-one-scenario-out
# ----------------------------------------------------------------------

st.subheader("Per-scenario accuracy")

C.note(
    "Shown as meters rather than bars: every value sits at or near the ceiling, "
    "so bar lengths would be indistinguishable while the figures still need "
    "reading."
)

per_scenario = metrics["per_scenario"]
rows = [
    (
        name.replace("_", " "),
        block["accuracy"],
        f"{block['accuracy']:.4f}",
    )
    for name, block in sorted(per_scenario.items(), key=lambda kv: kv[1]["accuracy"])
]
C.meter_list(rows, palette)

with st.expander("Table view — per-scenario accuracy"):
    st.dataframe(
        pd.DataFrame(
            {
                "Scenario": list(per_scenario),
                "Label": [b["label"] for b in per_scenario.values()],
                "Windows": [b["windows"] for b in per_scenario.values()],
                "Correct": [b["correct"] for b in per_scenario.values()],
                "Accuracy": [round(b["accuracy"], 6) for b in per_scenario.values()],
            }
        ),
        width="stretch",
        hide_index=True,
    )

st.subheader("Generalisation to an unseen fault type")

C.note(
    "Each run trains on only two of the three fault scenarios and tests on the "
    "third, plus held-out healthy windows. This asks whether the model "
    "recognises a fault mode it never saw."
)

loso = metrics["leave_one_scenario_out"]
rows = [
    (
        f"held out: {name.replace('_', ' ')}",
        block["test"]["accuracy"],
        f"{block['test']['accuracy']:.4f}",
    )
    for name, block in sorted(loso.items(), key=lambda kv: kv[1]["test"]["accuracy"])
]
C.meter_list(rows, palette)

loso_table = pd.DataFrame(
    {
        "Held-out scenario": list(loso),
        "Train windows": [b["train_windows"] for b in loso.values()],
        "Test windows": [b["test_windows"] for b in loso.values()],
        "Unseen abnormal": [b["unseen_abnormal_windows"] for b in loso.values()],
        "Accuracy": [round(b["test"]["accuracy"], 6) for b in loso.values()],
        "False negatives": [
            b["test"]["confusion_matrix"]["false_negative"] for b in loso.values()
        ],
        "False positives": [
            b["test"]["confusion_matrix"]["false_positive"] for b in loso.values()
        ],
    }
)

with st.expander("Table view — leave-one-scenario-out"):
    st.dataframe(loso_table, width="stretch", hide_index=True)

worst = min(loso.items(), key=lambda kv: kv[1]["test"]["accuracy"])
worst_cm = worst[1]["test"]["confusion_matrix"]
st.caption(
    f"Weakest case: holding out *{worst[0].replace('_', ' ')}* gives "
    f"{worst[1]['test']['accuracy']:.4f} — "
    f"{worst_cm['false_negative']} missed fault(s) and "
    f"{worst_cm['false_positive']} false alarm(s) out of {worst[1]['test']['n']} "
    "test windows. Not perfect, and reported as such."
)
