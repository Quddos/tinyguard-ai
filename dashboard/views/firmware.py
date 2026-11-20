"""
Firmware page: what the QEMU demo actually runs, and a live inference explorer.

This is where the saturation limitation becomes visible rather than stated: the
severe demo window's INT8 logit reads far below its FP32 value because two of
its three inputs clip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import components as C
from lib.data import FEATURES, SAMPLE_COLUMNS, require_artifacts
from lib.inference import (
    extract_features,
    fp32_forward,
    int8_forward,
    representable_input_limit,
)
from lib.theme import active_palette, plotly_layout

palette = active_palette()
df, results, int8_header, fp32_header = require_artifacts()

quant = results["quantization"]
prov = results["provenance"]
demo = quant["firmware_demo_windows"]

FP32_C, INT8_C = palette.series[0], palette.series[1]

st.title("Firmware")
st.caption(
    "The three synthetic windows `firmware/sensor_sim.c` feeds the model, and a "
    "live explorer running the committed C constants."
)

C.limitation_banner(prov["limitations"])

st.error(
    "ARM Cortex-M4 execution is **QEMU emulation** (`-M mps2-an386`). No "
    "physical microcontroller was programmed, run or timed, and no latency or "
    "energy figure is reported anywhere in this dashboard.",
    icon=":material/memory:",
)

# ----------------------------------------------------------------------
# The three demo windows
# ----------------------------------------------------------------------

st.subheader("Demo windows")

LABELS = {
    "SENSOR_NORMAL": "Healthy",
    "SENSOR_ABNORMAL_TYPICAL": "Abnormal (typical)",
    "SENSOR_ABNORMAL": "Abnormal (severe)",
}
ORDER = ["SENSOR_NORMAL", "SENSOR_ABNORMAL_TYPICAL", "SENSOR_ABNORMAL"]

cols = st.columns(3, gap="medium")
for col, name in zip(cols, ORDER):
    block = demo[name]
    with col:
        st.markdown(f"###### {LABELS[name]}")
        st.code(name, language=None)

        chips = C.status_chip(
            "in training range" if block["within_training_range_for_its_class"]
            else "out of distribution",
            "good" if block["within_training_range_for_its_class"] else "warning",
            palette,
        ) + C.status_chip(
            "saturated" if block["input_saturated"] else "no clipping",
            "warning" if block["input_saturated"] else "good",
            palette,
        )
        st.markdown(chips, unsafe_allow_html=True)

        f = block["features"]
        C.kv_table(
            [
                ("Features", f"{f['mean_absolute']}, {f['rms']}, {f['peak']}"),
                ("INT8 input", str(block["int8_quantized_input"])),
                ("FP32 logit", f"{block['fp32_logit']:+.3f}"),
                ("INT8 logit", f"{block['int8_logit']:+.3f}"),
                ("Logit error", f"{block['logit_abs_error']:.3f}"),
                ("Decision",
                 "Anomaly" if block["int8_prediction"] == 1 else "Normal"),
            ],
            palette,
        )

# ----------------------------------------------------------------------
# FP32 vs INT8 per demo window — makes the compression obvious
# ----------------------------------------------------------------------

st.subheader("Where saturation costs precision")

names = [LABELS[n] for n in ORDER]
fp32_vals = [demo[n]["fp32_logit"] for n in ORDER]
int8_vals = [demo[n]["int8_logit"] for n in ORDER]

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=names, y=fp32_vals, name="FP32 logit",
        marker=dict(color=FP32_C, line=dict(width=0)),
        text=[f"{v:+.1f}" for v in fp32_vals],
        textposition="outside",
        textfont=dict(color=palette.ink_secondary, size=12),
        cliponaxis=False,
        hovertemplate="FP32 %{y:.3f}<extra></extra>",
    )
)
fig.add_trace(
    go.Bar(
        x=names, y=int8_vals, name="INT8 logit",
        marker=dict(color=INT8_C, line=dict(width=0)),
        text=[f"{v:+.1f}" for v in int8_vals],
        textposition="outside",
        textfont=dict(color=palette.ink_secondary, size=12),
        cliponaxis=False,
        hovertemplate="INT8 %{y:.3f}<extra></extra>",
    )
)
fig.add_hline(y=0, line=dict(color=palette.axis, width=1))

layout = plotly_layout(palette, height=380, showlegend=True)
layout["yaxis"].update(title=dict(text="Logit"))
layout["xaxis"].update(showgrid=False)
fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, **layout)
fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))

C.chart(
    fig,
    pd.DataFrame(
        {
            "Window": names,
            "Features": [
                ", ".join(str(demo[n]["features"][k]) for k in FEATURES)
                for n in ORDER
            ],
            "FP32 logit": [round(v, 4) for v in fp32_vals],
            "INT8 logit": [round(v, 4) for v in int8_vals],
            "Absolute error": [round(demo[n]["logit_abs_error"], 4) for n in ORDER],
            "Input saturated": [demo[n]["input_saturated"] for n in ORDER],
            "Predictions agree": [demo[n]["predictions_agree"] for n in ORDER],
        }
    ),
    label="demo window logits",
)

severe = demo["SENSOR_ABNORMAL"]
compression = (1 - severe["int8_logit"] / severe["fp32_logit"]) * 100

st.warning(
    f"The severe window's features "
    f"({severe['features']['mean_absolute']}, {severe['features']['rms']}, "
    f"{severe['features']['peak']}) lie outside the trained range, so two of "
    f"three INT8 inputs clip at ±127 and the logit reads "
    f"**{compression:.0f}% low** ({severe['fp32_logit']:+.1f} → "
    f"{severe['int8_logit']:+.1f}). The class decision is still correct, but "
    "use the *typical* window when quoting quantization fidelity.",
    icon=":material/warning:",
)

st.divider()

# ----------------------------------------------------------------------
# Live inference explorer
# ----------------------------------------------------------------------

st.subheader("Inference explorer")

C.note(
    "Pick any window from the dataset, or enter features directly, and watch "
    "both engines run. Every number below is produced by the constants in "
    "<code>firmware/model_int8_params.h</code> using C float32 semantics — the "
    "same arithmetic the Cortex-M4 performs."
)

source = st.radio(
    "Input source",
    ["Dataset window", "Manual features"],
    horizontal=True,
    label_visibility="collapsed",
)

if source == "Dataset window":
    c1, c2 = st.columns([1, 1])
    with c1:
        scenario = st.selectbox(
            "Scenario", sorted(df["scenario"].unique()),
            format_func=lambda s: s.replace("_", " "),
            key="fw_scenario",
        )
    subset = df[df.scenario == scenario].reset_index(drop=True)
    with c2:
        idx = st.number_input(
            "Window", min_value=0, max_value=len(subset) - 1, value=0, step=1,
            key="fw_idx",
        )
    row = subset.iloc[int(idx)]
    window = [int(row[c]) for c in SAMPLE_COLUMNS]
    feats = extract_features(window)
    true_label = int(row["label"])
else:
    window = None
    true_label = None
    c1, c2, c3 = st.columns(3)
    with c1:
        ma = st.number_input("mean absolute", 0, 5000, 30, 1)
    with c2:
        rms = st.number_input("rms", 0, 5000, 35, 1)
    with c3:
        pk = st.number_input("peak", 0, 5000, 74, 1)
    feats = (int(ma), int(rms), int(pk))

features_arr = np.array([feats], dtype=np.float64)

fp32_logit = float(fp32_forward(features_arr, fp32_header).ravel()[0])
out = int8_forward(features_arr, int8_header)

int8_logit = float(out["logit"].ravel()[0])
int8_pred = int(out["prediction"][0])
fp32_pred = int(fp32_logit >= 0)
is_saturated = bool(out["saturated"][0])
limit = representable_input_limit(int8_header)

if window is not None:
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=palette.axis, width=1))
    colour = palette.series[0] if true_label == 0 else palette.series[1]
    fig.add_trace(
        go.Scatter(
            x=list(range(16)), y=window,
            mode="lines+markers",
            line=dict(color=colour, width=2),
            marker=dict(color=colour, size=8,
                        line=dict(width=2, color=palette.surface)),
            hovertemplate="sample %{x}: %{y} mg<extra></extra>",
            showlegend=False,
        )
    )
    layout = plotly_layout(palette, height=250)
    layout["xaxis"].update(title=dict(text="Sample index"), dtick=1)
    layout["yaxis"].update(title=dict(text="mg"))
    fig.update_layout(**layout)
    C.chart(
        fig,
        pd.DataFrame({"Sample index": list(range(16)), "Value (mg)": window}),
        label="selected window",
    )

C.kpi_row(
    [
        ("Features", f"{feats[0]}, {feats[1]}, {feats[2]}", "mean abs, rms, peak"),
        ("FP32 logit", f"{fp32_logit:+.3f}",
         "Anomaly" if fp32_pred else "Normal"),
        ("INT8 logit", f"{int8_logit:+.3f}",
         "Anomaly" if int8_pred else "Normal"),
        ("Logit error", f"{abs(fp32_logit - int8_logit):.3f}", "absolute"),
        ("Agreement", "yes" if fp32_pred == int8_pred else "NO",
         "FP32 vs INT8 class"),
    ]
)

if true_label is not None:
    correct = int8_pred == true_label
    st.markdown(
        C.status_chip(
            f"ground truth: {'Abnormal' if true_label else 'Healthy'}",
            "good" if correct else "critical",
            palette,
        )
        + C.status_chip(
            "classified correctly" if correct else "misclassified",
            "good" if correct else "critical",
            palette,
        ),
        unsafe_allow_html=True,
    )

if is_saturated:
    st.warning(
        "This input saturates the INT8 quantizer — at least one normalized "
        f"feature exceeds the representable range (|z| ≤ {limit:.4f}). The "
        "reported INT8 logit is compressed.",
        icon=":material/warning:",
    )

# ----------------------------------------------------------------------
# Layer-by-layer trace
# ----------------------------------------------------------------------

st.subheader("Layer-by-layer INT8 trace")

C.note(
    "The same intermediates <code>firmware/debug_int8_host.c</code> prints, "
    "stage by stage through <code>tinyguard_int8_predict()</code>."
)

z = out["normalized"].ravel()
q_in = out["q_input"].ravel()

st.markdown("**1 — Normalize and quantize the inputs**")
st.dataframe(
    pd.DataFrame(
        {
            "Feature": FEATURES,
            "Raw": list(feats),
            "Normalized (z)": [round(float(v), 6) for v in z],
            "z / INPUT_SCALE": [
                round(float(v) / float(int8_header["input_scale"]), 2) for v in z
            ],
            "INT8 input": [int(v) for v in q_in],
            "Clipped": [bool(abs(int(v)) >= 127) for v in q_in],
        }
    ),
    width="stretch",
    hide_index=True,
)

acc1 = out["acc1"].ravel()
q_hidden = out["q_hidden"].ravel()

st.markdown("**2 — Dense layer, ReLU, requantize the hidden activations**")

hidden_table = pd.DataFrame(
    {
        "Hidden unit": list(range(len(acc1))),
        "INT32 accumulator (post-ReLU)": [int(v) for v in acc1],
        "× ACC1_SCALE": [
            round(float(v) * float(int8_header["acc1_scale"]), 5) for v in acc1
        ],
        "INT8 hidden": [int(v) for v in q_hidden],
        "W2": [int(v) for v in int8_header["weights2"]],
        "Contribution": [
            int(h) * int(w)
            for h, w in zip(q_hidden, int8_header["weights2"])
        ],
    }
)

fig = go.Figure(
    go.Bar(
        x=hidden_table["Hidden unit"],
        y=hidden_table["Contribution"],
        marker=dict(
            color=[
                palette.series[0] if v >= 0 else palette.series[1]
                for v in hidden_table["Contribution"]
            ],
            line=dict(width=0),
        ),
        width=0.6,
        hovertemplate="unit %{x}<br>contribution %{y}<extra></extra>",
    )
)
fig.add_hline(y=0, line=dict(color=palette.axis, width=1))
layout = plotly_layout(palette, height=260)
layout["xaxis"].update(title=dict(text="Hidden unit"), dtick=1, showgrid=False)
layout["yaxis"].update(title=dict(text="Contribution to output accumulator"))
fig.update_layout(**layout)
fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))

C.chart(fig, hidden_table, label="hidden layer trace")

st.caption(
    "Positive contributions push toward *anomaly*, negative toward *normal*. "
    "Blue is positive, orange negative — a diverging pair around a zero "
    "baseline, which is what the data's polarity calls for."
)

acc2 = int(out["acc2"].ravel()[0])

st.markdown("**3 — Output accumulator and decision**")
C.kv_table(
    [
        ("Bias2", str(int(int8_header["bias2"][0]))),
        ("Sum of contributions", str(int(hidden_table["Contribution"].sum()))),
        ("INT32 output accumulator", str(acc2)),
        ("× ACC2_SCALE", f"{int8_logit:+.6f}"),
        ("Decision rule", "accumulator >= 0 → anomaly"),
        ("Prediction", "Anomaly" if int8_pred == 1 else "Normal"),
    ],
    palette,
)

st.caption(
    "The class decision reads the **integer** accumulator, not the scaled "
    "logit, so the float scale never affects the outcome — which is why the "
    "1-ULP scale representation gap cannot change a prediction."
)
