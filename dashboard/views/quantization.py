"""Quantization: fidelity, error, saturation and footprint."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import components as C
from lib.data import FEATURES, require_artifacts
from lib.inference import fp32_forward, int8_forward, representable_input_limit
from lib.theme import active_palette, plotly_layout

palette = active_palette()
df, results, int8_header, fp32_header = require_artifacts()

quant = results["quantization"]
prov = results["provenance"]

st.title("Quantization")
st.caption(quant["scheme"] + ".")

C.limitation_banner(prov["limitations"])

# ----------------------------------------------------------------------
# Fidelity headline
# ----------------------------------------------------------------------

fid = quant["fidelity_whole_dataset"]

C.kpi_row(
    [
        ("Prediction agreement", f"{fid['prediction_agreement'] * 100:.2f}%",
         f"all {fid['n']:,} windows"),
        ("Mismatches", f"{fid['prediction_mismatches']}", "FP32 vs INT8 class"),
        ("Mean logit error", f"{fid['logit_abs_error_mean']:.3f}", "absolute"),
        ("95th pct logit error", f"{fid['logit_abs_error_p95']:.3f}", "absolute"),
        ("Max logit error", f"{fid['logit_abs_error_max']:.3f}", "absolute"),
    ]
)

# ----------------------------------------------------------------------
# FP32 vs INT8 logits, computed live from the committed headers
# ----------------------------------------------------------------------

st.subheader("FP32 against INT8, window by window")

C.note(
    "Computed live from <code>firmware/model_params.h</code> and "
    "<code>firmware/model_int8_params.h</code> — the constants the firmware "
    "compiles — using C's float32 arithmetic and round-away-from-zero cast. "
    "These are the on-target numbers, not an approximation of them."
)

X = df[FEATURES].to_numpy(dtype=np.float64)

fp32_logits = fp32_forward(X, fp32_header).ravel()
int8_out = int8_forward(X, int8_header)
int8_logits = int8_out["logit"].ravel()
saturated = int8_out["saturated"]

lo = float(min(fp32_logits.min(), int8_logits.min()))
hi = float(max(fp32_logits.max(), int8_logits.max()))
pad = (hi - lo) * 0.05

fig = go.Figure()

# Identity reference: perfect agreement lies on this line.
fig.add_trace(
    go.Scatter(
        x=[lo - pad, hi + pad],
        y=[lo - pad, hi + pad],
        mode="lines",
        line=dict(color=palette.axis, width=1),
        name="Perfect agreement",
        hoverinfo="skip",
    )
)

fig.add_trace(
    go.Scattergl(
        x=fp32_logits[~saturated],
        y=int8_logits[~saturated],
        mode="markers",
        name="Within INT8 input range",
        marker=dict(color=palette.series[0], size=5, opacity=0.45,
                    line=dict(width=0)),
        hovertemplate="FP32 %{x:.2f}<br>INT8 %{y:.2f}<extra></extra>",
    )
)

if saturated.any():
    fig.add_trace(
        go.Scatter(
            x=fp32_logits[saturated],
            y=int8_logits[saturated],
            mode="markers",
            name="Input saturated",
            marker=dict(color=palette.series[1], size=11, symbol="diamond",
                        line=dict(width=2, color=palette.surface)),
            hovertemplate=(
                "Saturated<br>FP32 %{x:.2f}<br>INT8 %{y:.2f}<extra></extra>"
            ),
        )
    )

layout = plotly_layout(palette, height=430, showlegend=True)
layout["xaxis"].update(title=dict(text="FP32 logit"))
layout["yaxis"].update(title=dict(text="INT8 logit"))
fig.update_layout(**layout)

C.chart(
    fig,
    pd.DataFrame(
        {
            "Statistic": [
                "Windows", "Saturated inputs", "Mean |error|",
                "95th pct |error|", "Max |error|", "Prediction mismatches",
            ],
            "Value": [
                fid["n"], fid["windows_with_saturated_input"],
                round(fid["logit_abs_error_mean"], 6),
                round(fid["logit_abs_error_p95"], 6),
                round(fid["logit_abs_error_max"], 6),
                fid["prediction_mismatches"],
            ],
        }
    ),
    label="FP32 vs INT8 agreement",
)

st.caption(
    "Points sit on the identity line where the quantized model reproduces the "
    "float logit. Spread grows with magnitude, which is expected for a "
    "per-tensor scheme — but no point crosses zero, so no class flips."
)

# ----------------------------------------------------------------------
# Error distribution
# ----------------------------------------------------------------------

st.subheader("Logit error distribution")

err = np.abs(fp32_logits - int8_logits.astype(np.float64))

fig = go.Figure(
    go.Histogram(
        x=err,
        nbinsx=70,
        marker=dict(color=palette.series[0], line=dict(width=0)),
        hovertemplate="|error| %{x:.3f}: %{y} windows<extra></extra>",
    )
)
for value, label in (
    (float(err.mean()), "mean"),
    (float(np.percentile(err, 95)), "95th pct"),
):
    fig.add_vline(
        x=value,
        line=dict(color=palette.ink_muted, width=1),
        annotation_text=f"{label} {value:.3f}",
        annotation_position="top",
        annotation_font=dict(color=palette.ink_secondary, size=11),
    )

layout = plotly_layout(palette, height=320)
layout["xaxis"].update(title=dict(text="Absolute logit error"))
layout["yaxis"].update(title=dict(text="Windows"))
fig.update_layout(**layout)

C.chart(
    fig,
    pd.DataFrame(
        {
            "Percentile": ["min", "25", "50", "75", "95", "99", "max"],
            "Absolute error": [
                round(float(np.percentile(err, p)), 6)
                for p in (0, 25, 50, 75, 95, 99, 100)
            ],
        }
    ),
    label="error percentiles",
)

st.divider()

# ----------------------------------------------------------------------
# Saturation — the known limitation
# ----------------------------------------------------------------------

st.subheader("Input quantizer headroom")

limit = representable_input_limit(int8_header)
observed = float(
    np.abs((np.float32(X) - int8_header["scaler_mean"]) / int8_header["scaler_std"]).max()
)

C.note(
    "<code>INPUT_SCALE</code> is calibrated on the training split only, so the "
    "representable normalized input range is "
    f"<code>|z| ≤ 127 × INPUT_SCALE = {limit:.4f}</code>. Anything beyond that "
    "clips at ±127 and the logit is compressed. The dataset only just fits."
)

C.meter_list(
    [("Dataset max |z|", observed, f"{observed:.4f}")],
    palette,
    limit=limit,
)

C.kpi_row(
    [
        ("Representable |z|", f"{limit:.4f}", "127 × INPUT_SCALE"),
        ("Dataset max |z|", f"{observed:.4f}", "all 3000 windows"),
        ("Headroom", f"{(limit - observed):.2e}", "margin before clipping"),
        ("Saturated windows", f"{int(saturated.sum())}",
         f"{saturated.mean() * 100:.2f}% of dataset"),
    ]
)

st.warning(
    "The dataset sits at 99.99999% of the representable range. Any future "
    "dataset with stronger faults will saturate and have its logits compressed "
    "unless INPUT_SCALE is recalibrated. The firmware keeps one deliberately "
    "severe window past this limit to demonstrate the effect — see the "
    "Firmware page.",
    icon=":material/warning:",
)

st.divider()

# ----------------------------------------------------------------------
# Scales and per-tensor error
# ----------------------------------------------------------------------

st.subheader("Quantization scales")

scales = quant["scales"]
C.kv_table(
    [
        ("INPUT_SCALE", f"{scales['input']:.12g}"),
        ("W1_SCALE", f"{scales['layer1_weight']:.12g}"),
        ("ACC1_SCALE", f"{scales['layer1_accumulator']:.12g}"),
        ("HIDDEN_SCALE", f"{scales['hidden_activation']:.12g}"),
        ("W2_SCALE", f"{scales['layer2_weight']:.12g}"),
        ("ACC2_SCALE", f"{scales['layer2_accumulator']:.12g}"),
        ("Calibration partition",
         f"{quant['calibration']['partition']} (n = {quant['calibration']['n']})"),
    ],
    palette,
)

st.subheader("Per-tensor reconstruction error")

C.note(
    "How much precision each tensor loses when rounded to INT8 and scaled back. "
    "Shown as a table: six tensors with two error measures each read better as "
    "figures than as marks."
)

tensors = quant["tensor_statistics"]
st.dataframe(
    pd.DataFrame(
        {
            "Tensor": [k.replace("_", " ") for k in tensors],
            "Shape": [str(tuple(v["shape"])) for v in tensors.values()],
            "Min": [round(v["min"], 5) for v in tensors.values()],
            "Max": [round(v["max"], 5) for v in tensors.values()],
            "INT8 scale": [float(f"{v['int8_scale']:.6g}") for v in tensors.values()],
            "Max |error|": [round(v["reconstruction_error_max"], 6) for v in tensors.values()],
            "Mean |error|": [round(v["reconstruction_error_mean"], 6) for v in tensors.values()],
        }
    ),
    width="stretch",
    hide_index=True,
)

st.divider()

# ----------------------------------------------------------------------
# Footprint — both ratios, so the weight-only figure isn't mistaken for the whole
# ----------------------------------------------------------------------

st.subheader("Parameter footprint")

fp = quant["parameter_footprint_bytes"]

footprint_table = pd.DataFrame(
    {
        "Component": ["Weight tensors", "Biases", "StandardScaler", "Total"],
        "FP32 bytes": [
            fp["fp32_weights_only"],
            fp["int8_biases_only"],
            fp["scaler_float32"],
            fp["fp32_total"],
        ],
        "INT8 bytes": [
            fp["int8_weights_only"],
            fp["int8_biases_only"],
            fp["scaler_float32"],
            fp["int8_total"],
        ],
    }
)
footprint_table["Reduction"] = (
    footprint_table["FP32 bytes"] / footprint_table["INT8 bytes"]
).round(2)

st.dataframe(footprint_table, width="stretch", hide_index=True)

st.info(fp["note"], icon=":material/info:")

# ----------------------------------------------------------------------
# Header consistency proof
# ----------------------------------------------------------------------

st.subheader("Header consistency")

eq = quant["scale_representation_equivalence"]

st.markdown(
    C.status_chip(
        "integer tensors bit-exact" if quant["integer_tensors_bit_exact"]
        else "integer tensors DRIFTED",
        "good" if quant["integer_tensors_bit_exact"] else "critical",
        palette,
    )
    + C.status_chip(
        f"max scale gap {eq['max_scale_ulp_distance']} ULP",
        "good" if eq["max_scale_ulp_distance"] <= 1 else "warning",
        palette,
    )
    + C.status_chip(
        "predictions identical" if eq["identical_predictions"]
        else "predictions differ",
        "good" if eq["identical_predictions"] else "critical",
        palette,
    ),
    unsafe_allow_html=True,
)

C.kv_table(
    [
        ("Header file", quant["header_file"]),
        ("Header SHA-256", quant["header_sha256"]),
        ("Exporter reproduces header",
         str(quant["exporter_reproduces_committed_header"])),
        ("Identical quantized inputs", str(eq["identical_quantized_inputs"])),
        ("Identical accumulators", str(eq["identical_output_accumulators"])),
        ("Max logit difference between scale sets",
         f"{eq['max_logit_difference']:.3e}"),
        ("Smallest decision margin",
         f"{eq['min_abs_output_accumulator']} accumulator counts"),
    ],
    palette,
)

with st.expander("Why the scales differ by 1 ULP"):
    st.markdown(eq["note"])
    st.markdown(
        "Each scale is computed in float64 and written to the header as "
        "12-significant-digit decimal text. Parsing that text back as `float32` "
        "can land one unit-in-the-last-place away from casting the float64 "
        "directly. The integer tensors — which alone decide the class — are "
        "bit-exact, and the gap leaves every quantized input, accumulator and "
        "prediction unchanged across the whole dataset."
    )
