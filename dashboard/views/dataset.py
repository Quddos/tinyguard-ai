"""Dataset explorer: the actual synthetic windows, features and separability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import components as C
from lib.data import FEATURES, SAMPLE_COLUMNS, require_artifacts
from lib.theme import active_palette, plotly_layout

palette = active_palette()
df, results, int8_header, fp32_header = require_artifacts()

dataset = results["dataset_summary"]
prov = results["provenance"]

HEALTHY, ABNORMAL = palette.series[0], palette.series[1]

st.title("Dataset")
st.caption(
    f"{dataset['total_windows']:,} synthetic vibration windows, 16 samples each, "
    "balanced across two classes."
)

C.limitation_banner(prov["limitations"])

# ----------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------

st.subheader("Composition")

scenario_counts = dataset["scenario_distribution"]
ordered = sorted(scenario_counts.items(), key=lambda kv: -kv[1])

fig = C.ranked_bars(
    [k.replace("_", " ") for k, _ in ordered],
    [v for _, v in ordered],
    palette,
    value_fmt="{:,.0f}",
    x_title="Windows",
)
C.chart(
    fig,
    pd.DataFrame(
        {
            "Scenario": [k for k, _ in ordered],
            "Windows": [v for _, v in ordered],
            "Label": [0 if k == "healthy" else 1 for k, _ in ordered],
        }
    ),
    label="scenario counts",
)

C.note(
    "One healthy scenario and three fault scenarios. The three fault types are "
    "pooled into a single <em>abnormal</em> class for training, so the model is "
    "binary."
)

# ----------------------------------------------------------------------
# Feature distributions
# ----------------------------------------------------------------------

st.subheader("Feature distributions by class")

feature = st.radio(
    "Feature",
    FEATURES,
    horizontal=True,
    format_func=lambda s: s.replace("_", " "),
    label_visibility="collapsed",
)

healthy_vals = df.loc[df.label == 0, feature]
abnormal_vals = df.loc[df.label == 1, feature]

lo = int(min(healthy_vals.min(), abnormal_vals.min()))
hi = int(max(healthy_vals.max(), abnormal_vals.max()))
bins = dict(start=lo, end=hi + 1, size=max(1, (hi - lo) // 60))

fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=healthy_vals,
        name="Healthy",
        marker=dict(color=HEALTHY, line=dict(width=0)),
        xbins=bins,
        opacity=0.85,
        hovertemplate="Healthy<br>%{x}: %{y} windows<extra></extra>",
    )
)
fig.add_trace(
    go.Histogram(
        x=abnormal_vals,
        name="Abnormal",
        marker=dict(color=ABNORMAL, line=dict(width=0)),
        xbins=bins,
        opacity=0.85,
        hovertemplate="Abnormal<br>%{x}: %{y} windows<extra></extra>",
    )
)

layout = plotly_layout(palette, height=340, showlegend=True)
layout["xaxis"].update(title=dict(text=feature.replace("_", " ")))
layout["yaxis"].update(title=dict(text="Windows"))
fig.update_layout(barmode="overlay", bargap=0.02, **layout)
C.chart(
    fig,
    pd.DataFrame(
        {
            "Class": ["Healthy", "Abnormal"],
            "Min": [healthy_vals.min(), abnormal_vals.min()],
            "Median": [healthy_vals.median(), abnormal_vals.median()],
            "Max": [healthy_vals.max(), abnormal_vals.max()],
            "Mean": [round(healthy_vals.mean(), 2), round(abnormal_vals.mean(), 2)],
        }
    ),
    label=f"{feature} summary",
)

overlap = dataset["per_feature_class_overlap"][feature]
st.markdown(
    C.status_chip(
        f"healthy max {overlap['healthy_max']:.0f} vs abnormal min {overlap['abnormal_min']:.0f}",
        "warning" if overlap["overlaps"] else "good",
        palette,
    )
    + C.status_chip(
        "ranges overlap" if overlap["overlaps"] else "ranges disjoint",
        "warning" if overlap["overlaps"] else "good",
        palette,
    ),
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Separability — the central caveat, shown rather than asserted
# ----------------------------------------------------------------------

st.subheader("Why every classifier scores 100%")

C.note(
    "Each feature taken alone overlaps slightly between classes. Taken together "
    "they separate almost perfectly, which is a property of how "
    "<code>generate_dataset.py</code> draws the two classes — not a property of "
    "the model. This is the single most important caveat in the project."
)

x_feat, y_feat = st.columns(2)
with x_feat:
    fx = st.selectbox("Horizontal axis", FEATURES, index=0,
                      format_func=lambda s: s.replace("_", " "))
with y_feat:
    fy = st.selectbox("Vertical axis", FEATURES, index=1,
                      format_func=lambda s: s.replace("_", " "))

# Jitter: features are coarse integers, so thousands of windows land on
# identical coordinates. A small deterministic jitter makes density visible
# without moving any point far enough to misrepresent its value.
rng = np.random.default_rng(42)
jitter_x = rng.uniform(-0.35, 0.35, len(df))
jitter_y = rng.uniform(-0.35, 0.35, len(df))

fig = go.Figure()
for label, name, colour in ((0, "Healthy", HEALTHY), (1, "Abnormal", ABNORMAL)):
    mask = (df.label == label).to_numpy()
    fig.add_trace(
        go.Scattergl(
            x=df.loc[mask, fx].to_numpy() + jitter_x[mask],
            y=df.loc[mask, fy].to_numpy() + jitter_y[mask],
            mode="markers",
            name=name,
            marker=dict(color=colour, size=5, opacity=0.5,
                        line=dict(width=0)),
            customdata=np.stack(
                [df.loc[mask, fx], df.loc[mask, fy], df.loc[mask, "scenario"]],
                axis=-1,
            ),
            hovertemplate=(
                f"{name}<br>{fx}: %{{customdata[0]}}"
                f"<br>{fy}: %{{customdata[1]}}"
                "<br>%{customdata[2]}<extra></extra>"
            ),
        )
    )

layout = plotly_layout(palette, height=420, showlegend=True)
layout["xaxis"].update(title=dict(text=fx.replace("_", " ")))
layout["yaxis"].update(title=dict(text=fy.replace("_", " ")))
fig.update_layout(**layout)
C.chart(
    fig,
    df.groupby("class_name")[[fx, fy]]
    .agg(["min", "median", "max"])
    .round(2)
    .reset_index(),
    label=f"{fx} vs {fy} by class",
)

st.caption(
    "Points carry a small deterministic jitter (±0.35) so overlapping windows "
    "are visible. Hover values are the true integers."
)

red = dataset["redundancy"]
leak = dataset["naive_split_leakage"]

C.kpi_row(
    [
        ("Unique feature vectors", f"{red['unique_feature_vectors']:,}",
         f"of {dataset['total_windows']:,} rows"),
        ("Duplicate rows", f"{red['duplicate_rows']:,}",
         "coarse integer features"),
        ("Conflicting vectors", f"{red['conflicting_duplicate_vectors']}",
         "no vector carries both labels"),
        ("Naive split leakage", f"{leak['fraction'] * 100:.1f}%",
         "test rows also seen in training"),
    ]
)

C.note(
    "The leakage is real and worth knowing about, but it is not the cause of the "
    "perfect score: a leakage-free grouped split reaches the same figure. "
    "See the Model page."
)

# ----------------------------------------------------------------------
# Raw waveform inspector
# ----------------------------------------------------------------------

st.subheader("Raw window inspector")

C.note(
    "The committed dataset keeps all 16 raw samples per window alongside the "
    "three derived features, so any window can be re-derived and audited."
)

controls = st.columns([1, 1, 2])
with controls[0]:
    scenario = st.selectbox(
        "Scenario", sorted(df["scenario"].unique()),
        format_func=lambda s: s.replace("_", " "),
    )
with controls[1]:
    subset = df[df.scenario == scenario].reset_index(drop=True)
    idx = st.number_input(
        "Window", min_value=0, max_value=len(subset) - 1, value=0, step=1,
        help=f"{len(subset):,} windows in this scenario",
    )

row = subset.iloc[int(idx)]
samples = [int(row[c]) for c in SAMPLE_COLUMNS]
colour = HEALTHY if int(row["label"]) == 0 else ABNORMAL

fig = go.Figure()
fig.add_hline(y=0, line=dict(color=palette.axis, width=1))
fig.add_trace(
    go.Scatter(
        x=list(range(16)),
        y=samples,
        mode="lines+markers",
        line=dict(color=colour, width=2, shape="linear"),
        marker=dict(color=colour, size=8, line=dict(width=2, color=palette.surface)),
        hovertemplate="sample %{x}: %{y} mg<extra></extra>",
        showlegend=False,
    )
)
# Peak is one of the three model inputs, so label it directly.
peak_i = int(np.argmax(np.abs(samples)))
fig.add_annotation(
    x=peak_i, y=samples[peak_i],
    text=f"peak {abs(samples[peak_i])}",
    showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=palette.ink_muted,
    ax=0, ay=-28,
    font=dict(color=palette.ink_secondary, size=11),
)

layout = plotly_layout(palette, height=300)
layout["xaxis"].update(title=dict(text="Sample index"), dtick=1)
layout["yaxis"].update(title=dict(text="Acceleration deviation (mg)"))
fig.update_layout(**layout)

C.chart(
    fig,
    pd.DataFrame({"Sample index": list(range(16)), "Value (mg)": samples}),
    label="raw samples",
)

C.kpi_row(
    [
        ("Class", str(row["class_name"]), scenario.replace("_", " ")),
        ("Mean absolute", f"{int(row['mean_absolute'])}", "mg"),
        ("RMS", f"{int(row['rms'])}", "mg, integer sqrt"),
        ("Peak", f"{int(row['peak'])}", "mg"),
    ]
)

st.caption(
    "Features use truncating integer arithmetic and an integer square root, "
    "matching `firmware/signal_features.c` exactly — so these are the values "
    "the firmware would compute for this window."
)
