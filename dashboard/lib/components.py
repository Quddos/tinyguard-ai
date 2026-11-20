"""
Reusable presentation pieces.

Two conventions run through all of them:

  * A chart never carries a value the reader cannot get another way. Every chart
    ships a table view beside it, so no number is reachable only by hovering.
  * A single headline number is a stat tile, not a one-bar bar chart.
"""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .theme import Palette, plotly_layout


def limitation_banner(limitations: list[str]) -> None:
    """
    The standing scientific limitations, shown on every page.

    These are read from results/provenance.json, so they cannot drift from the
    generated artifacts.
    """
    items = "".join(f"<li>{html.escape(x)}</li>" for x in limitations)
    st.markdown(
        f"""
        <div class="tg-banner">
          <div class="tg-banner-title">Scientific limitations</div>
          <ul style="margin:0 0 0 18px; padding:0;">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(tiles: list[tuple[str, str, str]]) -> None:
    """A row of stat tiles: (label, value, sub-caption)."""
    cards = "".join(
        f"""<div class="tg-kpi">
              <div class="tg-kpi-label">{html.escape(label)}</div>
              <div class="tg-kpi-value">{html.escape(value)}</div>
              <div class="tg-kpi-sub">{html.escape(sub)}</div>
            </div>"""
        for label, value, sub in tiles
    )
    st.markdown(f'<div class="tg-kpi-row">{cards}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="tg-note">{text}</div>', unsafe_allow_html=True)


def meter_list(
    rows: list[tuple[str, float, str]],
    palette: Palette,
    *,
    limit: float = 1.0,
) -> None:
    """
    One named ratio per row, each against the same limit, bars aligned.

    Used instead of a bar chart where every value sits near the ceiling: the
    bars would be indistinguishable, but the figures still need reading.
    """
    html_rows = []
    for name, value, label in rows:
        pct = 0.0 if limit == 0 else max(0.0, min(1.0, value / limit)) * 100
        html_rows.append(
            f"""<div class="tg-meter-row">
                  <div class="tg-meter-name">{html.escape(name)}</div>
                  <div class="tg-meter-track">
                    <div class="tg-meter-fill" style="width:{pct:.4f}%"></div>
                  </div>
                  <div class="tg-meter-value">{html.escape(label)}</div>
                </div>"""
        )
    st.markdown("".join(html_rows), unsafe_allow_html=True)


def chart(fig: go.Figure, table: pd.DataFrame | None = None, *, label: str = "values") -> None:
    """
    Render a chart with its table-view twin.

    The table is the accessible equivalent, not an extra: it carries every value
    the chart encodes, for colour-vision, screen-reader and export cases.
    """
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    if table is not None:
        with st.expander(f"Table view — {label}"):
            st.dataframe(table, width="stretch", hide_index=True)


def ranked_bars(
    names: list[str],
    values: list[float],
    palette: Palette,
    *,
    value_fmt: str = "{:.3f}",
    emphasis: str | None = None,
    x_title: str = "",
    height: int | None = None,
    x_range: tuple[float, float] | None = None,
) -> go.Figure:
    """
    Horizontal bars on one baseline, one hue.

    Nominal categories get a single colour — colouring each bar by its own value
    would double-encode length as hue. `emphasis` optionally highlights one
    named bar and greys the rest, which is the honest form when one of the bars
    is the subject and the others are context.
    """
    if emphasis is None:
        colors = [palette.series[0]] * len(names)
    else:
        colors = [
            palette.series[0] if n == emphasis else palette.muted_mark
            for n in names
        ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            width=0.62,
            text=[value_fmt.format(v) for v in values],
            textposition="outside",
            textfont=dict(color=palette.ink_secondary, size=12),
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x}<extra></extra>",
        )
    )

    layout = plotly_layout(palette, height=height or max(180, 42 * len(names) + 70))
    layout["xaxis"].update(title=dict(text=x_title), showgrid=True)
    layout["yaxis"].update(showgrid=False, autorange="reversed")
    if x_range is not None:
        layout["xaxis"]["range"] = list(x_range)
    fig.update_layout(**layout)
    fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))

    return fig


def confusion_grid(cm: dict, palette: Palette, *, title: str = "") -> None:
    """
    A 2x2 confusion matrix as a labelled grid.

    Four numbers do not need a heatmap; the figures themselves are the content,
    so they are printed rather than encoded as colour.
    """
    tn, fp = int(cm["true_negative"]), int(cm["false_positive"])
    fn, tp = int(cm["false_negative"]), int(cm["true_positive"])

    def cell(value: int, is_error: bool) -> str:
        colour = palette.critical if (is_error and value > 0) else palette.ink_primary
        return (
            f'<td style="padding:12px 16px;text-align:right;'
            f'font-variant-numeric:tabular-nums;font-size:19px;font-weight:600;'
            f'color:{colour};border:1px solid {palette.grid};">{value}</td>'
        )

    head = (
        f'<th style="padding:8px 16px;font-size:11px;letter-spacing:.06em;'
        f'text-transform:uppercase;color:{palette.ink_muted};'
        f'font-weight:500;text-align:right;">'
    )
    side = (
        f'<th style="padding:12px 16px;font-size:12px;color:{palette.ink_secondary};'
        f'font-weight:500;text-align:left;border:1px solid {palette.grid};">'
    )

    st.markdown(
        f"""
        <table style="border-collapse:collapse;margin:6px 0 14px 0;">
          <tr>
            <th></th>{head}Predicted healthy</th>{head}Predicted abnormal</th>
          </tr>
          <tr>{side}Actually healthy</th>{cell(tn, False)}{cell(fp, True)}</tr>
          <tr>{side}Actually abnormal</th>{cell(fn, True)}{cell(tp, False)}</tr>
        </table>
        """,
        unsafe_allow_html=True,
    )


def kv_table(rows: list[tuple[str, str]], palette: Palette) -> None:
    """A two-column definition table for scales, hashes and environment."""
    body = "".join(
        f"""<tr>
              <td style="padding:6px 18px 6px 0;font-size:12.5px;
                         color:{palette.ink_secondary};
                         border-bottom:1px solid {palette.grid};
                         white-space:nowrap;">{html.escape(k)}</td>
              <td style="padding:6px 0;font-size:12.5px;color:{palette.ink_primary};
                         font-variant-numeric:tabular-nums;
                         border-bottom:1px solid {palette.grid};
                         word-break:break-all;">{html.escape(v)}</td>
            </tr>"""
        for k, v in rows
    )
    st.markdown(
        f'<table style="border-collapse:collapse;width:100%;margin:4px 0 16px 0;">'
        f"{body}</table>",
        unsafe_allow_html=True,
    )


def status_chip(text: str, state: str, palette: Palette) -> str:
    """
    A status pill. Status colour always ships with its label, never alone, so
    the meaning survives colour-vision deficiency and monochrome printing.
    """
    colour = {
        "good": palette.good,
        "warning": palette.warning,
        "critical": palette.critical,
    }.get(state, palette.ink_muted)

    glyph = {"good": "✓", "warning": "!", "critical": "✕"}.get(state, "·")

    return (
        f'<span class="tg-chip" style="border-color:{colour};color:{palette.ink_primary};">'
        f'<span style="color:{colour};font-weight:700;">{glyph}</span> '
        f"{html.escape(text)}</span>"
    )
