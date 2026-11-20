"""
Chart theme for the TinyGuard-AI dashboard.

Colour decisions are deliberate, not decorative:

  * Three categorical slots only. The dashboard never shows more than three
    series at once, which keeps scatter plots inside the all-pairs colour-vision
    gate. Slots are assigned in fixed order and follow the entity, never its
    rank, so filtering never repaints a series.
  * Magnitude uses one blue hue, light to dark. No rainbow ramps.
  * Status colours (good/warning/critical) are reserved for state and never
    reused as a fourth series.
  * Dark mode uses its own steps chosen for the dark surface, not an automatic
    inversion of the light palette.

Both palettes were checked with the data-visualisation validator for lightness
band, chroma floor, colour-vision separation, normal-vision separation and
contrast, under the all-pairs rule, against the surfaces below - which are
Streamlit's own light (#ffffff) and dark (#0e1117) backgrounds, so the check
matches what the charts actually render on. Light slot 3 (aqua) measures 2.82:1
on the light surface, just under 3:1, so every chart using it ships visible
labels and a table view rather than relying on the fill alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st


@dataclass(frozen=True)
class Palette:
    mode: str

    # categorical slots, fixed order
    series: tuple[str, str, str]

    # surfaces and ink
    surface: str
    page: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    border: str

    # de-emphasis for the emphasis form
    muted_mark: str

    # single-hue sequential ramp, light -> dark
    sequential: tuple[str, ...] = field(default=())

    # reserved status colours
    good: str = "#0ca30c"
    warning: str = "#fab219"
    critical: str = "#d03b3b"


_SEQUENTIAL = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
)

LIGHT = Palette(
    mode="light",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    surface="#ffffff",
    page="#f0f2f6",
    ink_primary="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    border="rgba(11,11,11,0.10)",
    muted_mark="#c3c2b7",
    sequential=_SEQUENTIAL,
)

DARK = Palette(
    mode="dark",
    series=("#3987e5", "#d95926", "#199e70"),
    surface="#0e1117",
    page="#262730",
    ink_primary="#f0efec",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    border="rgba(255,255,255,0.10)",
    muted_mark="#52514e",
    sequential=_SEQUENTIAL,
)

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def active_palette() -> Palette:
    """
    Resolve the palette from the viewer's active Streamlit theme.

    Falls back to light if the running Streamlit cannot report a theme, so the
    dashboard never renders against an unknown surface.
    """
    try:
        theme = getattr(st.context, "theme", None)
        if theme is not None and getattr(theme, "type", None) == "dark":
            return DARK
    except Exception:
        pass
    return LIGHT


def plotly_layout(palette: Palette, *, height: int = 360, showlegend: bool = False) -> dict:
    """
    Shared Plotly layout: hairline recessive grid, no chart-level title (the
    Streamlit heading carries it), generous padding, unified hover.
    """
    axis = dict(
        showgrid=True,
        gridcolor=palette.grid,
        gridwidth=1,
        zeroline=False,
        linecolor=palette.axis,
        linewidth=1,
        ticks="outside",
        tickcolor=palette.axis,
        ticklen=4,
        tickfont=dict(color=palette.ink_muted, size=12),
        title=dict(font=dict(color=palette.ink_secondary, size=12)),
        automargin=True,
    )

    return dict(
        height=height,
        showlegend=showlegend,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=palette.ink_secondary, size=12),
        margin=dict(l=8, r=16, t=16, b=8),
        xaxis=dict(axis),
        yaxis=dict(axis),
        hoverlabel=dict(
            bgcolor=palette.surface,
            bordercolor=palette.axis,
            font=dict(family=FONT_STACK, color=palette.ink_primary, size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=palette.ink_secondary, size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
    )


def inject_css(palette: Palette) -> None:
    """Page-level styling: surfaces, stat tiles, banners, hairline rules."""
    st.markdown(
        f"""
        <style>
          :root {{
            --tg-surface: {palette.surface};
            --tg-page: {palette.page};
            --tg-ink: {palette.ink_primary};
            --tg-ink-2: {palette.ink_secondary};
            --tg-ink-3: {palette.ink_muted};
            --tg-border: {palette.border};
            --tg-grid: {palette.grid};
            --tg-s1: {palette.series[0]};
            --tg-s2: {palette.series[1]};
            --tg-s3: {palette.series[2]};
            --tg-warning: {palette.warning};
            --tg-critical: {palette.critical};
            --tg-good: {palette.good};
          }}

          .tg-kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            margin: 4px 0 20px 0;
          }}
          .tg-kpi {{
            background: var(--tg-surface);
            border: 1px solid var(--tg-border);
            border-radius: 10px;
            padding: 14px 16px;
          }}
          .tg-kpi-label {{
            font-family: {FONT_STACK};
            font-size: 11px;
            letter-spacing: .06em;
            text-transform: uppercase;
            color: var(--tg-ink-3);
            margin-bottom: 6px;
          }}
          .tg-kpi-value {{
            font-family: {FONT_STACK};
            font-size: 30px;
            line-height: 1.1;
            font-weight: 600;
            color: var(--tg-ink);
          }}
          .tg-kpi-sub {{
            font-family: {FONT_STACK};
            font-size: 12px;
            color: var(--tg-ink-2);
            margin-top: 4px;
          }}

          .tg-hero {{
            font-family: {FONT_STACK};
            font-size: 52px;
            font-weight: 650;
            line-height: 1.05;
            color: var(--tg-ink);
          }}

          .tg-banner {{
            border: 1px solid var(--tg-border);
            border-left: 3px solid var(--tg-warning);
            background: var(--tg-surface);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 18px;
            font-family: {FONT_STACK};
            font-size: 13px;
            color: var(--tg-ink-2);
          }}
          .tg-banner strong {{ color: var(--tg-ink); }}
          .tg-banner-title {{
            font-size: 12px;
            letter-spacing: .06em;
            text-transform: uppercase;
            color: var(--tg-ink-3);
            margin-bottom: 6px;
          }}

          .tg-note {{
            font-family: {FONT_STACK};
            font-size: 12.5px;
            color: var(--tg-ink-2);
            border-left: 2px solid var(--tg-grid);
            padding: 2px 0 2px 12px;
            margin: 6px 0 16px 0;
          }}

          .tg-meter-row {{
            display: grid;
            grid-template-columns: 180px 1fr 72px;
            align-items: center;
            gap: 12px;
            padding: 7px 0;
            border-bottom: 1px solid var(--tg-grid);
          }}
          .tg-meter-name {{
            font-family: {FONT_STACK};
            font-size: 13px;
            color: var(--tg-ink);
          }}
          .tg-meter-track {{
            background: var(--tg-grid);
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
          }}
          .tg-meter-fill {{
            height: 100%;
            border-radius: 4px;
            background: var(--tg-s1);
          }}
          .tg-meter-value {{
            font-family: {FONT_STACK};
            font-size: 13px;
            font-variant-numeric: tabular-nums;
            text-align: right;
            color: var(--tg-ink);
          }}

          .tg-chip {{
            display: inline-block;
            font-family: {FONT_STACK};
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid var(--tg-border);
            color: var(--tg-ink-2);
            margin-right: 6px;
          }}

          @media (max-width: 640px) {{
            .tg-meter-row {{ grid-template-columns: 1fr 64px; }}
            .tg-meter-track {{ grid-column: 1 / -1; }}
            .tg-hero {{ font-size: 40px; }}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
