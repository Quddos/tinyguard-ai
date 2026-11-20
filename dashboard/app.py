"""
TinyGuard-AI visualization dashboard.

Run from the repository root:

    streamlit run dashboard/app.py

Everything shown comes from committed repository artifacts — the synthetic
dataset, the metrics JSON written by ml/run_evaluation.py, and the C headers the
firmware compiles. The dashboard computes no metric of its own that
run_evaluation.py already computes, and hardcodes none.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_DIR = DASHBOARD_DIR.parent

# View modules are executed as scripts by st.navigation and import `lib`.
for path in (DASHBOARD_DIR, REPO_DIR / "ml"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

st.set_page_config(
    page_title="TinyGuard-AI",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

from lib.theme import active_palette, inject_css  # noqa: E402

inject_css(active_palette())

VIEWS = DASHBOARD_DIR / "views"

pages = [
    st.Page(VIEWS / "overview.py", title="Overview", icon=":material/dashboard:",
            default=True),
    st.Page(VIEWS / "dataset.py", title="Dataset", icon=":material/waves:"),
    st.Page(VIEWS / "model.py", title="Model", icon=":material/network_node:"),
    st.Page(VIEWS / "quantization.py", title="Quantization",
            icon=":material/compress:"),
    st.Page(VIEWS / "firmware.py", title="Firmware", icon=":material/memory:"),
    st.Page(VIEWS / "provenance.py", title="Provenance",
            icon=":material/fingerprint:"),
]

navigation = st.navigation(pages)

with st.sidebar:
    st.markdown("### TinyGuard-AI")
    st.caption(
        "ARM Cortex-M4 TinyML predictive maintenance — INT8 quantization study."
    )
    st.divider()
    st.caption(
        "**Synthetic data.** No physical accelerometer, motor or machine was "
        "measured.\n\n"
        "**QEMU emulation.** No physical microcontroller was programmed or "
        "timed.\n\n"
        "No claim of physical sensor or hardware validation is made anywhere in "
        "this dashboard."
    )

navigation.run()
