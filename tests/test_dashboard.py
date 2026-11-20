"""
Dashboard tests.

Streamlit pages are scripts, so they are not imported here. What is tested is
everything that can go wrong without anyone noticing:

  * the inference layer the dashboard displays agrees with the firmware's own
    numbers in results/quantization.json,
  * the dashboard reads the committed C headers and never the .pkl, so it
    cannot display a model other than the deployed one,
  * no metric is hardcoded into a page,
  * the scientific limitations are rendered on every page,
  * the chart palette is the validated one, in both light and dark.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_DIR / "dashboard"
VIEWS_DIR = DASHBOARD_DIR / "views"

VIEW_FILES = [
    "overview.py",
    "dataset.py",
    "model.py",
    "quantization.py",
    "firmware.py",
    "provenance.py",
]

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


# ------------------------------------------------------------------
# structure
# ------------------------------------------------------------------

def test_dashboard_layout_is_present():
    assert (DASHBOARD_DIR / "app.py").exists()
    assert (DASHBOARD_DIR / "lib" / "__init__.py").exists()
    for name in VIEW_FILES:
        assert (VIEWS_DIR / name).exists(), f"missing view: {name}"


def test_every_view_is_registered_in_the_app():
    app = (DASHBOARD_DIR / "app.py").read_text(encoding="utf-8")
    for name in VIEW_FILES:
        assert name in app, f"{name} is never registered with st.navigation"


def test_all_dashboard_modules_parse():
    for path in DASHBOARD_DIR.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ------------------------------------------------------------------
# the dashboard must show the deployed model, not the pickle
# ------------------------------------------------------------------

def test_dashboard_never_loads_the_pickle():
    """
    The dashboard reads firmware/model_*.h so what it displays is provably the
    model the firmware compiles. Loading the .pkl would reintroduce the chance
    of showing something else.
    """
    for path in DASHBOARD_DIR.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue

            assert "joblib" not in names, (
                f"{path.name} imports joblib; the dashboard must read the C "
                "headers, not the pickle"
            )

        assert "joblib.load" not in source, f"{path.name} loads a pickle"

    # Naming joblib's version on the provenance page is fine — that is read
    # from provenance.json, not from the library.

    data_source = (DASHBOARD_DIR / "lib" / "data.py").read_text(encoding="utf-8")
    assert "load_int8_header" in data_source
    assert "load_fp32_header" in data_source


def test_no_hardcoded_metrics_in_views():
    """
    Every figure must come from results/*.json or live inference. Catch a
    pasted-in accuracy or logit value.
    """
    suspicious = [
        r"accuracy\s*=\s*[01]\.\d",
        r"=\s*100\.0\s*#",
        r'"\s*100\.00\s*%\s*"',
        r"logit\s*=\s*[-+]?\d+\.\d{2,}",
    ]
    for name in VIEW_FILES:
        source = (VIEWS_DIR / name).read_text(encoding="utf-8")
        for pattern in suspicious:
            assert not re.search(pattern, source), (
                f"{name} appears to hardcode a metric matching {pattern!r}"
            )


def test_every_view_renders_the_limitations():
    for name in VIEW_FILES:
        source = (VIEWS_DIR / name).read_text(encoding="utf-8")
        assert "limitation_banner" in source, (
            f"{name} does not render the scientific limitations banner"
        )


def test_limitations_banner_reads_from_the_artifact():
    """
    The banner text must come from provenance.json, not be retyped in the
    dashboard where it could drift.
    """
    source = (DASHBOARD_DIR / "lib" / "components.py").read_text(encoding="utf-8")
    assert "def limitation_banner(limitations" in source

    for name in VIEW_FILES:
        view = (VIEWS_DIR / name).read_text(encoding="utf-8")
        assert 'limitation_banner(prov["limitations"])' in view, (
            f"{name} should pass provenance limitations into the banner"
        )


def test_dashboard_makes_no_hardware_or_latency_claim():
    forbidden = [
        r"\bon real hardware\b",
        r"\bphysical (?:board|device|sensor) (?:test|measurement)\b",
        r"\d+\s*(?:ms|µs|us)\s+per\s+inference\b",
        r"\bmeasured latency\b",
    ]
    for path in DASHBOARD_DIR.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not re.search(pattern, source, flags=re.I), (
                f"{path.name} contains a hardware/latency claim: {pattern!r}"
            )


# ------------------------------------------------------------------
# inference layer correctness
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def dash_inference():
    from lib import inference

    return inference


def test_dashboard_features_match_the_c_extractor(dash_inference, dataset):
    """The dashboard's feature extraction must equal the committed dataset."""
    sample_cols = [f"sample_{i}" for i in range(16)]
    sub = dataset.head(400)

    for _, row in sub.iterrows():
        window = [int(row[c]) for c in sample_cols]
        got = dash_inference.extract_features(window)
        expected = (
            int(row["mean_absolute"]), int(row["rms"]), int(row["peak"])
        )
        assert got == expected, f"feature mismatch on window {window}"


def test_dashboard_int8_matches_published_demo_results(
    dash_inference, int8_header, fp32_header, results_dir
):
    """
    The dashboard's live inference must reproduce the firmware demo figures in
    results/quantization.json exactly. If these drift apart, the dashboard is
    showing something the metrics do not.
    """
    import json

    quant = json.loads(
        (results_dir / "quantization.json").read_text(encoding="utf-8")
    )
    demo = quant["firmware_demo_windows"]

    for name, block in demo.items():
        feats = np.array(
            [[
                block["features"]["mean_absolute"],
                block["features"]["rms"],
                block["features"]["peak"],
            ]],
            dtype=np.float64,
        )

        out = dash_inference.int8_forward(feats, int8_header)
        fp32 = float(
            dash_inference.fp32_forward(feats, fp32_header).ravel()[0]
        )

        assert out["q_input"][0].tolist() == block["int8_quantized_input"], (
            f"{name}: dashboard INT8 input differs from quantization.json"
        )
        assert int(out["prediction"][0]) == block["int8_prediction"], name
        assert bool(out["saturated"][0]) == block["input_saturated"], name

        np.testing.assert_allclose(
            float(out["logit"].ravel()[0]), block["int8_logit"],
            rtol=0, atol=1e-4, err_msg=f"{name}: INT8 logit drifted",
        )
        np.testing.assert_allclose(
            fp32, block["fp32_logit"],
            rtol=0, atol=1e-3, err_msg=f"{name}: FP32 logit drifted",
        )


def test_dashboard_int8_agrees_with_fp32_on_whole_dataset(
    dash_inference, dataset, int8_header, fp32_header
):
    """The dashboard must reproduce the project's headline claim itself."""
    X = dataset[["mean_absolute", "rms", "peak"]].to_numpy(dtype=np.float64)

    fp32_logits = dash_inference.fp32_forward(X, fp32_header).ravel()
    out = dash_inference.int8_forward(X, int8_header)

    fp32_pred = (fp32_logits >= 0).astype(np.int64)
    int8_pred = out["prediction"]

    assert int(np.sum(fp32_pred != int8_pred)) == 0


def test_representable_limit_matches_published_value(
    dash_inference, int8_header, results_dir
):
    import json

    quant = json.loads(
        (results_dir / "quantization.json").read_text(encoding="utf-8")
    )
    published = quant["representable_normalized_input_range"]["max_abs"]

    assert dash_inference.representable_input_limit(int8_header) == pytest.approx(
        published, rel=0, abs=1e-9
    )


# ------------------------------------------------------------------
# palette
# ------------------------------------------------------------------

def test_palette_matches_the_validated_values():
    """
    These exact hexes were checked with the data-visualisation validator
    against the surfaces the charts render on. Changing one without re-running
    the validator would silently break colour-vision separation.
    """
    from lib.theme import DARK, LIGHT

    assert LIGHT.series == ("#2a78d6", "#eb6834", "#1baf7a")
    assert DARK.series == ("#3987e5", "#d95926", "#199e70")

    assert LIGHT.surface == "#ffffff"
    assert DARK.surface == "#0e1117"


def test_palette_caps_at_three_categorical_slots():
    """
    Three slots is the all-pairs colour-vision cap. A fourth would need
    re-validation and would put yellow next to orange.
    """
    from lib.theme import DARK, LIGHT

    assert len(LIGHT.series) == 3
    assert len(DARK.series) == 3


def test_status_colours_are_not_reused_as_series():
    from lib.theme import DARK, LIGHT

    for palette in (LIGHT, DARK):
        status = {palette.good, palette.warning, palette.critical}
        assert status.isdisjoint(set(palette.series)), (
            "a status colour is being reused as a categorical series"
        )


def test_dark_mode_is_distinct_from_light():
    """Dark steps are selected for the dark surface, not an inverted copy."""
    from lib.theme import DARK, LIGHT

    assert LIGHT.series != DARK.series
    assert LIGHT.grid != DARK.grid


def test_sequential_ramp_is_single_hue_and_monotonic():
    """
    A magnitude ramp must be one hue, light to dark. A rainbow would encode
    order as hue, which is unreadable under colour-vision deficiency.
    """
    from lib.theme import LIGHT

    def luminance(hex_colour: str) -> float:
        r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lums = [luminance(c) for c in LIGHT.sequential]
    assert len(lums) >= 5
    assert all(a > b for a, b in zip(lums, lums[1:])), (
        "sequential ramp is not monotonically darkening"
    )


def test_charts_ship_a_table_view():
    """
    No value may be reachable only by hovering. The chart helper pairs every
    figure with a table.
    """
    components = (DASHBOARD_DIR / "lib" / "components.py").read_text(encoding="utf-8")
    assert "Table view" in components

    for name in ("dataset.py", "quantization.py", "firmware.py", "overview.py"):
        source = (VIEWS_DIR / name).read_text(encoding="utf-8")
        assert "C.chart(" in source, f"{name} should render charts via C.chart"


def test_streamlit_config_does_not_pin_a_theme():
    """
    Pinning a theme would force one palette and make the other unreachable, so
    one of the two validated palettes would never render.
    """
    config = REPO_DIR / ".streamlit" / "config.toml"
    assert config.exists()

    text = config.read_text(encoding="utf-8")
    active = [
        line for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    joined = "\n".join(active)

    assert "base =" not in joined, "config.toml pins a theme base"
    assert "primaryColor" not in joined, "config.toml pins theme colours"
