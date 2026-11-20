"""
Parser for the generated INT8 C header.

Why this exists (defect fix)
----------------------------
verify_int8_c.py and verify_embedded_model.py previously RE-DERIVED the
quantization constants from the .pkl instead of reading the header that is
actually compiled into the firmware. That means the thing being verified was
not necessarily the thing being shipped: any drift between the exporter and the
committed header would pass verification silently.

These helpers read the committed header so callers can assert that their
re-derived constants match the shipped ones exactly.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

FIRMWARE_DIR = Path(__file__).resolve().parent.parent / "firmware"
INT8_HEADER = FIRMWARE_DIR / "model_int8_params.h"
FP32_HEADER = FIRMWARE_DIR / "model_params.h"


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _defines(text: str) -> dict[str, float]:
    out = {}
    for name, value in re.findall(
        r"#define\s+(\w+)\s+([-+0-9.eE]+)f?\s*$", text, flags=re.M
    ):
        out[name] = float(value)
    return out


def _numbers(block: str) -> list[float]:
    return [
        float(tok)
        for tok in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", block)
    ]


def _array(text: str, declaration: str) -> list[float]:
    """Extract the brace-enclosed initialiser following a declaration."""
    start = text.index(declaration)
    open_brace = text.index("{", start)

    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return _numbers(text[open_brace : i + 1])

    raise ValueError(f"unterminated initialiser for {declaration!r}")


def load_int8_header(path: Path | None = None) -> dict:
    """
    Read the committed INT8 header into plain numpy arrays.

    Returns the quantization scales, the StandardScaler parameters as they
    appear in the header (float32 precision), and the quantized weights and
    biases, shaped as the C code declares them.
    """
    path = Path(path) if path is not None else INT8_HEADER
    text = _strip_comments(path.read_text())

    defines = _defines(text)

    inputs = int(defines["MODEL_INPUTS"])
    hidden = int(defines["MODEL_HIDDEN"])

    w1 = np.array(_array(text, "weights1"), dtype=np.int8)
    if w1.size != inputs * hidden:
        raise ValueError(f"weights1 has {w1.size} entries, expected {inputs * hidden}")

    return {
        "path": path,
        "inputs": inputs,
        "hidden": hidden,
        "input_scale": np.float32(defines["INPUT_SCALE"]),
        "w1_scale": np.float32(defines["W1_SCALE"]),
        "acc1_scale": np.float32(defines["ACC1_SCALE"]),
        "hidden_scale": np.float32(defines["HIDDEN_SCALE"]),
        "w2_scale": np.float32(defines["W2_SCALE"]),
        "acc2_scale": np.float32(defines["ACC2_SCALE"]),
        "scaler_mean": np.array(_array(text, "scaler_mean"), dtype=np.float32),
        "scaler_std": np.array(_array(text, "scaler_std"), dtype=np.float32),
        "weights1": w1.reshape(inputs, hidden),
        "bias1": np.array(_array(text, "bias1"), dtype=np.int32),
        "weights2": np.array(_array(text, "weights2"), dtype=np.int8),
        "bias2": np.array(_array(text, "bias2"), dtype=np.int32),
    }


def float32_ulp_distance(a, b) -> int:
    """
    Distance in float32 units-in-the-last-place between two values.

    0 means bit-identical. 1 means adjacent representable float32 values.
    """
    fa = np.float32(a).view(np.int32)
    fb = np.float32(b).view(np.int32)
    return int(abs(int(fa) - int(fb)))


def scale_ulp_report(
    *, input_scale, acc1_scale, hidden_scale, acc2_scale, header=None
) -> dict:
    """
    ULP distance between freshly derived scales and the committed header.

    WHY THIS IS NOT AN EQUALITY CHECK
    ---------------------------------
    The exporter computes each scale in float64 and writes it as decimal text
    with 12 significant digits. The C compiler then parses that text into a
    float32. Casting the original float64 straight to float32 does not always
    land on the same float32 as parsing the truncated decimal text, so the two
    can differ by 1 ULP. That gap is a property of the text representation, not
    exporter drift.

    It is provably harmless for this model: the INT8 quantized inputs, the
    output accumulators and therefore every prediction are bit-identical under
    both scale sets across all 3000 windows (verified in
    tests/test_quantization.py), and the smallest decision margin in the
    dataset is 382 accumulator counts.

    The integer tensors, which alone determine the class decision, are still
    required to match EXACTLY.
    """
    h = header if header is not None else load_int8_header()
    return {
        "INPUT_SCALE": float32_ulp_distance(input_scale, h["input_scale"]),
        "ACC1_SCALE": float32_ulp_distance(acc1_scale, h["acc1_scale"]),
        "HIDDEN_SCALE": float32_ulp_distance(hidden_scale, h["hidden_scale"]),
        "ACC2_SCALE": float32_ulp_distance(acc2_scale, h["acc2_scale"]),
    }


def assert_matches_header(
    *,
    weights1,
    bias1,
    weights2,
    bias2,
    input_scale,
    acc1_scale,
    hidden_scale,
    acc2_scale,
    header: dict | None = None,
    max_scale_ulp: int = 1,
) -> dict:
    """
    Assert that freshly derived quantization parameters match the committed
    header.

    Integer tensors (weights and biases) must match EXACTLY - they determine
    every class decision. Float scales must match to within `max_scale_ulp`
    float32 ULPs; see scale_ulp_report for why exact equality is not the right
    criterion here.

    Raises AssertionError naming the first field that drifted.
    """
    h = header if header is not None else load_int8_header()

    def same_int(name, got, want):
        got = np.asarray(got).reshape(np.asarray(want).shape)
        if not np.array_equal(got.astype(np.int64), np.asarray(want).astype(np.int64)):
            raise AssertionError(
                f"{name} drifted from {h['path'].name}:\n"
                f"  derived  = {got.ravel().tolist()}\n"
                f"  in header= {np.asarray(want).ravel().tolist()}"
            )

    def same_scale(name, got, want):
        ulp = float32_ulp_distance(got, want)
        if ulp > max_scale_ulp:
            raise AssertionError(
                f"{name} drifted from {h['path'].name}: "
                f"derived={np.float32(got)!r} header={np.float32(want)!r} "
                f"({ulp} float32 ULPs apart, tolerance {max_scale_ulp})"
            )

    same_int("weights1", weights1, h["weights1"])
    same_int("bias1", bias1, h["bias1"])
    same_int("weights2", weights2, h["weights2"])
    same_int("bias2", bias2, h["bias2"])

    same_scale("INPUT_SCALE", input_scale, h["input_scale"])
    same_scale("ACC1_SCALE", acc1_scale, h["acc1_scale"])
    same_scale("HIDDEN_SCALE", hidden_scale, h["hidden_scale"])
    same_scale("ACC2_SCALE", acc2_scale, h["acc2_scale"])

    return h


def load_fp32_header(path: Path | None = None) -> dict:
    """Read the committed FP32 header (float weights, scalar output bias)."""
    path = Path(path) if path is not None else FP32_HEADER
    text = _strip_comments(path.read_text())

    defines = _defines(text)
    inputs = int(defines["MODEL_INPUTS"])
    hidden = int(defines["MODEL_HIDDEN"])

    bias2 = re.search(r"bias2\s*=\s*(-?[0-9.eE+-]+)f", text)
    if bias2 is None:
        raise ValueError("could not parse bias2 from FP32 header")

    return {
        "path": path,
        "inputs": inputs,
        "hidden": hidden,
        "scaler_mean": np.array(_array(text, "scaler_mean"), dtype=np.float32),
        "scaler_scale": np.array(_array(text, "scaler_scale"), dtype=np.float32),
        "weights1": np.array(_array(text, "weights1"), dtype=np.float32).reshape(
            inputs, hidden
        ),
        "bias1": np.array(_array(text, "bias1"), dtype=np.float32),
        "weights2": np.array(_array(text, "weights2"), dtype=np.float32),
        "bias2": np.float32(bias2.group(1)),
    }


if __name__ == "__main__":
    h = load_int8_header()
    print("INT8 header:", h["path"])
    print("  dimensions:", h["inputs"], "->", h["hidden"], "-> 1")
    print("  INPUT_SCALE:", h["input_scale"])
    print("  weights1:\n", h["weights1"])
    print("  bias1:", h["bias1"].tolist())
    print("  weights2:", h["weights2"].tolist())
    print("  bias2:", h["bias2"].tolist())

    f = load_fp32_header()
    print("\nFP32 header:", f["path"])
    print("  weights1 shape:", f["weights1"].shape)
    print("  bias2:", f["bias2"])
