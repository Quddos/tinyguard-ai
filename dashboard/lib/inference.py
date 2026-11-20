"""
FP32 and INT8 inference driven entirely by the committed C headers.

Both functions mirror the firmware sources exactly:

  fp32_forward  <-> firmware/tinyguard_model.c
  int8_forward  <-> firmware/tinyguard_int8.c

In particular int8_forward reproduces C's float32 arithmetic and its
round-away-from-zero cast, `(int32_t)(x + 0.5f)` / `(int32_t)(x - 0.5f)`, rather
than numpy's banker's rounding. That is what makes the dashboard's numbers equal
the on-target numbers instead of merely resembling them.

Nothing here reads the .pkl, so the dashboard cannot display a model other than
the one the firmware compiles.
"""

from __future__ import annotations

import math

import numpy as np

# Feature extraction mirrors firmware/signal_features.c: truncating integer
# division and an integer square root, accumulated in 64-bit.


def extract_features(window) -> tuple[int, int, int]:
    w = np.asarray(window, dtype=np.int64)
    a = np.abs(w)

    mean_absolute = int(a.sum() // w.size)
    mean_square = int((w ** 2).sum() // w.size)
    rms = math.isqrt(mean_square)
    peak = int(a.max())

    return mean_absolute, rms, peak


def fp32_forward(features, header) -> np.ndarray:
    """
    Float reference path, float32 throughout, as tinyguard_model.c computes it.

    `features` is (n, 3) raw [mean_absolute, rms, peak].
    """
    raw = np.atleast_2d(np.asarray(features, dtype=np.float32))

    z = (raw - header["scaler_mean"]) / header["scaler_scale"]

    hidden = np.maximum(
        z @ header["weights1"] + header["bias1"], np.float32(0)
    ).astype(np.float32)

    logits = (hidden @ header["weights2"] + header["bias2"]).astype(np.float32)

    return logits


def _round_away_from_zero_f32(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return np.where(
        v >= 0,
        np.floor(v + np.float32(0.5)),
        np.ceil(v - np.float32(0.5)),
    ).astype(np.int32)


def int8_forward(features, header) -> dict:
    """
    Quantized integer path, as tinyguard_int8.c computes it.

    Returns every intermediate the dashboard visualises:
      normalized, q_input, acc1, q_hidden, acc2, logit, prediction, saturated.
    """
    raw = np.atleast_2d(np.asarray(features, dtype=np.float32))

    z = ((raw - header["scaler_mean"]) / header["scaler_std"]).astype(np.float32)

    q_input = np.clip(
        _round_away_from_zero_f32(z / header["input_scale"]), -127, 127
    ).astype(np.int32)

    acc1 = np.maximum(
        q_input @ header["weights1"].astype(np.int32) + header["bias1"], 0
    ).astype(np.int32)

    hidden_real = (acc1.astype(np.float32) * header["acc1_scale"]).astype(np.float32)

    q_hidden = np.clip(
        _round_away_from_zero_f32(hidden_real / header["hidden_scale"]), 0, 127
    ).astype(np.int32)

    acc2 = (
        q_hidden @ header["weights2"].astype(np.int32) + int(header["bias2"][0])
    ).astype(np.int32)

    logit = (acc2.astype(np.float32) * header["acc2_scale"]).astype(np.float32)

    return {
        "normalized": z,
        "q_input": q_input,
        "acc1": acc1,
        "q_hidden": q_hidden,
        "acc2": acc2,
        "logit": logit,
        "prediction": (acc2 >= 0).astype(np.int64),
        "saturated": np.any(np.abs(q_input) >= 127, axis=1),
    }


def representable_input_limit(header) -> float:
    """
    Largest normalized feature magnitude the INT8 input quantizer can encode.

    Beyond this, inputs clip at +/-127 and the logit is compressed. The firmware
    keeps one deliberately severe window past this limit to demonstrate it.
    """
    return float(127 * header["input_scale"])
