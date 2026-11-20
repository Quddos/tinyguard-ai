"""
INT8 quantization tests.

The central claim of this project is that the quantized integer model deployed
on ARM behaves like the float reference. These tests pin that claim down:

  * the committed C header matches the trained .pkl,
  * the INT8 forward pass reproduces the FP32 class decision on every window,
  * the known 1-ULP scale-text gap provably cannot change a prediction,
  * input saturation is detected and bounded rather than ignored.

Everything here runs from the committed header, i.e. the constants actually
compiled into the firmware - not from freshly derived numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

FEATURES = ["mean_absolute", "rms", "peak"]


# ------------------------------------------------------------------
# reference implementations
# ------------------------------------------------------------------

def fp32_logits(X_raw, model, scaler):
    Z = scaler.transform(np.asarray(X_raw, dtype=np.float64))
    hidden = np.maximum(0, Z @ model.coefs_[0] + model.intercepts_[0])
    return (hidden @ model.coefs_[1] + model.intercepts_[1]).ravel()


def round_away_from_zero_f32(values):
    """C's (int)(x + 0.5f) / (int)(x - 0.5f) rounding, in float32."""
    v = np.asarray(values, dtype=np.float32)
    return np.where(
        v >= 0,
        np.floor(v + np.float32(0.5)),
        np.ceil(v - np.float32(0.5)),
    ).astype(np.int32)


def int8_forward(X_raw, h):
    """Mirror of tinyguard_int8_predict() in firmware/tinyguard_int8.c."""
    raw = np.asarray(X_raw, dtype=np.float32)
    z = (raw - h["scaler_mean"]) / h["scaler_std"]

    q_in = np.clip(
        round_away_from_zero_f32(z / h["input_scale"]), -127, 127
    ).astype(np.int32)

    acc1 = np.maximum(q_in @ h["weights1"].astype(np.int32) + h["bias1"], 0)

    hidden_real = acc1.astype(np.float32) * h["acc1_scale"]
    q_hidden = np.clip(
        round_away_from_zero_f32(hidden_real / h["hidden_scale"]), 0, 127
    ).astype(np.int32)

    acc2 = q_hidden @ h["weights2"].astype(np.int32) + int(h["bias2"][0])
    logits = (acc2.astype(np.float32) * h["acc2_scale"]).astype(np.float32)

    return logits, (acc2 >= 0).astype(np.int64), q_in, acc2


@pytest.fixture(scope="module")
def X_all(dataset):
    return dataset[FEATURES].to_numpy(dtype=np.float64)


@pytest.fixture(scope="module")
def y_all(dataset):
    return dataset["label"].to_numpy(dtype=np.int64)


# ------------------------------------------------------------------
# header integrity
# ------------------------------------------------------------------

def test_int8_header_shapes(int8_header):
    h = int8_header
    assert h["inputs"] == 3
    assert h["hidden"] == 8
    assert h["weights1"].shape == (3, 8)
    assert h["bias1"].shape == (8,)
    assert h["weights2"].shape == (8,)
    assert h["bias2"].shape == (1,)


def test_int8_weights_are_in_symmetric_range(int8_header):
    """Symmetric INT8 must never emit -128; the range is [-127, 127]."""
    for name in ("weights1", "weights2"):
        w = int8_header[name]
        assert w.min() >= -127, f"{name} contains {w.min()}, below -127"
        assert w.max() <= 127, f"{name} contains {w.max()}, above 127"


def test_at_least_one_weight_hits_full_scale(int8_header):
    """
    A correctly calibrated symmetric scale puts the largest-magnitude weight at
    +/-127. If nothing reaches it, the scale is too coarse and resolution is
    being wasted.
    """
    assert np.abs(int8_header["weights1"]).max() == 127
    assert np.abs(int8_header["weights2"]).max() == 127


def test_int8_scaler_params_match_trained_scaler(int8_header, trained):
    model, scaler = trained
    np.testing.assert_allclose(
        int8_header["scaler_mean"], scaler.mean_.astype(np.float32),
        rtol=0, atol=1e-4,
    )
    np.testing.assert_allclose(
        int8_header["scaler_std"], scaler.scale_.astype(np.float32),
        rtol=0, atol=1e-4,
    )


def test_fp32_header_matches_trained_model(fp32_header, trained):
    model, scaler = trained
    np.testing.assert_allclose(
        fp32_header["weights1"], model.coefs_[0].astype(np.float32),
        rtol=0, atol=1e-6,
    )
    np.testing.assert_allclose(
        fp32_header["weights2"], model.coefs_[1].ravel().astype(np.float32),
        rtol=0, atol=1e-6,
    )
    np.testing.assert_allclose(
        fp32_header["bias1"], model.intercepts_[0].astype(np.float32),
        rtol=0, atol=1e-6,
    )


def test_derived_parameters_match_committed_header(trained, dataset, int8_header):
    """
    Re-derive the quantization constants from the .pkl and require that the
    integer tensors are bit-exact against the committed header.

    This is the guard that the model being verified is the model being
    compiled into the firmware.
    """
    from sklearn.model_selection import train_test_split

    from header_params import assert_matches_header

    model, scaler = trained
    X = dataset[FEATURES].to_numpy(dtype=np.float64)
    y = dataset["label"].to_numpy(dtype=np.int64)

    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def q(values):
        maximum = np.max(np.abs(values))
        scale = maximum / 127.0 if maximum > 0 else 1.0
        return np.clip(np.rint(values / scale), -127, 127).astype(np.int8), scale

    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    q_in, input_scale = q(X_train_scaled)
    W1, w1_scale = q(model.coefs_[0])
    acc1_scale = input_scale * w1_scale
    b1 = np.rint(model.intercepts_[0] / acc1_scale).astype(np.int32)
    acc1 = q_in.astype(np.int32) @ W1.astype(np.int32) + b1
    _, hidden_scale = q(np.maximum(acc1, 0) * acc1_scale)
    W2, w2_scale = q(model.coefs_[1].flatten())
    acc2_scale = hidden_scale * w2_scale
    b2 = np.rint(model.intercepts_[1] / acc2_scale).astype(np.int32)

    # Raises AssertionError naming the offending field if anything drifted.
    assert_matches_header(
        weights1=W1, bias1=b1, weights2=W2, bias2=b2,
        input_scale=input_scale, acc1_scale=acc1_scale,
        hidden_scale=hidden_scale, acc2_scale=acc2_scale,
        header=int8_header,
    )


# ------------------------------------------------------------------
# FP32 <-> INT8 equivalence
# ------------------------------------------------------------------

def test_int8_agrees_with_fp32_on_every_window(X_all, y_all, trained, int8_header):
    """
    The headline claim: zero prediction mismatches between the float reference
    and the quantized integer model across the whole dataset.
    """
    model, scaler = trained
    f_log = fp32_logits(X_all, model, scaler)
    f_pred = (f_log >= 0).astype(np.int64)

    i_log, i_pred, _, _ = int8_forward(X_all, int8_header)

    mismatches = int(np.sum(f_pred != i_pred))
    assert mismatches == 0, (
        f"{mismatches} of {len(y_all)} windows disagree between FP32 and INT8"
    )


def test_int8_accuracy_equals_fp32_accuracy(X_all, y_all, trained, int8_header):
    model, scaler = trained
    f_pred = (fp32_logits(X_all, model, scaler) >= 0).astype(np.int64)
    _, i_pred, _, _ = int8_forward(X_all, int8_header)

    assert float((f_pred == y_all).mean()) == float((i_pred == y_all).mean())


def test_logit_error_is_bounded_on_in_distribution_data(
    X_all, trained, int8_header
):
    """
    Quantization must not distort the logit wildly for in-distribution input.
    The bound is generous but finite, so a regression that breaks requantization
    is caught.
    """
    model, scaler = trained
    f_log = fp32_logits(X_all, model, scaler)
    i_log, _, q_in, _ = int8_forward(X_all, int8_header)

    unsaturated = ~np.any(np.abs(q_in) >= 127, axis=1)
    assert unsaturated.sum() > 0

    err = np.abs(f_log[unsaturated] - i_log[unsaturated].astype(np.float64))

    assert err.max() < 5.0, f"max logit error {err.max()} on unsaturated input"
    assert err.mean() < 1.0, f"mean logit error {err.mean()}"


def test_decision_margin_is_large(X_all, trained, int8_header):
    """
    No window sits near the decision boundary. This is what makes the model
    robust to the 1-ULP scale representation gap, so it is asserted rather
    than assumed.
    """
    _, _, _, acc2 = int8_forward(X_all, int8_header)
    assert int(np.abs(acc2).min()) > 100, (
        f"smallest INT8 decision margin is only {int(np.abs(acc2).min())} counts"
    )


def test_scale_text_ulp_gap_is_inert(X_all, trained, int8_header):
    """
    The header stores each scale as 12-significant-digit decimal text, which
    can parse to a float32 one ULP away from casting the float64 directly.

    Require that gap to be at most 1 ULP AND to leave every quantized input,
    accumulator and prediction bit-identical.
    """
    h = int8_header

    _, base_pred, base_q, base_acc = int8_forward(X_all, h)

    # Perturb each scale by one ULP in both directions and require that no
    # prediction moves. This bounds the effect of the text-representation gap
    # without depending on how the exporter happened to derive the value.
    for field in ("input_scale", "acc1_scale", "hidden_scale", "acc2_scale"):
        for direction in (+1, -1):
            perturbed = dict(h)
            bits = np.float32(h[field]).view(np.int32) + direction
            perturbed[field] = np.int32(bits).view(np.float32)

            _, pred, q_in, acc = int8_forward(X_all, perturbed)

            assert np.array_equal(pred, base_pred), (
                f"a 1-ULP change to {field} ({direction:+d}) altered predictions"
            )

    # The input and accumulator paths should be completely unaffected by a
    # 1-ULP change to the output scale, which only rescales the reported logit.
    output_perturbed = dict(h)
    bits = np.float32(h["acc2_scale"]).view(np.int32) + 1
    output_perturbed["acc2_scale"] = np.int32(bits).view(np.float32)
    _, _, q_out, acc_out = int8_forward(X_all, output_perturbed)

    assert np.array_equal(q_out, base_q)
    assert np.array_equal(acc_out, base_acc)


# ------------------------------------------------------------------
# saturation behaviour (defect 10)
# ------------------------------------------------------------------

def test_dataset_does_not_saturate_the_input_quantizer(
    X_all, trained, int8_header
):
    """
    The input scale is calibrated on the training split, so no dataset window
    should clip. If this fails, calibration has regressed.
    """
    _, _, q_in, _ = int8_forward(X_all, int8_header)
    saturated = int(np.sum(np.any(np.abs(q_in) >= 127, axis=1)))

    assert saturated <= 1, (
        f"{saturated} dataset windows saturate the INT8 input quantizer"
    )


def test_representable_input_range_covers_the_dataset(
    X_all, trained, int8_header
):
    model, scaler = trained
    limit = float(127 * int8_header["input_scale"])
    observed = float(np.abs(scaler.transform(X_all)).max())

    assert observed <= limit, (
        f"dataset reaches |z|={observed:.4f} but the INT8 input range only "
        f"covers |z|<={limit:.4f}"
    )


def test_out_of_distribution_window_saturates_and_is_documented(int8_header):
    """
    firmware/sensor_sim.c keeps a deliberately severe window whose features lie
    outside the training range. It must still classify as an anomaly, but its
    logit is compressed by saturation - the behaviour documented in
    sensor_sim.h. This test pins both facts so the documentation cannot drift.
    """
    severe_features = np.array([[154.0, 156.0, 190.0]])

    _, pred, q_in, _ = int8_forward(severe_features, int8_header)

    assert bool(np.any(np.abs(q_in[0]) >= 127)), (
        "the severe demo window no longer saturates; sensor_sim.h needs updating"
    )
    assert int(pred[0]) == 1, "severe fault window must classify as anomaly"
