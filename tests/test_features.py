"""
Feature-extraction parity tests.

The Python generator and firmware/signal_features.c must compute identical
integer features for the same window, because the model is trained on the
Python features and deployed against the C features. A divergence here would
silently shift the model's input distribution on target.

These tests reimplement the C integer arithmetic in Python and check it against
the committed dataset, so they run without a C toolchain. The C source itself is
checked for structural agreement in test_firmware_sync.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

SAMPLE_COLUMNS = [f"sample_{i}" for i in range(16)]


def c_features(window):
    """
    Mirror of calculate_features() in firmware/signal_features.c.

    Uses truncating integer division and an integer square root, matching the
    C implementation rather than float means.
    """
    w = np.asarray(window, dtype=np.int64)
    a = np.abs(w)

    mean_absolute = int(a.sum() // w.size)
    mean_square = int((w ** 2).sum() // w.size)
    rms = int(math.isqrt(mean_square))
    peak = int(a.max())

    return mean_absolute, rms, peak


def test_dataset_features_match_c_integer_arithmetic(dataset):
    """Every committed row's features must be reproducible from its samples."""
    windows = dataset[SAMPLE_COLUMNS].to_numpy(dtype=np.int64)
    expected = dataset[["mean_absolute", "rms", "peak"]].to_numpy(dtype=np.int64)

    mismatches = []
    for i, window in enumerate(windows):
        got = c_features(window)
        if tuple(expected[i]) != got:
            mismatches.append((i, tuple(expected[i]), got))

    assert not mismatches, (
        f"{len(mismatches)} of {len(windows)} rows disagree with the C integer "
        f"feature maths. First few: {mismatches[:5]}"
    )


def test_integer_sqrt_matches_c_algorithm():
    """
    The C integer_sqrt is a restoring bit-by-bit square root. It must equal
    math.isqrt (floor of the true square root) across the input range.
    """

    def integer_sqrt(number):
        result = 0
        bit = 1 << 30
        while bit > number:
            bit >>= 2
        while bit != 0:
            if number >= result + bit:
                number -= result + bit
                result = (result >> 1) + bit
            else:
                result >>= 1
            bit >>= 2
        return result

    probes = list(range(0, 2048))
    probes += [10 ** k for k in range(3, 10)]
    probes += [2 ** k - 1 for k in range(2, 32)]
    probes += [2 ** k for k in range(2, 32)]
    probes += [0xFFFFFFFF, 0xFFFFFFFE, 577_000, 36_100]

    for n in probes:
        if n > 0xFFFFFFFF:
            continue
        assert integer_sqrt(n) == math.isqrt(n), f"integer_sqrt({n}) diverged"


def test_features_are_nonnegative_and_ordered(dataset):
    """
    mean_absolute <= rms <= peak holds for any real signal (power mean
    inequality), allowing for integer truncation.
    """
    ma = dataset["mean_absolute"].to_numpy(dtype=np.int64)
    rms = dataset["rms"].to_numpy(dtype=np.int64)
    peak = dataset["peak"].to_numpy(dtype=np.int64)

    assert (ma >= 0).all()
    assert (rms >= 0).all()
    assert (peak >= 0).all()

    # Truncating division can pull rms one count below mean_absolute.
    assert (rms >= ma - 1).all(), "rms fell below mean_absolute by more than truncation allows"
    assert (peak >= rms).all(), "peak must dominate rms"


def test_accumulator_width_handles_large_samples():
    """
    Regression test for the uint32 -> uint64 accumulator fix in
    signal_features.c.

    With 32-bit accumulators, a window of large samples overflows and yields a
    wrong RMS. The reference implementation must stay exact here.
    """
    window = [100_000] * 16
    mean_absolute, rms, peak = c_features(window)

    assert mean_absolute == 100_000
    assert peak == 100_000
    assert rms == 100_000

    # sum of squares for this window exceeds the 32-bit range, which is the
    # condition that used to overflow.
    assert sum(x * x for x in window) > 0xFFFFFFFF


@pytest.mark.parametrize(
    "window,expected",
    [
        # The three windows firmware/sensor_sim.c actually feeds the model.
        ([12, -10, 15, -13, 8, -14, 11, -9, 16, -12, 10, -11, 14, -8, 9, -13],
         (11, 11, 16)),
        ([-12, 39, 39, -11, -57, -14, 29, 46, -11, -47, -39, 13, 74, 10, -25, -28],
         (30, 35, 74)),
        ([120, -110, 145, -130, 180, -160, 135, -190,
          170, -140, 155, -175, 190, -120, 165, -185],
         (154, 156, 190)),
    ],
)
def test_firmware_demo_window_features(window, expected):
    assert c_features(window) == expected
