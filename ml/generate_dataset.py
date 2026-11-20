"""
TinyGuard-AI synthetic vibration dataset generator.

SCIENTIFIC LIMITATION
---------------------
This dataset is entirely SYNTHETIC. No physical accelerometer, machine or
sensor was involved at any point. Signals are produced by the analytic models
below. Any result derived from this dataset describes behaviour on synthetic
data only and must never be presented as physical sensor validation.

The two classes are drawn from near-disjoint amplitude regimes, so the
classification task is separable almost by construction. See README.md
("Interpreting the accuracy figures") before quoting any accuracy number.

Determinism: a single seeded Generator (seed 42) supplies every random draw,
and draws happen in a fixed order, so this script is byte-reproducible.
"""

import numpy as np
import pandas as pd

from pathlib import Path

# Reproducible experiment
rng = np.random.default_rng(42)

SAMPLES_PER_WINDOW = 16
WINDOWS_PER_CLASS = 1500

ABNORMAL_SCENARIOS = [
    "high_amplitude",
    "impulsive",
    "mixed_frequency",
]


def generate_healthy():
    """Low-amplitude sinusoid plus mild Gaussian noise."""
    amplitude = rng.uniform(5, 18)
    frequency = rng.uniform(0.5, 2.5)
    phase = rng.uniform(0, 2 * np.pi)

    t = np.arange(SAMPLES_PER_WINDOW)

    signal = (
        amplitude * np.sin(2 * np.pi * frequency * t / 16 + phase)
        + rng.normal(0, 2.5, SAMPLES_PER_WINDOW)
    )

    return np.rint(signal).astype(np.int32)


def generate_abnormal(scenario):
    """
    Fault signal for one of ABNORMAL_SCENARIOS.

    The scenario is chosen by the caller so that the random draw which selects
    it stays in the main loop. This keeps the draw order identical to the
    original implementation and therefore keeps the dataset byte-reproducible.
    """
    t = np.arange(SAMPLES_PER_WINDOW)

    if scenario == "high_amplitude":

        amplitude = rng.uniform(30, 120)
        frequency = rng.uniform(1, 4)

        signal = (
            amplitude * np.sin(2 * np.pi * frequency * t / 16)
            + rng.normal(0, 8, SAMPLES_PER_WINDOW)
        )

    elif scenario == "impulsive":

        signal = rng.normal(0, 8, SAMPLES_PER_WINDOW)

        positions = rng.choice(
            SAMPLES_PER_WINDOW,
            size=rng.integers(1, 4),
            replace=False
        )

        signal[positions] += rng.choice(
            [-1, 1],
            size=len(positions)
        ) * rng.uniform(50, 130, len(positions))

    elif scenario == "mixed_frequency":

        signal = (
            25 * np.sin(2 * np.pi * t / 16)
            + 20 * np.sin(2 * np.pi * 4 * t / 16)
            + rng.normal(0, 8, SAMPLES_PER_WINDOW)
        )

    else:
        raise ValueError(f"Unknown abnormal scenario: {scenario!r}")

    return np.rint(signal).astype(np.int32)


def extract_features(samples):
    """
    Integer feature extraction.

    Deliberately mirrors firmware/signal_features.c exactly: truncating
    integer division and integer square root, so the Python features and the
    on-target C features are identical for the same window.
    """
    absolute = np.abs(samples.astype(np.int64))

    mean_absolute = int(np.sum(absolute) // len(samples))

    mean_square = int(
        np.sum(samples.astype(np.int64) ** 2) // len(samples)
    )

    rms = int(np.sqrt(mean_square))

    peak = int(np.max(absolute))

    return mean_absolute, rms, peak


rows = []

for label in [0, 1]:

    for i in range(WINDOWS_PER_CLASS):

        if label == 0:
            samples = generate_healthy()
            scenario = "healthy"
        else:
            scenario = rng.choice(ABNORMAL_SCENARIOS)
            samples = generate_abnormal(scenario)

        mean_abs, rms, peak = extract_features(samples)

        row = {
            "mean_absolute": mean_abs,
            "rms": rms,
            "peak": peak,
            "label": label,
            "scenario": scenario
        }

        # Preserve the original sensor measurements
        for j, value in enumerate(samples):
            row[f"sample_{j}"] = int(value)

        rows.append(row)


df = pd.DataFrame(rows)

# Shuffle the dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

output = Path(__file__).resolve().parent / "data"
output.mkdir(exist_ok=True)

df.to_csv(
    output / "tinyguard_vibration_dataset.csv",
    index=False
)

print("\nTinyGuard-AI Dataset Generation Complete")
print("-----------------------------------------")
print("NOTE: synthetic data only - no physical sensor was used.")
print("Total windows:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())

print("\nScenario distribution:")
print(df["scenario"].value_counts())

print("\nFeature preview:")
print(df[[
    "mean_absolute",
    "rms",
    "peak",
    "label"
]].head(10))

print("\nSaved to:")
print(output / "tinyguard_vibration_dataset.csv")
