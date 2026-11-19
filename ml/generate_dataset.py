
import numpy as np
import pandas as pd

from pathlib import Path

# Reproducible experiment
rng = np.random.default_rng(42)

SAMPLES_PER_WINDOW = 16
WINDOWS_PER_CLASS = 1500

def generate_healthy():
    amplitude = rng.uniform(5, 18)
    frequency = rng.uniform(0.5, 2.5)
    phase = rng.uniform(0, 2 * np.pi)

    t = np.arange(SAMPLES_PER_WINDOW)

    signal = (
        amplitude * np.sin(2 * np.pi * frequency * t / 16 + phase)
        + rng.normal(0, 2.5, SAMPLES_PER_WINDOW)
    )

    return np.rint(signal).astype(np.int32)


def generate_abnormal():
    t = np.arange(SAMPLES_PER_WINDOW)

    scenario = rng.choice([
        "high_amplitude",
        "impulsive",
        "mixed_frequency"
    ])

    if scenario == "high_amplitude":

        amplitude = rng.uniform(30, 120)
        frequency = rng.uniform(1, 4)

        signal = (
            amplitude * np.sin(2 * np.pi * frequency * t / 16)
            + rng.normal(0, 8, SAMPLES_PER_WINDOW)
        )

    elif scenario == "impulsive":

        signal = (
            rng.normal(0, 8, SAMPLES_PER_WINDOW)
        )

        number_of_impulses = rng.integers(1, 4)

        positions = rng.choice(
            SAMPLES_PER_WINDOW,
            size=number_of_impulses,
            replace=False
        )

        signal[positions] += rng.choice(
            [-1, 1],
            size=number_of_impulses
        ) * rng.uniform(50, 130, number_of_impulses)

    else:

        signal = (
            25 * np.sin(2 * np.pi * t / 16)
            + 20 * np.sin(2 * np.pi * 4 * t / 16)
            + rng.normal(0, 8, SAMPLES_PER_WINDOW)
        )

    return np.rint(signal).astype(np.int32)


def extract_features(samples):

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
            scenario = rng.choice([
                "high_amplitude",
                "impulsive",
                "mixed_frequency"
            ])

            # Generate according to selected scenario
            if scenario == "high_amplitude":
                t = np.arange(SAMPLES_PER_WINDOW)
                amplitude = rng.uniform(30, 120)
                frequency = rng.uniform(1, 4)

                signal = (
                    amplitude * np.sin(
                        2 * np.pi * frequency * t / 16
                    )
                    + rng.normal(0, 8, SAMPLES_PER_WINDOW)
                )

            elif scenario == "impulsive":
                signal = rng.normal(
                    0, 8, SAMPLES_PER_WINDOW
                )

                positions = rng.choice(
                    SAMPLES_PER_WINDOW,
                    size=rng.integers(1, 4),
                    replace=False
                )

                signal[positions] += rng.choice(
                    [-1, 1], size=len(positions)
                ) * rng.uniform(
                    50, 130, len(positions)
                )

            else:
                t = np.arange(SAMPLES_PER_WINDOW)

                signal = (
                    25 * np.sin(2 * np.pi * t / 16)
                    + 20 * np.sin(2 * np.pi * 4 * t / 16)
                    + rng.normal(0, 8, SAMPLES_PER_WINDOW)
                )

            samples = np.rint(signal).astype(np.int32)

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

output = Path("data")
output.mkdir(exist_ok=True)

df.to_csv(
    output / "tinyguard_vibration_dataset.csv",
    index=False
)

print("\nTinyGuard-AI Dataset Generation Complete")
print("-----------------------------------------")
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
print("data/tinyguard_vibration_dataset.csv")
