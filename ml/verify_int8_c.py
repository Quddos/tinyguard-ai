
import subprocess
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from pathlib import Path

from header_params import assert_matches_header

# ============================================================
# 1. LOAD MODEL, SCALER AND DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "tinyguard_mlp.pkl")
scaler = joblib.load(MODEL_DIR / "tinyguard_scaler.pkl")

df = pd.read_csv(
    BASE_DIR / "data" / "tinyguard_vibration_dataset.csv"
)

features = ["mean_absolute", "rms", "peak"]

X = df[features].values.astype(np.float32)
y = df["label"].values.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 2. RECREATE QUANTIZATION PARAMETERS
# ============================================================

def quantize(values):
    scale = np.max(np.abs(values)) / 127.0

    q_values = np.clip(
        np.rint(values / scale),
        -127,
        127
    ).astype(np.int8)

    return q_values, scale


W1, W1_scale = quantize(model.coefs_[0])

X_train_scaled = scaler.transform(X_train).astype(np.float32)

q_input_train, input_scale = quantize(X_train_scaled)

acc1_scale = input_scale * W1_scale

q_b1 = np.rint(
    model.intercepts_[0] / acc1_scale
).astype(np.int32)

acc1_train = (
    q_input_train.astype(np.int32)
    @ W1.astype(np.int32)
    + q_b1
)

hidden_train = np.maximum(
    acc1_train, 0
) * acc1_scale

_, hidden_scale = quantize(hidden_train)

# IMPORTANT: Flatten second-layer weights.
W2, W2_scale = quantize(
    model.coefs_[1].flatten()
)

acc2_scale = hidden_scale * W2_scale

q_b2 = np.rint(
    model.intercepts_[1] / acc2_scale
).astype(np.int32)

# ============================================================
# 2b. GUARD: the re-derived parameters MUST equal the shipped header
# ============================================================
#
# DEFECT FIX: this script used to re-derive the quantization constants from the
# .pkl and verify those, never reading firmware/model_int8_params.h. The model
# being verified was therefore not provably the model being compiled. Assert
# equality first, so any exporter/header drift fails loudly here.

header = assert_matches_header(
    weights1=W1,
    bias1=q_b1,
    weights2=W2,
    bias2=q_b2,
    input_scale=input_scale,
    acc1_scale=acc1_scale,
    hidden_scale=hidden_scale,
    acc2_scale=acc2_scale,
)

print("Header consistency: derived parameters match", header["path"].name)

# Match C header precision
input_scale_c = np.float32(input_scale)
acc1_scale_c = np.float32(acc1_scale)
hidden_scale_c = np.float32(hidden_scale)
acc2_scale_c = np.float32(acc2_scale)

mean_c = scaler.mean_.astype(np.float32)
std_c = scaler.scale_.astype(np.float32)

# ============================================================
# 3. C-COMPATIBLE ROUNDING
# ============================================================

def round_away_from_zero(values):

    values = np.asarray(values, dtype=np.float32)

    return np.where(
        values >= 0,
        np.floor(values + np.float32(0.5)),
        np.ceil(values - np.float32(0.5))
    ).astype(np.int32)


def quantize_like_c(values, scale):

    scaled = (
        np.asarray(values, dtype=np.float32)
        / np.float32(scale)
    )

    return np.clip(
        round_away_from_zero(scaled),
        -127,
        127
    ).astype(np.int8)


# ============================================================
# 4. PYTHON INT8 INFERENCE
# ============================================================

def python_int8_predict(raw_features):

    raw_features = np.asarray(
        raw_features,
        dtype=np.float32
    )

    # Input normalization
    normalized = (
        raw_features - mean_c
    ) / std_c

    # Input INT8 quantization
    q_input = quantize_like_c(
        normalized,
        input_scale_c
    )

    # First dense layer
    acc1 = (
        q_input.astype(np.int32)
        @ W1.astype(np.int32)
        + q_b1
    )

    # ReLU
    acc1 = np.maximum(acc1, 0)

    # Hidden activation
    hidden_real = (
        acc1.astype(np.float32)
        * acc1_scale_c
    )

    # Hidden INT8 quantization
    hidden = quantize_like_c(
        hidden_real,
        hidden_scale_c
    )

    hidden = np.maximum(
        hidden,
        0
    ).astype(np.int8)

    # Second dense layer
    output_sum = (
        int(q_b2[0])
        + int(
            np.sum(
                hidden.astype(np.int32)
                * W2.astype(np.int32)
            )
        )
    )

    # Output logit
    logit = np.float32(
        np.float32(output_sum) * acc2_scale_c
    )

    prediction = 1 if output_sum >= 0 else 0

    return float(logit), prediction


# ============================================================
# 5. RUN PYTHON INFERENCE
# ============================================================

python_results = [
    python_int8_predict(sample)
    for sample in X_test
]

python_logits = np.array(
    [result[0] for result in python_results]
)

python_predictions = np.array(
    [result[1] for result in python_results]
)

# ============================================================
# 6. RUN C INFERENCE
# ============================================================

input_text = "".join(
    f"{float(row[0]):.9g} "
    f"{float(row[1]):.9g} "
    f"{float(row[2]):.9g}\n"
    for row in X_test
)

FIRMWARE_DIR = BASE_DIR.parent / "firmware"

binary = FIRMWARE_DIR / "verify_int8_host"
if not binary.exists():
    binary = FIRMWARE_DIR / "verify_int8_host.exe"

if not binary.exists():
    raise SystemExit(
        "Host verification binary not found in "
        + str(FIRMWARE_DIR)
        + ". Build it first:  make -C firmware hosttools"
    )

process = subprocess.run(
    [str(binary)],
    input=input_text,
    text=True,
    capture_output=True,
    check=True
)

c_results = np.array(
    [
        [float(value) for value in line.split()]
        for line in process.stdout.strip().splitlines()
    ],
    dtype=np.float64
)

c_logits = c_results[:, 0]
c_predictions = c_results[:, 1].astype(np.int32)

# ============================================================
# 7. PERFORMANCE AND NUMERICAL COMPARISON
# ============================================================

mismatches = np.sum(
    python_predictions != c_predictions
)

agreement = (
    np.mean(python_predictions == c_predictions) * 100
)

python_accuracy = np.mean(
    python_predictions == y_test
) * 100

c_accuracy = np.mean(
    c_predictions == y_test
) * 100

logit_errors = np.abs(
    python_logits - c_logits
)

# ============================================================
# 8. REPORT
# ============================================================

print("\nTINYGUARD-AI C vs PYTHON INT8 VERIFICATION")
print("==========================================")

print("Test samples:", len(X_test))

print("\nPREDICTION RESULTS")
print("------------------")

print(f"Python accuracy: {python_accuracy:.2f}%")
print(f"C accuracy: {c_accuracy:.2f}%")

print("Prediction mismatches:", mismatches)

print(
    f"Prediction agreement: {agreement:.4f}%"
)

print("\nNUMERICAL COMPARISON")
print("-------------------")

print(
    "Maximum logit difference:",
    np.max(logit_errors)
)

print(
    "Mean logit difference:",
    np.mean(logit_errors)
)

print("\nDIAGNOSTIC: FIRST 10 SAMPLES")
print("----------------------------")

print("True labels:       ", y_test[:10])
print("Python predictions:", python_predictions[:10])
print("C predictions:     ", c_predictions[:10])

print("\nPREDICTION DISTRIBUTION")
print("-----------------------")

print(
    "Python normal:",
    np.sum(python_predictions == 0)
)

print(
    "Python anomaly:",
    np.sum(python_predictions == 1)
)

print(
    "C normal:",
    np.sum(c_predictions == 0)
)

print(
    "C anomaly:",
    np.sum(c_predictions == 1)
)

print("\nFINAL RESULT")
print("------------")

if mismatches == 0:
    print("SUCCESS: C and Python predictions match.")
else:
    print("WARNING: Prediction differences detected.")

if np.max(logit_errors) < 0.0001:
    print("SUCCESS: C and Python logits agree within tolerance.")
else:
    print("NOTICE: Logit differences need investigation.")
