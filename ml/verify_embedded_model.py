
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from header_params import load_fp32_header

BASE_DIR = Path(__file__).resolve().parent
FIRMWARE_DIR = BASE_DIR.parent / "firmware"

# Load dataset and trained model
df = pd.read_csv(BASE_DIR / "data" / "tinyguard_vibration_dataset.csv")

model = joblib.load(BASE_DIR / "models" / "tinyguard_mlp.pkl")
scaler = joblib.load(BASE_DIR / "models" / "tinyguard_scaler.pkl")

# GUARD (defect fix): assert the committed FP32 header matches the .pkl, so the
# model being verified is provably the model being compiled into the firmware.
fp32 = load_fp32_header()

np.testing.assert_allclose(
    fp32["weights1"], model.coefs_[0].astype(np.float32), rtol=0, atol=1e-6,
    err_msg="weights1 in model_params.h does not match tinyguard_mlp.pkl",
)
np.testing.assert_allclose(
    fp32["bias1"], model.intercepts_[0].astype(np.float32), rtol=0, atol=1e-6,
    err_msg="bias1 in model_params.h does not match tinyguard_mlp.pkl",
)
np.testing.assert_allclose(
    fp32["weights2"], model.coefs_[1].ravel().astype(np.float32), rtol=0, atol=1e-6,
    err_msg="weights2 in model_params.h does not match tinyguard_mlp.pkl",
)
np.testing.assert_allclose(
    fp32["scaler_mean"], scaler.mean_.astype(np.float32), rtol=0, atol=1e-4,
    err_msg="scaler_mean in model_params.h does not match tinyguard_scaler.pkl",
)
np.testing.assert_allclose(
    fp32["scaler_scale"], scaler.scale_.astype(np.float32), rtol=0, atol=1e-4,
    err_msg="scaler_scale in model_params.h does not match tinyguard_scaler.pkl",
)

print("Header consistency: FP32 header matches the trained .pkl")

features = ["mean_absolute", "rms", "peak"]

X = df[features].to_numpy(dtype=np.float64)
y = df["label"].to_numpy()

# Python reference inference
X_scaled = scaler.transform(X)

hidden = np.maximum(
    0,
    X_scaled @ model.coefs_[0] + model.intercepts_[0]
)

python_logits = (
    hidden @ model.coefs_[1] + model.intercepts_[1]
).ravel()

python_predictions = (python_logits >= 0).astype(int)

# Send all feature vectors to the C executable
input_text = "".join(
    f"{a:.9g},{b:.9g},{c:.9g}\n"
    for a, b, c in X
)

binary = FIRMWARE_DIR / "verify_model_host"
if not binary.exists():
    binary = FIRMWARE_DIR / "verify_model_host.exe"

if not binary.exists():
    raise SystemExit(
        "Host verification binary not found in "
        + str(FIRMWARE_DIR)
        + ". Build it first:  make -C firmware hosttools"
    )

result = subprocess.run(
    [str(binary)],
    input=input_text,
    text=True,
    capture_output=True,
    check=True
)

# Parse C output
c_results = np.array([
    [float(value) for value in line.split(",")]
    for line in result.stdout.strip().splitlines()
])

c_logits = c_results[:, 0]
c_predictions = c_results[:, 1].astype(int)

# Compare predictions and logits
prediction_mismatches = np.sum(
    python_predictions != c_predictions
)

logit_differences = np.abs(
    python_logits - c_logits
)

agreement = np.mean(
    python_predictions == c_predictions
) * 100

print("\nTINYGUARD-AI EMBEDDED MODEL VERIFICATION")
print("=======================================")

print("Total samples:", len(X))
print("Python accuracy:", np.mean(python_predictions == y))
print("C accuracy:", np.mean(c_predictions == y))

print("Prediction mismatches:", prediction_mismatches)
print("Prediction agreement:", f"{agreement:.4f}%")

print(
    "Maximum logit difference:",
    f"{np.max(logit_differences):.9g}"
)

print(
    "Mean logit difference:",
    f"{np.mean(logit_differences):.9g}"
)

if prediction_mismatches == 0:
    print("\nSUCCESS: Python and C predictions match.")
else:
    print("\nWARNING: Prediction mismatches detected.")
