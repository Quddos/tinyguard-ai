
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Load trained model and scaler
model = joblib.load("models/tinyguard_mlp.pkl")
scaler = joblib.load("models/tinyguard_scaler.pkl")

# Load synthetic dataset
df = pd.read_csv("data/tinyguard_vibration_dataset.csv")

features = ["mean_absolute", "rms", "peak"]

X = df[features].to_numpy(dtype=np.float64)
y = df["label"].to_numpy()

# Use training partition for calibration only
X_train, _, _, _ = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Floating-point forward pass
X_scaled = scaler.transform(X_train)

hidden = np.maximum(
    0,
    X_scaled @ model.coefs_[0] + model.intercepts_[0]
)

logits = (
    hidden @ model.coefs_[1] + model.intercepts_[1]
).ravel()

tensors = {
    "Normalized inputs": X_scaled,
    "Layer 1 weights": model.coefs_[0],
    "Layer 1 biases": model.intercepts_[0],
    "Hidden activations": hidden,
    "Layer 2 weights": model.coefs_[1],
    "Layer 2 biases": model.intercepts_[1],
    "Output logits": logits
}

print("\nTINYGUARD-AI QUANTIZATION ANALYSIS")
print("=================================")

for name, values in tensors.items():

    values = np.asarray(values, dtype=np.float64)

    maximum = np.max(np.abs(values))

    scale = maximum / 127.0 if maximum > 0 else 1.0

    quantized = np.clip(
        np.round(values / scale),
        -127,
        127
    ).astype(np.int8)

    reconstructed = quantized.astype(np.float64) * scale

    error = np.abs(values - reconstructed)

    print(f"\n{name}")
    print("-" * len(name))
    print("Minimum:", np.min(values))
    print("Maximum:", np.max(values))
    print("Maximum absolute:", maximum)
    print("Proposed INT8 scale:", scale)
    print("Quantization error (max):", np.max(error))
    print("Quantization error (mean):", np.mean(error))

print("\nCalibration samples:", len(X_train))
