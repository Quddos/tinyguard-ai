
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# ------------------------------------------------
# 1. Load model and dataset
# ------------------------------------------------

model = joblib.load("models/tinyguard_mlp.pkl")
scaler = joblib.load("models/tinyguard_scaler.pkl")

df = pd.read_csv("data/tinyguard_vibration_dataset.csv")

features = ["mean_absolute", "rms", "peak"]

X = df[features].to_numpy(dtype=np.float64)
y = df["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# 2. Floating-point reference model
# ------------------------------------------------

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

def float_forward(X_scaled):
    hidden = np.maximum(
        0,
        X_scaled @ model.coefs_[0] + model.intercepts_[0]
    )

    logits = (
        hidden @ model.coefs_[1] + model.intercepts_[1]
    ).ravel()

    return logits

float_logits = float_forward(X_test_scaled)
float_predictions = (float_logits >= 0).astype(np.int32)

# ------------------------------------------------
# 3. Symmetric INT8 quantization helpers
# ------------------------------------------------

def quantize_symmetric(values):
    maximum = np.max(np.abs(values))
    scale = maximum / 127.0 if maximum > 0 else 1.0

    quantized = np.clip(
        np.rint(values / scale),
        -127,
        127
    ).astype(np.int8)

    return quantized, scale

# ------------------------------------------------
# 4. Calibrate input and hidden activation scales
# ------------------------------------------------

q_input_train, input_scale = quantize_symmetric(
    X_train_scaled
)

W1 = model.coefs_[0]
b1 = model.intercepts_[0]

q_W1, W1_scale = quantize_symmetric(W1)

# First layer accumulator scale
acc1_scale = input_scale * W1_scale

# Quantize first-layer biases into INT32
q_b1 = np.rint(
    b1 / acc1_scale
).astype(np.int32)

# Calibrate hidden activations using training data
acc1_train = (
    q_input_train.astype(np.int32)
    @ q_W1.astype(np.int32)
    + q_b1
)

hidden_train = np.maximum(
    0,
    acc1_train.astype(np.float64) * acc1_scale
)

hidden_scale = (
    np.max(hidden_train) / 127.0
    if np.max(hidden_train) > 0
    else 1.0
)

# ------------------------------------------------
# 5. Quantize second-layer parameters
# ------------------------------------------------

W2 = model.coefs_[1].reshape(-1)
b2 = float(model.intercepts_[1][0])

q_W2, W2_scale = quantize_symmetric(W2)

acc2_scale = hidden_scale * W2_scale

q_b2 = np.int32(
    np.rint(b2 / acc2_scale)
)

# ------------------------------------------------
# 6. INT8 forward pass
# ------------------------------------------------

def int8_forward(X_scaled):

    # Quantize input
    q_X = np.clip(
        np.rint(X_scaled / input_scale),
        -127,
        127
    ).astype(np.int8)

    # Layer 1: INT8 x INT8 -> INT32
    acc1 = (
        q_X.astype(np.int32)
        @ q_W1.astype(np.int32)
        + q_b1
    )

    # ReLU and requantize hidden activations
    hidden_real = (
        np.maximum(acc1, 0).astype(np.float64)
        * acc1_scale
    )

    q_hidden = np.clip(
        np.rint(hidden_real / hidden_scale),
        0,
        127
    ).astype(np.int8)

    # Layer 2: INT8 x INT8 -> INT32
    acc2 = (
        q_hidden.astype(np.int32)
        @ q_W2.astype(np.int32)
        + q_b2
    )

    # Convert output accumulator to approximate float logit
    quantized_logits = acc2.astype(np.float64) * acc2_scale

    predictions = (acc2 >= 0).astype(np.int32)

    return quantized_logits, predictions

quant_logits, quant_predictions = int8_forward(X_test_scaled)

# ------------------------------------------------
# 7. Evaluate
# ------------------------------------------------

mismatches = np.sum(
    float_predictions != quant_predictions
)

agreement = np.mean(
    float_predictions == quant_predictions
) * 100

quant_accuracy = np.mean(
    quant_predictions == y_test
) * 100

float_accuracy = np.mean(
    float_predictions == y_test
) * 100

logit_error = np.abs(
    float_logits - quant_logits
)

print("\nTINYGUARD-AI INT8 QUANTIZATION TEST")
print("==================================")

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nQUANTIZATION SCALES")
print("-------------------")
print("Input scale:", input_scale)
print("Layer 1 weight scale:", W1_scale)
print("Layer 1 accumulator scale:", acc1_scale)
print("Hidden activation scale:", hidden_scale)
print("Layer 2 weight scale:", W2_scale)
print("Layer 2 accumulator scale:", acc2_scale)

print("\nMODEL PERFORMANCE")
print("-----------------")
print("Float accuracy:", float_accuracy)
print("INT8 accuracy:", quant_accuracy)
print("Prediction mismatches:", mismatches)
print("Prediction agreement:", f"{agreement:.4f}%")

print("\nNUMERICAL ERROR")
print("---------------")
print("Maximum logit error:", np.max(logit_error))
print("Mean logit error:", np.mean(logit_error))

print("\nINT32 BIAS RANGES")
print("-----------------")
print("Layer 1 bias range:", q_b1.min(), q_b1.max())
print("Layer 2 bias:", int(q_b2))

print("\nINT8 TEST COMPLETE")
