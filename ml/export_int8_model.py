
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from pathlib import Path

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = BASE_DIR / "data" / "tinyguard_vibration_dataset.csv"

OUTPUT_FILE = (
    BASE_DIR.parent / "firmware" / "model_int8_params.h"
)

# ============================================================
# 2. LOAD TRAINED MODEL, SCALER AND DATASET
# ============================================================

model = joblib.load(MODEL_DIR / "tinyguard_mlp.pkl")
scaler = joblib.load(MODEL_DIR / "tinyguard_scaler.pkl")

df = pd.read_csv(DATA_FILE)

features = ["mean_absolute", "rms", "peak"]

# Network dimensions, asserted against the trained model below so a retrained
# model of a different shape fails loudly instead of emitting a wrong header.
INPUT_UNITS = 3
HIDDEN_UNITS = 8

X = df[features].values.astype(np.float32)

assert model.n_features_in_ == INPUT_UNITS, model.n_features_in_
assert model.hidden_layer_sizes == (HIDDEN_UNITS,), model.hidden_layer_sizes
assert len(model.coefs_) == 2, len(model.coefs_)

X_train, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

X_train_scaled = scaler.transform(X_train).astype(np.float32)

# ============================================================
# 3. QUANTIZATION FUNCTION
# ============================================================

def quantize(values):
    """
    Symmetric per-tensor INT8 quantization.

    DEFECT FIX: guard against an all-zero tensor. Previously this divided by
    zero when max(|values|) == 0, silently producing NaN scales and a header
    full of garbage. quantization_test.py already had this guard; the exporter
    that actually writes the shipped header did not.
    """
    maximum = np.max(np.abs(values))

    scale = maximum / 127.0 if maximum > 0 else 1.0

    q_values = np.clip(
        np.rint(values / scale),
        -127,
        127
    ).astype(np.int8)

    return q_values, scale


def format_int_array(values):

    values = np.asarray(values).flatten()

    return ", ".join(str(int(v)) for v in values)


def format_float_array(values):

    values = np.asarray(values).flatten()

    return ", ".join(
        f"{float(v):.12g}f" for v in values
    )


# ============================================================
# 4. INPUT QUANTIZATION
# ============================================================

q_input_train, input_scale = quantize(X_train_scaled)

# ============================================================
# 5. FIRST NEURAL NETWORK LAYER
# ============================================================

W1, W1_scale = quantize(model.coefs_[0])

acc1_scale = input_scale * W1_scale

q_b1 = np.rint(
    model.intercepts_[0] / acc1_scale
).astype(np.int32)

# ============================================================
# 6. HIDDEN ACTIVATION CALIBRATION
# ============================================================

acc1_train = (
    q_input_train.astype(np.int32)
    @ W1.astype(np.int32)
    + q_b1
)

hidden_train = np.maximum(acc1_train, 0) * acc1_scale

q_hidden_train, hidden_scale = quantize(hidden_train)

# ============================================================
# 7. SECOND NEURAL NETWORK LAYER
# ============================================================

W2, W2_scale = quantize(model.coefs_[1])

acc2_scale = hidden_scale * W2_scale

q_b2 = np.rint(
    model.intercepts_[1] / acc2_scale
).astype(np.int32)

# ============================================================
# 8. EXPORT MODEL PARAMETERS TO ARM C HEADER
# ============================================================

with open(OUTPUT_FILE, "w", newline="\n") as f:

    f.write("#ifndef MODEL_INT8_PARAMS_H\n")
    f.write("#define MODEL_INT8_PARAMS_H\n\n")

    f.write("#include <stdint.h>\n\n")

    # Model dimensions
    f.write("#define MODEL_INPUTS 3\n")
    f.write("#define MODEL_HIDDEN 8\n\n")

    # Quantization scales
    f.write(f"#define INPUT_SCALE {input_scale:.12g}f\n")
    f.write(f"#define W1_SCALE {W1_scale:.12g}f\n")
    f.write(f"#define ACC1_SCALE {acc1_scale:.12g}f\n")
    f.write(f"#define HIDDEN_SCALE {hidden_scale:.12g}f\n")
    f.write(f"#define W2_SCALE {W2_scale:.12g}f\n")
    f.write(f"#define ACC2_SCALE {acc2_scale:.12g}f\n\n")

    # Original StandardScaler parameters
    f.write("static const float scaler_mean[3] = {\n")
    f.write("    " + format_float_array(scaler.mean_) + "\n")
    f.write("};\n\n")

    f.write("static const float scaler_std[3] = {\n")
    f.write("    " + format_float_array(scaler.scale_) + "\n")
    f.write("};\n\n")

    # First layer INT8 weights
    f.write("static const int8_t weights1[3][8] = {\n")

    for row in W1:
        f.write("    {" + format_int_array(row) + "},\n")

    f.write("};\n\n")

    # First layer INT32 biases
    f.write("static const int32_t bias1[8] = {\n")
    f.write("    " + format_int_array(q_b1) + "\n")
    f.write("};\n\n")

    # Second layer INT8 weights
    f.write("static const int8_t weights2[8] = {\n")
    f.write("    " + format_int_array(W2) + "\n")
    f.write("};\n\n")

    # Second layer INT32 bias
    f.write("static const int32_t bias2[1] = {\n")
    f.write("    " + format_int_array(q_b2) + "\n")
    f.write("};\n\n")

    f.write("#endif\n")

# ============================================================
# 9. EXPORT REPORT
# ============================================================

print("\nTINYGUARD-AI INT8 MODEL EXPORT")
print("================================")

print("Output:", OUTPUT_FILE)

print("\nMODEL INFORMATION")
print("-----------------")

print("Input features:", INPUT_UNITS)
print("Hidden neurons:", HIDDEN_UNITS)
print("Output neurons:", 1)

print("\nQUANTIZATION")
print("------------")

print("Input scale:", input_scale)
print("Layer 1 weight scale:", W1_scale)
print("Layer 1 accumulator scale:", acc1_scale)
print("Hidden activation scale:", hidden_scale)
print("Layer 2 weight scale:", W2_scale)
print("Layer 2 accumulator scale:", acc2_scale)

print("\nSCALER PARAMETERS")
print("-----------------")

print("Scaler means:", scaler.mean_)
print("Scaler standard deviations:", scaler.scale_)

print("\nMEMORY")
print("------")

print("INT8 weight storage:", W1.nbytes + W2.nbytes, "bytes")
print("INT32 bias storage:", q_b1.nbytes + q_b2.nbytes, "bytes")
print("Total quantized parameter storage:",
      W1.nbytes + W2.nbytes + q_b1.nbytes + q_b2.nbytes,
      "bytes")

print("\nINT8 MODEL EXPORT COMPLETE")
