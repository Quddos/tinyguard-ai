
import joblib
from pathlib import Path

# Load trained model and scaler
_BASE = Path(__file__).resolve().parent

model = joblib.load(_BASE / "models" / "tinyguard_mlp.pkl")
scaler = joblib.load(_BASE / "models" / "tinyguard_scaler.pkl")

# Verify expected network architecture
assert model.n_features_in_ == 3
assert model.hidden_layer_sizes == (8,)
assert len(model.coefs_) == 2

def format_array(values):
    return ", ".join(f"{float(v):.9g}f" for v in values)

def format_matrix(matrix):
    rows = []
    for row in matrix:
        rows.append("    {" + format_array(row) + "}")
    return ",\n".join(rows)

header = f"""#ifndef TINYGUARD_MODEL_PARAMS_H
#define TINYGUARD_MODEL_PARAMS_H

/* Automatically exported from the trained Python MLP.
   Input order: mean_absolute, rms, peak.
   Hidden activation: ReLU.
   Output: binary classification logit.
*/

#define MODEL_INPUTS 3
#define MODEL_HIDDEN 8

static const float scaler_mean[MODEL_INPUTS] = {{
    {format_array(scaler.mean_)}
}};

static const float scaler_scale[MODEL_INPUTS] = {{
    {format_array(scaler.scale_)}
}};

static const float weights1[MODEL_INPUTS][MODEL_HIDDEN] = {{
{format_matrix(model.coefs_[0])}
}};

static const float bias1[MODEL_HIDDEN] = {{
    {format_array(model.intercepts_[0])}
}};

static const float weights2[MODEL_HIDDEN] = {{
    {format_array(model.coefs_[1].flatten())}
}};

static const float bias2 = {float(model.intercepts_[1][0]):.9g}f;

#endif
"""

BASE_DIR = Path(__file__).resolve().parent

# Write LF explicitly so the export is byte-deterministic on every platform and
# the determinism test can compare the regenerated file against the committed
# one directly.
outputs = [
    BASE_DIR / "models" / "model_params.h",
    BASE_DIR.parent / "firmware" / "model_params.h",
]

for output in outputs:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="\n") as f:
        f.write(header)

print("Model parameters exported successfully.")
for output in outputs:
    print("Output:", output)
print("Input features:", model.n_features_in_)
print("Hidden neurons:", model.hidden_layer_sizes[0])
print("Output classes:", model.classes_)
