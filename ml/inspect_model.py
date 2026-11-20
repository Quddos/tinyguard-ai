import joblib

model = joblib.load("models/tinyguard_mlp.pkl")
scaler = joblib.load("models/tinyguard_scaler.pkl")

print("\nNEURAL NETWORK ARCHITECTURE")
print("===========================")

print("Input features:", model.n_features_in_)
print("Hidden layers:", model.hidden_layer_sizes)
print("Output classes:", model.classes_)

print("\nWEIGHT MATRIX SHAPES")
print("====================")

for i, weights in enumerate(model.coefs_):
    print(f"Layer {i + 1}: {weights.shape}")

print("\nBIAS VECTOR SHAPES")
print("==================")

for i, bias in enumerate(model.intercepts_):
    print(f"Layer {i + 1}: {bias.shape}")

print("\nSCALER PARAMETERS")
print("=================")

print("Feature means:", scaler.mean_)
print("Feature scales:", scaler.scale_)

print("\nMODEL OUTPUT EXAMPLE")
print("====================")

# Features of the healthy window used by firmware/sensor_sim.c
# (mean_absolute, rms, peak), computed with the same integer maths.
sample = [[11, 11, 16]]

scaled = scaler.transform(sample)

# DEFECT FIX: predict on the SCALED features. The model was trained on
# StandardScaler output, so feeding raw features here previously reported
# class 1 (anomaly) for a healthy window.
print("Original features:", sample)
print("Normalized features:", scaled)
print("Predicted class:", model.predict(scaled))
print("Class probabilities:", model.predict_proba(scaled))
