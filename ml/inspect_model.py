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

sample = [[11, 11, 16]]

scaled = scaler.transform(sample)

print("Original features:", sample)
print("Normalized features:", scaled)
print("Predicted class:", model.predict(sample))
print("Class probabilities:", model.predict_proba(sample))
