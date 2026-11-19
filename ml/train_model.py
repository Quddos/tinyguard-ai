
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib
from pathlib import Path

# Load dataset
df = pd.read_csv("data/tinyguard_vibration_dataset.csv")

features = [
    "mean_absolute",
    "rms",
    "peak"
]

X = df[features].values
y = df["label"].values

# Separate training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardize the input features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build neural network
model = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation="relu",
    solver="lbfgs",
    max_iter=2000,
    random_state=42
)

# Train
model.fit(X_train_scaled, y_train)

# Evaluate
predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print("\nTinyGuard-AI Neural Network")
print("---------------------------")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Healthy", "Abnormal"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model and scaler
Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/tinyguard_mlp.pkl"
)

joblib.dump(
    scaler,
    "models/tinyguard_scaler.pkl"
)

print("\nModel saved successfully.")
