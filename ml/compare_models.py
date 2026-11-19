
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

df = pd.read_csv("data/tinyguard_vibration_dataset.csv")

features = ["mean_absolute", "rms", "peak"]

X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Dummy baseline": DummyClassifier(strategy="most_frequent"),

    "Threshold baseline": None,

    "Logistic regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    ),

    "Decision tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    "Neural network": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="relu",
            solver="lbfgs",
            max_iter=2000,
            random_state=42
        )
    )
}

results = []

for name, model in models.items():

    if name == "Threshold baseline":
        predictions = (
            (X_test["rms"].values >= 50) |
            (X_test["peak"].values >= 100)
        ).astype(int)

    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions)
    })

    print(f"\n{name}")
    print(confusion_matrix(y_test, predictions))

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("================")
print(results_df.to_string(index=False))
