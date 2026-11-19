
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("data/tinyguard_vibration_dataset.csv")

features = ["mean_absolute", "rms", "peak"]

healthy = df[df["label"] == 0]

abnormal = df[df["label"] == 1]

scenarios = [
    "high_amplitude",
    "impulsive",
    "mixed_frequency"
]

for held_out in scenarios:

    print("\n================================")
    print("HELD-OUT SCENARIO:", held_out)
    print("================================")

    # Reserve 20% healthy data for testing
    healthy_train, healthy_test = train_test_split(
        healthy,
        test_size=0.20,
        random_state=42
    )

    # Train only on two abnormal scenarios
    abnormal_train = abnormal[
        abnormal["scenario"] != held_out
    ]

    # Test only on the unseen abnormal scenario
    abnormal_test = abnormal[
        abnormal["scenario"] == held_out
    ]

    train_df = pd.concat([
        healthy_train,
        abnormal_train
    ])

    test_df = pd.concat([
        healthy_test,
        abnormal_test
    ])

    X_train = train_df[features].values
    y_train = train_df["label"].values

    X_test = test_df[features].values
    y_test = test_df["label"].values

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="lbfgs",
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Training samples:", len(train_df))
    print("Testing samples:", len(test_df))

    print("\nResults:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Healthy", "Abnormal"],
        zero_division=0
    ))
