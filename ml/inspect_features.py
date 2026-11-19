
import pandas as pd

df = pd.read_csv("data/tinyguard_vibration_dataset.csv")

features = ["mean_absolute", "rms", "peak"]

print("\nFEATURE DISTRIBUTIONS")
print("=====================")

for feature in features:

    print(f"\n{feature.upper()}")

    print(
        df.groupby("label")[feature].describe()[
            ["min", "25%", "50%", "75%", "max"]
        ]
    )

print("\nSCENARIO DISTRIBUTIONS")
print("======================")

print(
    df.groupby(["scenario", "label"])[features]
      .agg(["min", "median", "max"])
)

