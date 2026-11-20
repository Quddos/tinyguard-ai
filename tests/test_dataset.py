"""
Dataset integrity tests.

These guard the committed research asset: shape, balance, label consistency and
the honest characterisation of its separability.
"""

from __future__ import annotations

import numpy as np

FEATURES = ["mean_absolute", "rms", "peak"]
SAMPLE_COLUMNS = [f"sample_{i}" for i in range(16)]
SCENARIOS = {"healthy", "high_amplitude", "impulsive", "mixed_frequency"}


def test_dataset_shape_and_columns(dataset):
    assert len(dataset) == 3000
    expected = set(FEATURES) | {"label", "scenario"} | set(SAMPLE_COLUMNS)
    assert set(dataset.columns) == expected


def test_classes_are_balanced(dataset):
    counts = dataset["label"].value_counts().to_dict()
    assert counts == {0: 1500, 1: 1500}


def test_no_missing_values(dataset):
    assert not dataset.isnull().to_numpy().any()


def test_scenario_label_consistency(dataset):
    """healthy windows are label 0; every fault scenario is label 1."""
    assert set(dataset["scenario"].unique()) == SCENARIOS

    healthy = dataset[dataset.scenario == "healthy"]
    assert (healthy["label"] == 0).all()

    faults = dataset[dataset.scenario != "healthy"]
    assert (faults["label"] == 1).all()


def test_raw_samples_are_integers(dataset):
    values = dataset[SAMPLE_COLUMNS].to_numpy()
    assert np.issubdtype(values.dtype, np.integer)


def test_no_feature_vector_carries_both_labels(dataset):
    """
    Features are coarse integers so rows collapse, but a vector mapping to both
    classes would make the task genuinely ambiguous.
    """
    conflicting = dataset.groupby(FEATURES)["label"].nunique().gt(1).sum()
    assert int(conflicting) == 0


def test_dataset_is_known_to_be_redundant(dataset):
    """
    Document the redundancy as a fact rather than letting it surprise a
    reader: there are far fewer unique feature vectors than rows, which is why
    a naive random split leaks.
    """
    unique = dataset[FEATURES].drop_duplicates().shape[0]
    assert unique < len(dataset), "expected duplicate feature vectors"
    assert unique > 1000, f"only {unique} unique vectors, dataset may be broken"


def test_classes_are_jointly_separable(dataset):
    """
    Pin the central scientific caveat: the classes are separable by
    construction. A linear model should reach a perfect score. If this ever
    stops holding, the README's interpretation section is wrong and must change.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = dataset[FEATURES].to_numpy(dtype=np.float64)
    y = dataset["label"].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(), LogisticRegression(random_state=42)
    ).fit(X_train, y_train)

    assert model.score(X_test, y_test) >= 0.99, (
        "logistic regression no longer separates the classes; the "
        "'separable by construction' caveat in the README needs revisiting"
    )


def test_healthy_windows_have_lower_energy_than_faults(dataset):
    """Directional sanity check on the generator's physics."""
    healthy = dataset[dataset.label == 0]
    faults = dataset[dataset.label == 1]

    for feature in FEATURES:
        assert healthy[feature].median() < faults[feature].median(), (
            f"healthy median {feature} is not below the fault median"
        )
