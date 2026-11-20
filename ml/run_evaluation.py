"""
TinyGuard-AI consolidated evaluation -> machine-readable metrics.

Runs the full evaluation suite once and writes JSON artifacts under results/
for the Streamlit dashboard, the README results table and any paper figures.

Every number here is COMPUTED at run time. Nothing is hardcoded. If the model,
dataset or quantization changes, re-running this script is the only way the
reported figures change.

SCIENTIFIC LIMITATIONS carried into every artifact (see results/provenance.json):
  * The dataset is SYNTHETIC. No physical sensor was ever read.
  * ARM Cortex-M4 execution is QEMU EMULATION, not silicon.
  * The two classes are near-separable by construction, so accuracy figures
    describe the generator, not model capability. Baselines are reported
    alongside so this is visible rather than implied.

Usage:
    python run_evaluation.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from header_params import (
    float32_ulp_distance,
    load_fp32_header,
    load_int8_header,
    scale_ulp_report,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DATA_FILE = BASE_DIR / "data" / "tinyguard_vibration_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = REPO_DIR / "results"

FEATURES = ["mean_absolute", "rms", "peak"]
SCENARIOS = ["high_amplitude", "impulsive", "mixed_frequency"]
TEST_SIZE = 0.20
SEED = 42

# Windows hardcoded in firmware/sensor_sim.c, mirrored here so the dashboard
# can show what the QEMU demo actually runs on. tests/test_firmware_sync.py
# asserts these stay identical to the C source.
FIRMWARE_WINDOWS = {
    "SENSOR_NORMAL": [12, -10, 15, -13, 8, -14, 11, -9,
                      16, -12, 10, -11, 14, -8, 9, -13],
    "SENSOR_ABNORMAL_TYPICAL": [-12, 39, 39, -11, -57, -14, 29, 46,
                                -11, -47, -39, 13, 74, 10, -25, -28],
    "SENSOR_ABNORMAL": [120, -110, 145, -130, 180, -160, 135, -190,
                        170, -140, 155, -175, 190, -120, 165, -185],
}


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def jsonable(obj):
    """Recursively convert numpy scalars/arrays into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_rel(path: Path) -> str:
    return str(Path(path).relative_to(REPO_DIR)).replace("\\", "/")


def write_json(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, sort_keys=False)
        f.write("\n")
    return path


def classification_block(y_true, y_pred, scores=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    block = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        block["roc_auc"] = float(roc_auc_score(y_true, scores))
    return block


def c_features(window):
    """Integer feature extraction identical to firmware/signal_features.c."""
    w = np.asarray(window, dtype=np.int64)
    a = np.abs(w)
    mean_absolute = int(a.sum() // w.size)
    mean_square = int((w ** 2).sum() // w.size)
    rms = int(np.sqrt(mean_square))
    peak = int(a.max())
    return mean_absolute, rms, peak


# ------------------------------------------------------------------
# load inputs
# ------------------------------------------------------------------

df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_DIR / "tinyguard_mlp.pkl")
scaler = joblib.load(MODEL_DIR / "tinyguard_scaler.pkl")

int8_header = load_int8_header()
fp32_header = load_fp32_header()

X = df[FEATURES].to_numpy(dtype=np.float64)
y = df["label"].to_numpy(dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)


def fp32_logits(X_raw, mdl=None, scl=None) -> np.ndarray:
    """Reference float forward pass (matches firmware/tinyguard_model.c)."""
    mdl = mdl if mdl is not None else model
    scl = scl if scl is not None else scaler
    Z = scl.transform(X_raw)
    hidden = np.maximum(0, Z @ mdl.coefs_[0] + mdl.intercepts_[0])
    return (hidden @ mdl.coefs_[1] + mdl.intercepts_[1]).ravel()


# ------------------------------------------------------------------
# 1. dataset summary
# ------------------------------------------------------------------

def build_dataset_summary() -> dict:
    per_class = {}
    for label, group in df.groupby("label"):
        per_class[int(label)] = {
            "name": "healthy" if int(label) == 0 else "abnormal",
            "windows": int(len(group)),
            "features": {
                feat: {
                    "min": float(group[feat].min()),
                    "p25": float(group[feat].quantile(0.25)),
                    "median": float(group[feat].median()),
                    "p75": float(group[feat].quantile(0.75)),
                    "max": float(group[feat].max()),
                    "mean": float(group[feat].mean()),
                    "std": float(group[feat].std(ddof=0)),
                }
                for feat in FEATURES
            },
        }

    healthy = df[df.label == 0]
    abnormal = df[df.label == 1]

    overlap = {
        feat: {
            "healthy_max": float(healthy[feat].max()),
            "abnormal_min": float(abnormal[feat].min()),
            "overlaps": bool(healthy[feat].max() >= abnormal[feat].min()),
        }
        for feat in FEATURES
    }

    keys = [f"{a}_{b}_{c}" for a, b, c in X.astype(np.int64)]
    unique_vectors = len(set(keys))

    train_keys = set(f"{a}_{b}_{c}" for a, b, c in X_train.astype(np.int64))
    test_keys = [f"{a}_{b}_{c}" for a, b, c in X_test.astype(np.int64)]
    leaked = sum(1 for k in test_keys if k in train_keys)

    conflicting = int(df.groupby(FEATURES)["label"].nunique().gt(1).sum())

    return {
        "source_file": repo_rel(DATA_FILE),
        "sha256": sha256(DATA_FILE),
        "data_origin": "synthetic",
        "physical_sensor_used": False,
        "total_windows": int(len(df)),
        "samples_per_window": 16,
        "feature_names": FEATURES,
        "class_distribution": {
            str(int(k)): int(v) for k, v in df["label"].value_counts().items()
        },
        "scenario_distribution": {
            str(k): int(v) for k, v in df["scenario"].value_counts().items()
        },
        "per_class": per_class,
        "per_feature_class_overlap": overlap,
        "separability_note": (
            "The classes are drawn from near-disjoint amplitude regimes. Any "
            "single feature overlaps slightly between classes, but the three "
            "features jointly separate them almost perfectly, which is why "
            "every classifier tested reaches the same score."
        ),
        "redundancy": {
            "unique_feature_vectors": int(unique_vectors),
            "duplicate_rows": int(len(df) - unique_vectors),
            "conflicting_duplicate_vectors": conflicting,
            "note": (
                "Features are coarse integers, so many windows collapse onto "
                "the same feature vector. No feature vector carries both "
                "labels."
            ),
        },
        "naive_split_leakage": {
            "split": (
                f"train_test_split(test_size={TEST_SIZE}, "
                f"random_state={SEED}, stratify=y)"
            ),
            "test_rows": int(len(test_keys)),
            "test_rows_whose_vector_appears_in_train": int(leaked),
            "fraction": float(leaked / len(test_keys)),
            "note": (
                "Because identical feature vectors straddle the split, the "
                "naive test accuracy is not a generalisation estimate. A "
                "leakage-free grouped split is reported in model_metrics.json."
            ),
        },
    }


# ------------------------------------------------------------------
# 2. model metrics, baselines, generalisation
# ------------------------------------------------------------------

def build_model_metrics() -> dict:
    logits_test = fp32_logits(X_test)
    preds_test = (logits_test >= 0).astype(np.int64)

    logits_all = fp32_logits(X)
    preds_all = (logits_all >= 0).astype(np.int64)

    shipped = {
        "architecture": {
            "inputs": int(model.n_features_in_),
            "hidden_units": int(model.hidden_layer_sizes[0]),
            "outputs": 1,
            "hidden_activation": "relu",
            "solver": str(model.solver),
            "random_state": SEED,
        },
        "naive_split_test": classification_block(
            y_test, preds_test, logits_test
        ),
        "whole_dataset": classification_block(y, preds_all, logits_all),
    }

    # leakage-free grouped split: identical vectors never straddle it
    groups = pd.Series(
        [f"{a}_{b}_{c}" for a, b, c in X.astype(np.int64)]
    ).factorize()[0]

    gtr, gte = next(
        GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=SEED
        ).split(X, y, groups=groups)
    )

    group_scaler = StandardScaler().fit(X[gtr])
    group_model = MLPClassifier(
        hidden_layer_sizes=(8,), activation="relu", solver="lbfgs",
        max_iter=2000, random_state=SEED,
    ).fit(group_scaler.transform(X[gtr]), y[gtr])

    g_logits = fp32_logits(X[gte], group_model, group_scaler)
    g_preds = (g_logits >= 0).astype(np.int64)

    grouped = {
        "description": (
            "Retrained on a GroupShuffleSplit grouped by unique feature "
            "vector, so no feature vector appears in both partitions. This "
            "removes the leakage reported in dataset_summary.json."
        ),
        "train_n": int(len(gtr)),
        "test_n": int(len(gte)),
        "shared_feature_vectors_across_split": int(
            len(set(groups[gtr]) & set(groups[gte]))
        ),
        "test": classification_block(y[gte], g_preds, g_logits),
    }

    # baselines on the same naive split
    baselines = {}

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baselines["dummy_most_frequent"] = classification_block(
        y_test, dummy.predict(X_test)
    )

    thresh_preds = (
        (X_test[:, FEATURES.index("rms")] >= 50)
        | (X_test[:, FEATURES.index("peak")] >= 100)
    ).astype(np.int64)
    baselines["hand_threshold_rms50_or_peak100"] = classification_block(
        y_test, thresh_preds
    )

    logreg = make_pipeline(
        StandardScaler(), LogisticRegression(random_state=SEED)
    ).fit(X_train, y_train)
    baselines["logistic_regression"] = classification_block(
        y_test, logreg.predict(X_test), logreg.decision_function(X_test)
    )

    tree = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(
        X_train, y_train
    )
    baselines["decision_tree_depth4"] = classification_block(
        y_test, tree.predict(X_test), tree.predict_proba(X_test)[:, 1]
    )

    baselines["mlp_8_shipped_architecture"] = shipped["naive_split_test"]

    # per-scenario accuracy for the shipped model
    per_scenario = {}
    for scenario, group in df.groupby("scenario"):
        gX = group[FEATURES].to_numpy(dtype=np.float64)
        gy = group["label"].to_numpy(dtype=np.int64)
        gp = (fp32_logits(gX) >= 0).astype(np.int64)
        per_scenario[str(scenario)] = {
            "windows": int(len(group)),
            "label": int(gy[0]),
            "correct": int((gp == gy).sum()),
            "accuracy": float((gp == gy).mean()),
        }

    # leave-one-abnormal-scenario-out
    healthy = df[df.label == 0]
    abnormal = df[df.label == 1]
    loso = {}

    for held_out in SCENARIOS:
        h_train, h_test = train_test_split(
            healthy, test_size=TEST_SIZE, random_state=SEED
        )
        a_train = abnormal[abnormal.scenario != held_out]
        a_test = abnormal[abnormal.scenario == held_out]

        tr = pd.concat([h_train, a_train])
        te = pd.concat([h_test, a_test])

        tr_X = tr[FEATURES].to_numpy(dtype=np.float64)
        sc = StandardScaler().fit(tr_X)
        mdl = MLPClassifier(
            hidden_layer_sizes=(8,), activation="relu", solver="lbfgs",
            max_iter=2000, random_state=SEED,
        ).fit(sc.transform(tr_X), tr["label"].to_numpy(dtype=np.int64))

        te_X = te[FEATURES].to_numpy(dtype=np.float64)
        te_y = te["label"].to_numpy(dtype=np.int64)
        te_logits = fp32_logits(te_X, mdl, sc)
        te_preds = (te_logits >= 0).astype(np.int64)

        loso[held_out] = {
            "train_windows": int(len(tr)),
            "test_windows": int(len(te)),
            "unseen_abnormal_windows": int(len(a_test)),
            "test": classification_block(te_y, te_preds, te_logits),
        }

    return {
        "split": {
            "method": "train_test_split",
            "test_size": TEST_SIZE,
            "random_state": SEED,
            "stratified": True,
            "train_n": int(len(X_train)),
            "test_n": int(len(X_test)),
        },
        "shipped_model": shipped,
        "leakage_free_grouped_split": grouped,
        "baselines": baselines,
        "baseline_interpretation": (
            "Logistic regression and a depth-4 decision tree reach the same "
            "accuracy as the 8-unit MLP, and so does the leakage-free "
            "grouped split. The accuracy therefore reflects the synthetic "
            "generator's separability, not the capacity of the neural "
            "network. The defensible contributions of this project are the "
            "INT8 quantization pipeline and the bit-exact C/Python "
            "equivalence, not the classification score."
        ),
        "per_scenario": per_scenario,
        "leave_one_scenario_out": loso,
    }


# ------------------------------------------------------------------
# 3. quantization: fidelity, error, footprint
# ------------------------------------------------------------------

def quantize_symmetric(values):
    maximum = np.max(np.abs(values))
    scale = maximum / 127.0 if maximum > 0 else 1.0
    q = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
    return q, scale


def round_away_from_zero_f32(values):
    v = np.asarray(values, dtype=np.float32)
    return np.where(
        v >= 0,
        np.floor(v + np.float32(0.5)),
        np.ceil(v - np.float32(0.5)),
    ).astype(np.int32)


def int8_forward_from_header(X_raw, h):
    """
    INT8 inference using ONLY the constants in the committed C header, with
    float32 arithmetic and C rounding semantics, mirroring
    firmware/tinyguard_int8.c.
    """
    raw = np.asarray(X_raw, dtype=np.float32)
    z = (raw - h["scaler_mean"]) / h["scaler_std"]

    q_in = np.clip(
        round_away_from_zero_f32(z / h["input_scale"]), -127, 127
    ).astype(np.int32)

    acc1 = q_in @ h["weights1"].astype(np.int32) + h["bias1"]
    acc1 = np.maximum(acc1, 0)

    hidden_real = acc1.astype(np.float32) * h["acc1_scale"]
    q_hidden = np.clip(
        round_away_from_zero_f32(hidden_real / h["hidden_scale"]), 0, 127
    ).astype(np.int32)

    acc2 = q_hidden @ h["weights2"].astype(np.int32) + int(h["bias2"][0])
    logits = (acc2.astype(np.float32) * h["acc2_scale"]).astype(np.float32)

    return logits, (acc2 >= 0).astype(np.int64), q_in, acc2


def derive_quantization_parameters():
    """Re-derive the INT8 constants from the .pkl, exactly as the exporter does."""
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    q_in_train, input_scale = quantize_symmetric(X_train_scaled)

    W1, w1_scale = quantize_symmetric(model.coefs_[0])
    acc1_scale = input_scale * w1_scale
    q_b1 = np.rint(model.intercepts_[0] / acc1_scale).astype(np.int32)

    acc1_train = q_in_train.astype(np.int32) @ W1.astype(np.int32) + q_b1
    hidden_train = np.maximum(acc1_train, 0) * acc1_scale
    _, hidden_scale = quantize_symmetric(hidden_train)

    W2, w2_scale = quantize_symmetric(model.coefs_[1].flatten())
    acc2_scale = hidden_scale * w2_scale
    q_b2 = np.rint(model.intercepts_[1] / acc2_scale).astype(np.int32)

    return {
        "weights1": W1, "bias1": q_b1,
        "weights2": W2, "bias2": q_b2,
        "input_scale": input_scale, "w1_scale": w1_scale,
        "acc1_scale": acc1_scale, "hidden_scale": hidden_scale,
        "w2_scale": w2_scale, "acc2_scale": acc2_scale,
    }


def build_quantization() -> dict:
    h = int8_header
    d = derive_quantization_parameters()

    # Integer tensors must be bit-exact: they alone decide every class.
    integers_exact = bool(
        np.array_equal(d["weights1"], h["weights1"])
        and np.array_equal(d["bias1"], h["bias1"])
        and np.array_equal(d["weights2"], h["weights2"])
        and np.array_equal(d["bias2"], h["bias2"])
    )

    # Scales are written as 12-significant-digit decimal text and parsed back
    # as float32, which can land 1 ULP away from the float64 cast. See
    # header_params.scale_ulp_report for the full explanation.
    ulps = scale_ulp_report(
        input_scale=d["input_scale"],
        acc1_scale=d["acc1_scale"],
        hidden_scale=d["hidden_scale"],
        acc2_scale=d["acc2_scale"],
        header=h,
    )

    # Prove the ULP gap is inert: run the full dataset through both scale sets
    # and require identical accumulators and predictions.
    h_derived = dict(h)
    h_derived.update(
        input_scale=np.float32(d["input_scale"]),
        acc1_scale=np.float32(d["acc1_scale"]),
        hidden_scale=np.float32(d["hidden_scale"]),
        acc2_scale=np.float32(d["acc2_scale"]),
    )
    log_h, pred_h, qin_h, acc_h = int8_forward_from_header(X, h)
    log_d, pred_d, qin_d, acc_d = int8_forward_from_header(X, h_derived)

    scale_equivalence = {
        "max_scale_ulp_distance": int(max(ulps.values())),
        "per_scale_ulp_distance": ulps,
        "identical_quantized_inputs": bool(np.array_equal(qin_h, qin_d)),
        "identical_output_accumulators": bool(np.array_equal(acc_h, acc_d)),
        "identical_predictions": bool(np.array_equal(pred_h, pred_d)),
        "max_logit_difference": float(np.abs(log_h - log_d).max()),
        "min_abs_output_accumulator": int(np.abs(acc_h).min()),
        "note": (
            "A 1-ULP difference between the header text and the float64-cast "
            "scale is a decimal-representation artifact, not exporter drift. "
            "It is inert here: quantized inputs, accumulators and predictions "
            "are bit-identical across the whole dataset, and the smallest "
            "decision margin is far larger than the perturbation."
        ),
    }

    header_consistent = bool(
        integers_exact
        and max(ulps.values()) <= 1
        and scale_equivalence["identical_predictions"]
        and scale_equivalence["identical_output_accumulators"]
    )

    # per-tensor quantization error on the calibration partition
    Z_train = scaler.transform(X_train)
    hidden_fp = np.maximum(0, Z_train @ model.coefs_[0] + model.intercepts_[0])

    tensors = {
        "normalized_inputs": Z_train,
        "layer1_weights": model.coefs_[0],
        "layer1_biases": model.intercepts_[0],
        "hidden_activations": hidden_fp,
        "layer2_weights": model.coefs_[1],
        "layer2_biases": model.intercepts_[1],
    }

    tensor_stats = {}
    for name, values in tensors.items():
        values = np.asarray(values, dtype=np.float64)
        maximum = float(np.max(np.abs(values)))
        scale = maximum / 127.0 if maximum > 0 else 1.0
        q = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
        err = np.abs(values - q.astype(np.float64) * scale)
        tensor_stats[name] = {
            "shape": list(values.shape),
            "min": float(values.min()),
            "max": float(values.max()),
            "max_abs": maximum,
            "int8_scale": float(scale),
            "reconstruction_error_max": float(err.max()),
            "reconstruction_error_mean": float(err.mean()),
        }

    def agreement(X_raw, y_true):
        f_log = fp32_logits(X_raw)
        f_pred = (f_log >= 0).astype(np.int64)
        i_log, i_pred, q_in, _ = int8_forward_from_header(X_raw, h)
        err = np.abs(f_log - i_log.astype(np.float64))
        saturated = int(np.sum(np.any(np.abs(q_in) >= 127, axis=1)))
        return {
            "n": int(len(y_true)),
            "fp32_accuracy": float(accuracy_score(y_true, f_pred)),
            "int8_accuracy": float(accuracy_score(y_true, i_pred)),
            "prediction_mismatches": int(np.sum(f_pred != i_pred)),
            "prediction_agreement": float(np.mean(f_pred == i_pred)),
            "logit_abs_error_max": float(err.max()),
            "logit_abs_error_mean": float(err.mean()),
            "logit_abs_error_p95": float(np.percentile(err, 95)),
            "windows_with_saturated_input": saturated,
            "fraction_saturated": float(saturated / len(y_true)),
        }

    # the three windows the QEMU firmware actually runs
    class_ranges = {
        int(label): {
            feat: (float(group[feat].min()), float(group[feat].max()))
            for feat in FEATURES
        }
        for label, group in df.groupby("label")
    }

    demo = {}
    for name, window in FIRMWARE_WINDOWS.items():
        feats = c_features(window)
        fv = np.array([feats], dtype=np.float64)
        f_log = float(fp32_logits(fv)[0])
        i_log, i_pred, q_in, _ = int8_forward_from_header(fv, h)

        expected_label = 1 if "ABNORMAL" in name else 0
        rng = class_ranges[expected_label]
        in_range = all(
            rng[feat][0] <= feats[k] <= rng[feat][1]
            for k, feat in enumerate(FEATURES)
        )

        demo[name] = {
            "window": list(window),
            "features": {
                "mean_absolute": feats[0],
                "rms": feats[1],
                "peak": feats[2],
            },
            "expected_label": expected_label,
            "within_training_range_for_its_class": bool(in_range),
            "int8_quantized_input": q_in[0].tolist(),
            "input_saturated": bool(np.any(np.abs(q_in[0]) >= 127)),
            "fp32_logit": f_log,
            "int8_logit": float(i_log[0]),
            "logit_abs_error": float(abs(f_log - float(i_log[0]))),
            "fp32_prediction": int(f_log >= 0),
            "int8_prediction": int(i_pred[0]),
            "predictions_agree": bool(int(f_log >= 0) == int(i_pred[0])),
        }

    fp32_bytes = (
        model.coefs_[0].size * 4
        + model.intercepts_[0].size * 4
        + model.coefs_[1].size * 4
        + model.intercepts_[1].size * 4
        + scaler.mean_.size * 4
        + scaler.scale_.size * 4
    )
    int8_bytes = (
        h["weights1"].size
        + h["bias1"].size * 4
        + h["weights2"].size
        + h["bias2"].size * 4
        + h["scaler_mean"].size * 4
        + h["scaler_std"].size * 4
    )
    fp32_weight_bytes = (model.coefs_[0].size + model.coefs_[1].size) * 4
    int8_weight_bytes = h["weights1"].size + h["weights2"].size

    return {
        "scheme": "symmetric per-tensor INT8, INT32 biases and accumulators",
        "calibration": {
            "partition": "training split only",
            "n": int(len(X_train)),
            "note": "Test data is never used for calibration.",
        },
        "header_file": repo_rel(h["path"]),
        "header_sha256": sha256(h["path"]),
        "exporter_reproduces_committed_header": header_consistent,
        "integer_tensors_bit_exact": integers_exact,
        "scale_representation_equivalence": scale_equivalence,
        "scales": {
            "input": float(h["input_scale"]),
            "layer1_weight": float(h["w1_scale"]),
            "layer1_accumulator": float(h["acc1_scale"]),
            "hidden_activation": float(h["hidden_scale"]),
            "layer2_weight": float(h["w2_scale"]),
            "layer2_accumulator": float(h["acc2_scale"]),
        },
        "representable_normalized_input_range": {
            "max_abs": float(127 * h["input_scale"]),
            "dataset_max_abs_normalized": float(
                np.abs(scaler.transform(X)).max()
            ),
            "note": (
                "Inputs beyond this range clip at +/-127. The dataset fits "
                "inside it; the SENSOR_ABNORMAL firmware demo window does "
                "not, which is why its INT8 logit is compressed."
            ),
        },
        "tensor_statistics": tensor_stats,
        "fidelity_naive_split_test": agreement(X_test, y_test),
        "fidelity_whole_dataset": agreement(X, y),
        "firmware_demo_windows": demo,
        "parameter_footprint_bytes": {
            "fp32_total": int(fp32_bytes),
            "int8_total": int(int8_bytes),
            "fp32_weights_only": int(fp32_weight_bytes),
            "int8_weights_only": int(int8_weight_bytes),
            "int8_biases_only": int((h["bias1"].size + h["bias2"].size) * 4),
            "scaler_float32": int(
                (h["scaler_mean"].size + h["scaler_std"].size) * 4
            ),
            "reduction_ratio_weights_only": float(
                fp32_weight_bytes / int8_weight_bytes
            ),
            "reduction_ratio_total": float(fp32_bytes / int8_bytes),
            "note": (
                "Biases are INT32 and the StandardScaler stays float32, so "
                "the total footprint does not shrink 4x. Only the weight "
                "tensors do. Both ratios are reported so the weight-only "
                "figure is not mistaken for the whole-model saving."
            ),
        },
    }


# ------------------------------------------------------------------
# 4. provenance
# ------------------------------------------------------------------

def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR, capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_provenance() -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "generator": "ml/run_evaluation.py",
        "git_commit": git_commit(),
        "limitations": [
            "The vibration dataset is SYNTHETIC. No physical accelerometer, "
            "motor or machine was measured at any point.",
            "ARM Cortex-M4 execution is QEMU emulation (machine mps2-an386). "
            "No physical microcontroller was programmed, run or timed.",
            "Reported accuracy reflects the separability of the synthetic "
            "generator, not demonstrated model capability. Simpler baselines "
            "reach the same score.",
            "No wall-clock inference latency or energy figure is included, "
            "because neither can be obtained from emulation without a "
            "calibrated cycle model.",
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "artifact_hashes": {
            "dataset": sha256(DATA_FILE),
            "model_pkl": sha256(MODEL_DIR / "tinyguard_mlp.pkl"),
            "scaler_pkl": sha256(MODEL_DIR / "tinyguard_scaler.pkl"),
            "int8_header": sha256(int8_header["path"]),
            "fp32_header": sha256(fp32_header["path"]),
        },
        "model_trained_with_sklearn": "1.9.1",
        "sklearn_version_matches_training": sklearn.__version__ == "1.9.1",
    }


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> int:
    dataset = build_dataset_summary()
    metrics = build_model_metrics()
    quant = build_quantization()
    prov = build_provenance()

    written = [
        write_json("dataset_summary.json", dataset),
        write_json("model_metrics.json", metrics),
        write_json("quantization.json", quant),
        write_json("provenance.json", prov),
    ]

    index = {
        "schema_version": 1,
        "description": (
            "TinyGuard-AI evaluation artifacts. Generated by "
            "ml/run_evaluation.py. All figures computed at run time."
        ),
        "generated_utc": prov["generated_utc"],
        "files": {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in written
        },
        "headline": {
            "dataset_windows": dataset["total_windows"],
            "data_origin": dataset["data_origin"],
            "fp32_naive_split_accuracy":
                metrics["shipped_model"]["naive_split_test"]["accuracy"],
            "int8_naive_split_accuracy":
                quant["fidelity_naive_split_test"]["int8_accuracy"],
            "fp32_int8_prediction_agreement":
                quant["fidelity_naive_split_test"]["prediction_agreement"],
            "logistic_regression_accuracy":
                metrics["baselines"]["logistic_regression"]["accuracy"],
            "leakage_free_grouped_accuracy":
                metrics["leakage_free_grouped_split"]["test"]["accuracy"],
            "int8_weight_bytes":
                quant["parameter_footprint_bytes"]["int8_weights_only"],
        },
        "limitations": prov["limitations"],
    }
    written.append(write_json("index.json", index))

    print("\nTINYGUARD-AI EVALUATION COMPLETE")
    print("=" * 33)
    print("Data origin:", dataset["data_origin"], "(no physical sensor)")
    print("ARM execution: QEMU emulation only")
    print()
    for p in written:
        print(f"  wrote {repo_rel(p)}  ({p.stat().st_size} bytes)")
    print()
    print("HEADLINE (computed, not hardcoded)")
    print("-" * 33)
    for k, v in index["headline"].items():
        print(f"  {k}: {v}")
    print()
    print("Exporter reproduces committed header:",
          quant["exporter_reproduces_committed_header"])
    print("FP32/INT8 prediction mismatches (test):",
          quant["fidelity_naive_split_test"]["prediction_mismatches"])

    if not quant["exporter_reproduces_committed_header"]:
        print("\nERROR: exporter output no longer matches the committed header.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
