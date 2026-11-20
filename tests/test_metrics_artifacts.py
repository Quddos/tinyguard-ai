"""
Tests for the machine-readable metrics artifacts under results/.

The Streamlit dashboard reads these files, so their schema is a contract. These
tests check that the contract holds, that the figures are internally
consistent, and - importantly - that the scientific limitations and the
baseline comparison survive in the artifacts rather than being quietly dropped.
"""

from __future__ import annotations

import json

import pytest

REQUIRED_FILES = [
    "index.json",
    "dataset_summary.json",
    "model_metrics.json",
    "quantization.json",
    "provenance.json",
]


@pytest.fixture(scope="module")
def artifacts(results_dir):
    missing = [n for n in REQUIRED_FILES if not (results_dir / n).exists()]
    if missing:
        pytest.skip(
            f"metrics artifacts absent ({', '.join(missing)}); "
            "run `python ml/run_evaluation.py` first"
        )
    return {
        name: json.loads((results_dir / name).read_text(encoding="utf-8"))
        for name in REQUIRED_FILES
    }


def test_all_artifacts_present_and_valid_json(artifacts):
    assert set(artifacts) == set(REQUIRED_FILES)
    for name, payload in artifacts.items():
        assert isinstance(payload, dict) and payload, f"{name} is empty"


def test_index_headline_matches_detailed_artifacts(artifacts):
    """The headline block must not drift from the files it summarises."""
    head = artifacts["index.json"]["headline"]
    metrics = artifacts["model_metrics.json"]
    quant = artifacts["quantization.json"]
    dataset = artifacts["dataset_summary.json"]

    assert head["dataset_windows"] == dataset["total_windows"]
    assert head["data_origin"] == dataset["data_origin"]
    assert head["fp32_naive_split_accuracy"] == (
        metrics["shipped_model"]["naive_split_test"]["accuracy"]
    )
    assert head["int8_naive_split_accuracy"] == (
        quant["fidelity_naive_split_test"]["int8_accuracy"]
    )
    assert head["logistic_regression_accuracy"] == (
        metrics["baselines"]["logistic_regression"]["accuracy"]
    )
    assert head["leakage_free_grouped_accuracy"] == (
        metrics["leakage_free_grouped_split"]["test"]["accuracy"]
    )


def test_limitations_are_present_and_mention_both_constraints(artifacts):
    """
    Synthetic data and QEMU emulation must both be stated. This test exists to
    make removing them a failing change.
    """
    for source in ("index.json", "provenance.json"):
        limitations = artifacts[source]["limitations"]
        assert isinstance(limitations, list) and len(limitations) >= 2

        blob = " ".join(limitations).upper()
        assert "SYNTHETIC" in blob, f"{source} does not state the data is synthetic"
        assert "QEMU" in blob, f"{source} does not state ARM runs under QEMU"


def test_dataset_summary_declares_no_physical_sensor(artifacts):
    dataset = artifacts["dataset_summary.json"]
    assert dataset["data_origin"] == "synthetic"
    assert dataset["physical_sensor_used"] is False


def test_baselines_are_reported_alongside_the_model(artifacts):
    """
    The accuracy figure is only interpretable next to the baselines, so the
    artifact must carry them.
    """
    baselines = artifacts["model_metrics.json"]["baselines"]

    for required in (
        "dummy_most_frequent",
        "hand_threshold_rms50_or_peak100",
        "logistic_regression",
        "decision_tree_depth4",
        "mlp_8_shipped_architecture",
    ):
        assert required in baselines, f"missing baseline: {required}"
        assert "accuracy" in baselines[required]

    assert artifacts["model_metrics.json"]["baseline_interpretation"]


def test_dummy_baseline_is_near_chance(artifacts):
    """Sanity check: a balanced binary task gives the dummy about 0.5."""
    acc = artifacts["model_metrics.json"]["baselines"]["dummy_most_frequent"]["accuracy"]
    assert 0.4 <= acc <= 0.6, f"dummy baseline accuracy {acc} is implausible"


def test_leakage_is_quantified_not_hidden(artifacts):
    leak = artifacts["dataset_summary.json"]["naive_split_leakage"]

    assert leak["test_rows"] > 0
    assert 0.0 <= leak["fraction"] <= 1.0
    assert leak["test_rows_whose_vector_appears_in_train"] <= leak["test_rows"]

    grouped = artifacts["model_metrics.json"]["leakage_free_grouped_split"]
    assert grouped["shared_feature_vectors_across_split"] == 0, (
        "the grouped split is supposed to be leakage free"
    )


def test_quantization_fidelity_claims_are_consistent(artifacts):
    quant = artifacts["quantization.json"]

    for block in ("fidelity_naive_split_test", "fidelity_whole_dataset"):
        f = quant[block]
        assert f["prediction_mismatches"] == 0, (
            f"{block} reports {f['prediction_mismatches']} FP32/INT8 mismatches"
        )
        assert f["prediction_agreement"] == 1.0
        assert f["fp32_accuracy"] == f["int8_accuracy"]
        assert f["logit_abs_error_mean"] <= f["logit_abs_error_max"]


def test_exporter_header_consistency_is_asserted(artifacts):
    quant = artifacts["quantization.json"]

    assert quant["exporter_reproduces_committed_header"] is True
    assert quant["integer_tensors_bit_exact"] is True

    eq = quant["scale_representation_equivalence"]
    assert eq["max_scale_ulp_distance"] <= 1
    assert eq["identical_quantized_inputs"] is True
    assert eq["identical_output_accumulators"] is True
    assert eq["identical_predictions"] is True


def test_footprint_reports_both_ratios(artifacts):
    """
    Reporting only the weight-only 4x saving would overstate the result, since
    biases are INT32 and the scaler stays float32.
    """
    fp = artifacts["quantization.json"]["parameter_footprint_bytes"]

    assert fp["int8_weights_only"] < fp["fp32_weights_only"]
    assert fp["reduction_ratio_weights_only"] == pytest.approx(4.0, abs=1e-9)
    assert fp["reduction_ratio_total"] < fp["reduction_ratio_weights_only"], (
        "total reduction cannot exceed the weight-only reduction"
    )


def test_firmware_demo_windows_are_characterised(artifacts):
    demo = artifacts["quantization.json"]["firmware_demo_windows"]

    assert set(demo) == {
        "SENSOR_NORMAL", "SENSOR_ABNORMAL_TYPICAL", "SENSOR_ABNORMAL"
    }

    # the in-distribution windows must not saturate
    assert demo["SENSOR_NORMAL"]["input_saturated"] is False
    assert demo["SENSOR_ABNORMAL_TYPICAL"]["input_saturated"] is False
    assert demo["SENSOR_ABNORMAL_TYPICAL"]["within_training_range_for_its_class"] is True

    # the severe window is knowingly out of range and saturating
    assert demo["SENSOR_ABNORMAL"]["input_saturated"] is True
    assert demo["SENSOR_ABNORMAL"]["within_training_range_for_its_class"] is False

    # every demo window must still agree between FP32 and INT8 on the class
    for name, block in demo.items():
        assert block["predictions_agree"] is True, f"{name} disagrees"

    # saturation should visibly compress the logit, which is the documented point
    assert (
        demo["SENSOR_ABNORMAL"]["logit_abs_error"]
        > demo["SENSOR_ABNORMAL_TYPICAL"]["logit_abs_error"]
    )


def test_provenance_records_environment_and_hashes(artifacts):
    prov = artifacts["provenance.json"]

    for key in ("python", "numpy", "pandas", "scikit_learn", "joblib"):
        assert prov["environment"][key]

    for key in ("dataset", "model_pkl", "scaler_pkl", "int8_header", "fp32_header"):
        digest = prov["artifact_hashes"][key]
        assert isinstance(digest, str) and len(digest) == 64, (
            f"{key} hash is not a sha256 hex digest"
        )


def test_provenance_flags_sklearn_version_mismatch(artifacts):
    """
    The .pkl was written by scikit-learn 1.9.1. The artifact must record
    whether the running version matches, so a reader can judge the figures.
    """
    prov = artifacts["provenance.json"]
    assert prov["model_trained_with_sklearn"] == "1.9.1"
    assert isinstance(prov["sklearn_version_matches_training"], bool)
