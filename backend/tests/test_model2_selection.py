"""Focused tests for Model 2 candidate selection against calibrated confidence and for the
error-detection / matched-coverage metric conventions."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.reliability.evaluation import (
    error_detection_metrics,
    error_probability_metrics,
    evaluate_system,
    matched_band_comparison,
)
from ml.reliability.model2 import (
    CONFIDENCE_CANDIDATE,
    PROBABILITY_CANDIDATE,
    positive_class_proba,
    train_model2,
)


def _features(n, rng, extra_signal=None):
    df = pd.DataFrame({
        "max_prob": rng.uniform(0.5, 1.0, n),
        "entropy": rng.uniform(0, 1, n),
        "margin": rng.uniform(0, 1, n),
        "predicted_class": rng.randint(0, 2, n),
        "bootstrap_disagreement": rng.uniform(0, 0.5, n),
        "bootstrap_std": rng.uniform(0, 0.1, n),
        "knn_distance": rng.uniform(0.5, 3.0, n),
        "knn_distance_percentile": rng.uniform(0, 100, n),
        "isolation_score": rng.uniform(0.3, 0.7, n),
        "missing_count": rng.randint(0, 3, n),
        "out_of_range_count": rng.randint(0, 2, n),
        "unseen_category_count": np.zeros(n),
    })
    df["missing_fraction"] = df["missing_count"] / 10.0
    if extra_signal is not None:
        df["knn_distance"] = extra_signal
    return df


def test_noise_features_lose_to_calibrated_confidence():
    """Correctness depends only on max_prob: the extra (pure-noise) features must not be deployed."""
    rng = np.random.RandomState(0)
    feats = _features(1000, rng)
    correct = (rng.uniform(size=1000) < feats["max_prob"]).astype(int)
    res = train_model2(feats, correct, num_classes=2)
    assert res["status"] == "success"
    assert res["winner_name"] == CONFIDENCE_CANDIDATE
    assert res["winner_feature_names"] == ["max_prob"]
    assert res["model2"].feature_names == ["max_prob"]
    assert res["selection_metric"] == "oof_brier"
    briers = {k: v["brier"] for k, v in res["candidates_scores"].items()}
    assert briers[CONFIDENCE_CANDIDATE] <= min(briers.values())
    assert sum(v["selected"] for v in res["candidates_scores"].values()) == 1
    # Binary: no separate probability-signals candidate (margin/entropy are functions of max_prob).
    assert PROBABILITY_CANDIDATE not in res["candidates_scores"]
    # missing_fraction is an exact duplicate of missing_count and is never fed twice.
    assert "missing_fraction" not in res["feature_names"]


def test_informative_extra_feature_beats_calibrated_confidence():
    """When a reliability signal carries real error information, the full-feature model wins."""
    rng = np.random.RandomState(1)
    n = 1000
    dist = rng.uniform(0, 3, n)
    feats = _features(n, rng, extra_signal=dist)
    p_correct = np.clip(feats["max_prob"] - 0.25 * dist, 0.02, 0.98)
    correct = (rng.uniform(size=n) < p_correct).astype(int)
    res = train_model2(feats, correct, num_classes=2)
    assert res["winner_name"] in ("Logistic Regression", "Gradient Boosting")
    assert res["candidates_scores"][res["winner_name"]]["cv_auroc"] > \
        res["candidates_scores"][CONFIDENCE_CANDIDATE]["cv_auroc"]


def test_multiclass_adds_probability_signal_candidate():
    rng = np.random.RandomState(2)
    feats = _features(600, rng)
    correct = (rng.uniform(size=600) < feats["max_prob"]).astype(int)
    res = train_model2(feats, correct, num_classes=3)
    assert res["candidates_scores"][PROBABILITY_CANDIDATE]["features"] == ["max_prob", "margin", "entropy"]


def test_confidence_winner_still_reports_out_of_support_signals():
    """Even when Model 2 uses max_prob only, inputs outside the calibration range of the other
    reliability signals are reported as unsupported (so they are never auto-TRUSTED)."""
    rng = np.random.RandomState(5)
    feats = _features(800, rng)
    correct = (rng.uniform(size=800) < feats["max_prob"]).astype(int)
    det = train_model2(feats, correct, num_classes=2)["model2"]
    assert det.feature_names == ["max_prob"]
    probe = feats.head(2).copy()
    probe.loc[probe.index[0], "knn_distance"] = 1e6
    probe.loc[probe.index[1], "knn_distance"] = -1.0  # safe direction: not a violation
    assert det.support_violations(probe) == [["knn_distance"], []]


def test_trustworthiness_direction_and_saturated_confidence():
    """Higher confidence -> higher P(correct); max_prob = 1.0 (logit = inf) is handled."""
    rng = np.random.RandomState(3)
    feats = _features(800, rng)
    correct = (rng.uniform(size=800) < feats["max_prob"]).astype(int)
    det = train_model2(feats, correct, num_classes=2)["model2"]
    probe = feats.head(3).copy()
    probe["max_prob"] = [0.5, 0.8, 1.0]
    trust = det.predict_trustworthiness(probe)
    assert np.all(np.isfinite(trust))
    assert trust[0] < trust[1] < trust[2]
    # Detector (FunctionTransformer inside) survives a pickle (joblib) round-trip.
    det2 = pickle.loads(pickle.dumps(det))
    np.testing.assert_allclose(det2.predict_trustworthiness(probe), trust)


def test_nan_reliability_feature_is_scored_and_flagged():
    rng = np.random.RandomState(4)
    feats = _features(800, rng)
    correct = (rng.uniform(size=800) < feats["max_prob"]).astype(int)
    det = train_model2(feats, correct, num_classes=2)["model2"]
    probe = feats.head(2).copy()
    probe.loc[probe.index[0], "max_prob"] = np.nan
    trust = det.predict_trustworthiness(probe)
    assert np.all(np.isfinite(trust))
    assert "max_prob" in det.support_violations(probe)[0]


def test_positive_class_proba_uses_classes_not_column_position():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    clf = LogisticRegression().fit(X, np.array([1, 1, 0, 0]))
    p1 = positive_class_proba(clf, X)
    assert p1[0] > p1[-1]  # label 1 = low X
    single = LogisticRegression()
    single.classes_ = np.array([0])
    single.predict_proba = lambda X: np.ones((len(X), 1))
    assert np.all(positive_class_proba(single, X) == 0.0)


# ---------------------------------------------------------------------------
# Metric conventions
# ---------------------------------------------------------------------------

def test_error_detection_undefined_values_are_none():
    correct = np.array([1, 1, 0, 0])
    nothing_flagged = error_detection_metrics(correct, np.zeros(4, dtype=bool))
    assert nothing_flagged["precision"] is None and nothing_flagged["f1"] is None
    assert nothing_flagged["recall"] == 0.0 and nothing_flagged["errors_missed"] == 2
    no_errors = error_detection_metrics(np.ones(4, dtype=int), np.array([1, 0, 0, 0], dtype=bool))
    assert no_errors["recall"] is None and no_errors["precision"] == 0.0
    assert no_errors["positive_class"] == "model1_incorrect"


def test_error_probability_direction():
    correct = np.array([1, 1, 1, 0, 0])
    good = np.array([0.9, 0.8, 0.7, 0.2, 0.1])  # P(correct): errors have low trust
    m = error_probability_metrics(correct, good)
    assert m["auroc"] == 1.0 and m["auprc"] == 1.0
    assert error_probability_metrics(correct, 1 - good)["auroc"] == 0.0
    assert error_probability_metrics(np.ones(5, dtype=int), good)["auroc"] is None


def test_matched_band_comparison_uses_identical_band_sizes():
    correct = np.array([1, 1, 1, 0, 1, 0, 0, 1])
    conf = np.array([0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.55])
    decisions = np.array(["TRUST", "TRUST", "REVIEW", "REVIEW", "TRUST", "REVIEW", "REJECT", "REJECT"])
    res = matched_band_comparison(correct, conf, decisions)
    md = {r["decision"]: r for r in res["modeldoctor"]}
    bl = {r["decision"]: r for r in res["confidence_baseline"]}
    for d in ("TRUST", "REVIEW", "REJECT"):
        assert md[d]["count"] == bl[d]["count"]
    assert bl["TRUST"]["errors"] == 0           # top-3 confidence: 1, 1, 1
    assert bl["REJECT"]["errors"] == 1          # bottom-2 confidence: 0, 1
    assert md["REJECT"]["risk"] == 0.5
    assert res["trust_error_reduction"] == bl["TRUST"]["errors"] - md["TRUST"]["errors"]


def test_evaluate_system_reports_small_error_sample_and_all_trust_fallback():
    n = 50
    correct = np.ones(n, dtype=int)
    correct[:3] = 0
    conf = np.linspace(0.6, 0.99, n)
    res = evaluate_system(correct, conf, conf, ["TRUST"] * n, reliability_mode="confidence_fallback")
    assert res["n_test_errors"] == 3 and res["metrics_reliability_note"]
    assert res["error_detection"]["modeldoctor"]["precision"] is None  # nothing flagged
    assert res["error_probability"]["modeldoctor"] == res["error_probability"]["confidence_baseline"]
    assert res["selective_risk_improvement"] == 0.0


def test_fallback_mode_never_trusts_inputs_outside_calibration_support():
    """Without Model 2 (too few calibration errors) a far out-of-range input must not be auto-TRUSTED."""
    import warnings
    from pathlib import Path

    from ml.data.loader import load_dataset
    from ml.inference import InferenceEngine, save_artifacts
    from ml.pipeline import TrainingConfig, train_modeldoctor

    data = Path(__file__).resolve().parents[2] / "sample_data" / "breast_cancer.csv"
    df = load_dataset(str(data), "diagnosis")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifacts, _ = train_modeldoctor(df, "diagnosis", TrainingConfig(fast=True))
    assert artifacts["reliability_mode"] == "confidence_fallback" and artifacts["model2"] is None
    record = df.drop(columns=["diagnosis"]).iloc[0].to_dict()
    record["worst_fractal_dimension"] = 1e7
    eng = InferenceEngine(artifacts)
    res = eng.predict([record])[0]
    assert res["decision"] != "TRUST"
    assert "UNSUPPORTED_ESTIMATE" in {w["code"] for w in res["warnings"]}
    # Artifacts saved before reliability_support existed still load and score.
    legacy = {k: v for k, v in artifacts.items() if k != "reliability_support"}
    assert InferenceEngine(legacy).predict([record])[0]["decision"] in ("TRUST", "REVIEW", "REJECT")
