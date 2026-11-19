"""Unit and integration tests for Model 2 and Mitigation Engine (Task 5 / Task 5b)."""

import json
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest

from ml.data.loader import load_dataset
from ml.data.profiler import profile_dataset
from ml.data.splitter import split_data
from ml.models.selector import select_model
from ml.reliability.features import ReliabilityFeatureExtractor
from ml.reliability.model2 import (
    train_model2,
    Model2Detector,
    compute_ece,
    compute_brier_score,
)
from ml.reliability.mitigation import (
    tune_mitigation_thresholds,
    generate_warnings,
    MitigationEngine,
)


@pytest.fixture
def fast_calib_setup(monkeypatch):
    """Provide fast calibration split features and models for testing."""
    candidates = [
        Path("sample_data/loans.csv"),
        Path("../sample_data/loans.csv"),
        Path(__file__).resolve().parents[2] / "sample_data" / "loans.csv",
    ]
    data_path = None
    for p in candidates:
        if p.exists():
            data_path = p
            break
    if not data_path:
        raise FileNotFoundError("Could not locate sample_data/loans.csv")

    full_df = load_dataset(str(data_path), target_column="default")
    df_sample = full_df.sample(n=600, random_state=42)

    profile = profile_dataset(df_sample, target_column="default")
    num_cols = profile["numeric_columns"]
    cat_cols = profile["categorical_columns"]

    splits = split_data(df_sample, target_column="default", random_state=42)

    # Fast candidates override
    import ml.models.candidates
    import ml.models.selector

    orig_get_candidates = ml.models.candidates.get_candidate_models

    def _fast_candidates(*args, **kwargs):
        models = orig_get_candidates(*args, **kwargs)
        if "Random Forest" in models:
            models["Random Forest"].named_steps["classifier"].set_params(n_estimators=10)
        if "XGBoost" in models:
            models["XGBoost"].named_steps["classifier"].set_params(n_estimators=10)
        return models

    monkeypatch.setattr(ml.models.candidates, "get_candidate_models", _fast_candidates)
    monkeypatch.setattr(ml.models.selector, "get_candidate_models", _fast_candidates)

    # Train Model 1 (Logistic Regression)
    m1_res = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model="Logistic Regression",
    )

    # Fit extractor
    extractor = ReliabilityFeatureExtractor(n_bootstraps=2, isolation_n_estimators=10)
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=m1_res["base_pipeline"],
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    # Extract calibration features
    calib_feats, calib_flags = extractor.transform_with_flags(splits.X_calib_raw, model1=m1_res["model"])
    calib_preds = m1_res["model"].predict(splits.X_calib_raw)
    y_correct = (calib_preds == splits.y_calib).astype(int)

    return splits, m1_res, extractor, calib_feats, y_correct, num_cols, cat_cols


def test_model2_trained_only_on_calib_features_without_labels(fast_calib_setup):
    """Verify Model 2 is trained only from calibration features and true labels never appear."""
    _, _, _, calib_feats, y_correct, _, _ = fast_calib_setup

    res = train_model2(calib_feats, y_correct, exclude_predicted_class=True)

    # Check status
    assert res["status"] in ["success", "insufficient_signal"]
    feature_cols = res["feature_names"]

    # Assert true labels and predicted_class are NOT in feature names
    forbidden = ["target", "label", "y_true", "y", "default", "predicted_class"]
    for col in forbidden:
        assert col not in feature_cols


def test_oof_probabilities_properties(fast_calib_setup):
    """Verify OOF probabilities match row count, lie in [0, 1], and final model is fitted."""
    _, _, _, calib_feats, y_correct, _, _ = fast_calib_setup

    # Make synthetic balanced calibration features to guarantee enough signals for CV test
    n = 200
    rng = np.random.RandomState(42)
    syn_feats = pd.DataFrame({
        "max_prob": rng.uniform(0.5, 1.0, size=n),
        "entropy": rng.uniform(0.0, 1.0, size=n),
        "margin": rng.uniform(0.0, 1.0, size=n),
        "bootstrap_disagreement": rng.uniform(0.0, 0.5, size=n),
        "bootstrap_std": rng.uniform(0.0, 0.1, size=n),
        "knn_distance": rng.uniform(0.5, 3.0, size=n),
        "knn_distance_percentile": rng.uniform(0.0, 100.0, size=n),
        "isolation_score": rng.uniform(0.3, 0.7, size=n),
        "missing_count": rng.randint(0, 3, size=n),
        "missing_fraction": rng.uniform(0.0, 0.2, size=n),
        "out_of_range_count": rng.randint(0, 2, size=n),
        "unseen_category_count": np.zeros(n),
    })
    syn_correct = np.array([1]*150 + [0]*50)  # 50 incorrect > 30 min required

    res = train_model2(syn_feats, syn_correct, exclude_predicted_class=True)
    assert res["status"] == "success"

    oof = res["oof_trustworthiness"]
    assert len(oof) == n
    assert np.all(oof >= 0.0) and np.all(oof <= 1.0)

    model2 = res["model2"]
    assert isinstance(model2, Model2Detector)

    # Test scoring on new input
    trust = model2.predict_trustworthiness(syn_feats.iloc[:5])
    assert len(trust) == 5
    assert np.all(trust >= 0.0) and np.all(trust <= 1.0)


def test_model2_probability_calibration():
    """
    On synthetic data with a known error rate, mean OOF trustworthiness is within
    0.03 of observed accuracy, and ECE is below 0.05.
    """
    n = 500
    rng = np.random.RandomState(42)
    # Generate informative confidence max_prob
    max_prob = rng.uniform(0.60, 0.98, size=n)
    # True correctness drawn directly from max_prob probability
    y_correct = (rng.uniform(0.0, 1.0, size=n) < max_prob).astype(int)

    syn_feats = pd.DataFrame({
        "max_prob": max_prob,
        "bootstrap_disagreement": rng.uniform(0.0, 0.3, size=n),
        "bootstrap_std": rng.uniform(0.01, 0.05, size=n),
        "knn_distance": rng.uniform(0.5, 2.0, size=n),
        "knn_distance_percentile": rng.uniform(0.0, 100.0, size=n),
        "isolation_score": rng.uniform(0.3, 0.6, size=n),
        "missing_count": np.zeros(n),
        "missing_fraction": np.zeros(n),
        "out_of_range_count": np.zeros(n),
        "unseen_category_count": np.zeros(n),
    })

    res = train_model2(syn_feats, y_correct, random_state=42)
    assert res["status"] == "success"

    oof = res["oof_trustworthiness"]
    obs_acc = float(np.mean(y_correct))
    mean_oof = float(np.mean(oof))

    # Mean OOF trustworthiness within 0.03 of observed accuracy
    assert abs(mean_oof - obs_acc) < 0.03, f"Calibration gap too large: mean OOF={mean_oof:.4f}, obs acc={obs_acc:.4f}"

    # ECE below 0.05
    ece = compute_ece(y_correct, oof, n_bins=10)
    assert ece < 0.05, f"ECE too high: {ece:.4f}"


def test_fallback_warning_for_review_reject_and_none_for_trust():
    """Verify fallback warning for non-TRUST decisions without flags, and empty list for TRUST."""
    # 1. TRUST with no flags -> empty list
    w_trust = generate_warnings([], features={"max_prob": 0.90}, decision="TRUST")
    assert w_trust == []

    # 2. REVIEW with no flags -> LOW_TRUSTWORTHINESS fallback warning
    w_review = generate_warnings([], features={"max_prob": 0.70}, decision="REVIEW")
    assert len(w_review) == 1
    assert w_review[0]["code"] == "LOW_TRUSTWORTHINESS"
    assert "Estimated trustworthiness is below the trust level" in w_review[0]["message"]

    # 3. REJECT with no flags -> LOW_TRUSTWORTHINESS fallback warning
    w_reject = generate_warnings([], features={"max_prob": 0.80}, decision="REJECT")
    assert len(w_reject) == 1
    assert w_reject[0]["code"] == "LOW_TRUSTWORTHINESS"
    assert "Estimated trustworthiness is below the trust level" in w_reject[0]["message"]

    # Ensure "verified" or "standard" is never printed for non-TRUST
    for w in w_review + w_reject:
        msg = w["message"].lower()
        assert "verified" not in msg
        assert "standard" not in msg


def test_data_driven_reject_threshold_and_none_case():
    """Verify data-driven reject threshold calculation and handling of None case."""
    # 1. Clear reject band case:
    # 40 rows below 0.60 with 50% error rate (>= 30% error and >= 30 rows)
    n = 200
    oof = np.concatenate([np.linspace(0.40, 0.58, 40), np.linspace(0.75, 0.95, 160)])
    # 20 errors in the bottom 40 rows (50% error rate)
    y_correct = np.concatenate([np.array([1]*20 + [0]*20), np.ones(160, dtype=int)])

    thresh = tune_mitigation_thresholds(
        oof_trustworthiness=oof,
        y_correct=y_correct,
        target_risk=0.05,
        min_coverage=0.30,
        reject_min_error=0.30,
        reject_min_rows=30,
    )

    assert thresh["trust_threshold"] is not None
    assert thresh["reject_threshold"] is not None
    assert thresh["reject_threshold"] < thresh["trust_threshold"]
    assert 0.50 <= thresh["reject_threshold"] < 0.60

    # 2. None case:
    # All rows have very low error rate (< 10%), so no threshold has error >= 30% with 30 rows
    y_mostly_correct = np.array([1]*190 + [0]*10)
    thresh_none = tune_mitigation_thresholds(
        oof_trustworthiness=oof,
        y_correct=y_mostly_correct,
        target_risk=0.05,
        min_coverage=0.30,
        reject_min_error=0.30,
        reject_min_rows=30,
    )
    assert thresh_none["reject_threshold"] is None
    assert "REJECT band disabled" in thresh_none["message"]

    # 3. MitigationEngine handles reject_threshold=None
    engine = MitigationEngine(
        trust_threshold=0.80,
        reject_threshold=None,
    )
    # Scores below trust_threshold become REVIEW, never REJECT
    assert engine.decide(0.85) == "TRUST"
    assert engine.decide(0.50) == "REVIEW"
    assert engine.decide(0.20) == "REVIEW"


def test_redundant_binary_features_excluded():
    """Verify entropy and margin are excluded by default for binary targets (num_classes=2)."""
    n = 100
    df = pd.DataFrame({
        "max_prob": np.linspace(0.6, 0.9, n),
        "entropy": np.linspace(0.1, 0.8, n),
        "margin": np.linspace(0.2, 0.9, n),
        "knn_distance": np.linspace(1.0, 2.0, n),
    })
    y = np.array([1]*70 + [0]*30)

    # By default, exclude_redundant_binary_features=True
    res_ex = train_model2(df, y, num_classes=2, exclude_redundant_binary_features=True)
    assert "entropy" not in res_ex["feature_names"]
    assert "margin" not in res_ex["feature_names"]
    assert "max_prob" in res_ex["feature_names"]

    # If disabled, both are included
    res_in = train_model2(df, y, num_classes=2, exclude_redundant_binary_features=False)
    assert "entropy" in res_in["feature_names"]
    assert "margin" in res_in["feature_names"]


def test_insufficient_signal_handling():
    """Verify that fewer than 30 incorrect predictions returns insufficient_signal without crashing."""
    n = 100
    features = pd.DataFrame({"max_prob": np.linspace(0.8, 0.99, n)})
    # Only 5 incorrect predictions (< 30)
    y_correct = np.array([1]*95 + [0]*5)

    res = train_model2(features, y_correct)
    assert res["status"] == "insufficient_signal"
    assert res["model2"] is None
    assert "insufficient error signal" in res["message"]
    assert res["n_incorrect"] == 5


def test_mitigation_engine_serialization_and_joblib_roundtrip():
    """Verify MitigationEngine decision logic, JSON serialization, and joblib roundtrip."""
    engine = MitigationEngine(
        trust_threshold=0.80,
        reject_threshold=0.50,
        target_risk=0.05,
        min_coverage=0.30,
        force_review_on_anomalies=True,
    )

    # Decisions
    assert engine.decide(0.85) == "TRUST"
    assert engine.decide(0.65) == "REVIEW"
    assert engine.decide(0.40) == "REJECT"

    # Downgrade on domain anomaly
    anomaly_flags = [{"type": "out_of_range", "feature": "age"}]
    assert engine.decide(0.85, record_flags=anomaly_flags) == "REVIEW"

    # JSON serialization
    engine_dict = engine.to_dict()
    json_str = json.dumps(engine_dict)
    assert len(json_str) > 0

    reconstructed = MitigationEngine.from_dict(engine_dict)
    assert reconstructed.trust_threshold == engine.trust_threshold

    # Joblib round-trip
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        joblib.dump(engine, tmp_path)
        loaded_engine = joblib.load(tmp_path)
        assert loaded_engine.decide(0.85) == "TRUST"
        assert loaded_engine.decide(0.40) == "REJECT"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# =============================================================================
# Restored Tests (Part 1) and New Tests (Parts 2 & 5)
# =============================================================================

def test_thresholds_tuning_known_synthetic_oof():
    """Verify thresholds are chosen from OOF values using synthetic OOF values with a known correct answer."""
    # Build synthetic OOF probabilities and true correctness:
    # Block A: 50 rows, score=0.25, all incorrect (error rate = 100%)
    # Block B: 100 rows, score=0.65, 85 correct, 15 incorrect (risk = 15/200 = 7.5% > 5%)
    # Block C: 100 rows, score=0.85, all 100 correct (selective risk = 0.0% <= 5%, coverage = 100/250 = 40% >= 30%)
    oof = np.concatenate([[0.25]*50, [0.65]*100, [0.85]*100])
    y_correct = np.concatenate([
        np.zeros(50, dtype=int),
        np.array([1]*85 + [0]*15),
        np.ones(100, dtype=int),
    ])

    thresh = tune_mitigation_thresholds(
        oof_trustworthiness=oof,
        y_correct=y_correct,
        target_risk=0.05,
        min_coverage=0.30,
        reject_min_error=0.50,
        reject_min_rows=30,
    )

    assert thresh["target_risk_met"] is True
    # Trust threshold should be exactly 0.85
    assert thresh["trust_threshold"] == 0.85
    assert thresh["calibration_selective_risk"] <= 0.05
    assert thresh["calibration_coverage"] >= 0.30
    # Reject threshold isolates Block A (score < 0.65) where error rate is 100% >= 50%
    assert thresh["reject_threshold"] == 0.65
    assert thresh["reject_threshold"] < thresh["trust_threshold"]


def test_target_risk_not_met_flag_and_message():
    """Verify target_risk_met is False and an informative message is returned when the target is impossible."""
    # Impossible target: random scores with 50% error rate everywhere
    n = 200
    rng = np.random.RandomState(42)
    oof = rng.uniform(0.50, 0.90, size=n)
    y_correct = rng.binomial(1, 0.50, size=n)  # 50% errors everywhere

    thresh = tune_mitigation_thresholds(
        oof_trustworthiness=oof,
        y_correct=y_correct,
        target_risk=0.05,
        min_coverage=0.30,
    )

    assert thresh["target_risk_met"] is False
    assert "could not be achieved" in thresh["message"] or "Defaulting to threshold" in thresh["message"]
    assert thresh["trust_threshold"] is not None


@pytest.mark.parametrize("model_name", [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "KNN",
])
def test_model2_and_mitigation_work_for_all_candidate_models(model_name, monkeypatch):
    """Verify Model 2 and mitigation work when Model 1 is each candidate algorithm."""
    candidates = [
        Path("sample_data/loans.csv"),
        Path("../sample_data/loans.csv"),
        Path(__file__).resolve().parents[2] / "sample_data" / "loans.csv",
    ]
    data_path = None
    for p in candidates:
        if p.exists():
            data_path = p
            break
    if not data_path:
        raise FileNotFoundError("Could not locate sample_data/loans.csv")

    full_df = load_dataset(str(data_path), target_column="default")
    df_sample = full_df.sample(n=600, random_state=42)

    profile = profile_dataset(df_sample, target_column="default")
    num_cols = profile["numeric_columns"]
    cat_cols = profile["categorical_columns"]
    splits = split_data(df_sample, target_column="default", random_state=42)

    import ml.models.candidates
    import ml.models.selector
    orig_get_candidates = ml.models.candidates.get_candidate_models

    def _fast_candidates(*args, **kwargs):
        models = orig_get_candidates(*args, **kwargs)
        if "Random Forest" in models:
            models["Random Forest"].named_steps["classifier"].set_params(n_estimators=10)
        if "XGBoost" in models:
            models["XGBoost"].named_steps["classifier"].set_params(n_estimators=10)
        return models

    monkeypatch.setattr(ml.models.candidates, "get_candidate_models", _fast_candidates)
    monkeypatch.setattr(ml.models.selector, "get_candidate_models", _fast_candidates)

    # 1. Train Model 1 with force_model
    m1_res = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model=model_name,
    )
    assert m1_res["winner_name"] == model_name

    # 2. Extract features
    extractor = ReliabilityFeatureExtractor(n_bootstraps=2, isolation_n_estimators=10)
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=m1_res["base_pipeline"],
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )
    calib_feats, _ = extractor.transform_with_flags(splits.X_calib_raw, model1=m1_res["model"])
    calib_preds = m1_res["model"].predict(splits.X_calib_raw)
    y_correct = (calib_preds == splits.y_calib).astype(int)

    # 3. Model 2 training
    res_m2 = train_model2(calib_feats, y_correct, random_state=42)
    assert res_m2["status"] in ["success", "insufficient_signal"]
    if res_m2["status"] == "success":
        oof = res_m2["oof_trustworthiness"]
        assert len(oof) == len(splits.X_calib_raw)
        # 4. Mitigation thresholds
        thresh = tune_mitigation_thresholds(oof, y_correct)
        assert thresh["trust_threshold"] is not None
        engine = MitigationEngine(
            trust_threshold=thresh["trust_threshold"],
            reject_threshold=thresh["reject_threshold"],
        )
        assert engine.decide(0.99) == "TRUST"


def test_warnings_produced_for_anomalous_record_and_none_for_typical(fast_calib_setup):
    """Verify warnings are produced for an anomalous record and none for a typical one."""
    splits, m1_res, extractor, _, _, num_cols, cat_cols = fast_calib_setup

    engine = MitigationEngine(
        trust_threshold=0.85,
        reject_threshold=0.50,
        force_review_on_anomalies=True,
    )

    # 1. Typical record: locate a training sample with zero anomaly flags and high confidence
    feats_train, flags_train = extractor.transform_with_flags(splits.X_train_raw, model1=m1_res["model"])
    typical_idx = None
    for i, flags in enumerate(flags_train):
        if len(flags) == 0 and feats_train.iloc[i]["max_prob"] > 0.85:
            typical_idx = i
            break

    if typical_idx is None:
        # Fallback to first row without domain anomalies
        typical_idx = 0
        flags_train[typical_idx] = []

    typical_row = splits.X_train_raw.iloc[[typical_idx]].copy()
    rec_flags_typ = flags_train[typical_idx]
    eval_typ = engine.evaluate_record(
        trustworthiness=0.95,
        record_flags=rec_flags_typ,
        features=feats_train.iloc[typical_idx].to_dict(),
    )
    # Typical record with high trust should get TRUST and zero warnings
    assert eval_typ["decision"] == "TRUST"
    assert eval_typ["warnings"] == []

    # 2. Anomalous record: extreme out of range value, unseen category, missing feature, low confidence
    anom_row = typical_row.copy()
    anom_row.iloc[0, anom_row.columns.get_loc(num_cols[0])] = 99999999.0  # Out of range
    if cat_cols:
        anom_row.iloc[0, anom_row.columns.get_loc(cat_cols[0])] = "NON_EXISTENT_CATEGORY_XYZ"
    anom_row.iloc[0, anom_row.columns.get_loc(num_cols[1])] = np.nan  # Missing field

    feats_anom, flags_anom = extractor.transform_with_flags(anom_row, model1=m1_res["model"])
    rec_flags_anom = flags_anom[0]
    eval_anom = engine.evaluate_record(
        trustworthiness=0.45,
        record_flags=rec_flags_anom,
        features={"max_prob": 0.52},
    )
    assert eval_anom["decision"] in ["REVIEW", "REJECT"]
    assert len(eval_anom["warnings"]) >= 2
    warning_codes = [w["code"] for w in eval_anom["warnings"]]
    assert "OUT_OF_RANGE" in warning_codes
    assert "MISSING_FIELD" in warning_codes or "MISSING_FIELDS" in warning_codes


def test_warning_merging_binary_vs_multiclass():
    """Verify HIGH_ENTROPY, LOW_MARGIN, and LOW_CONFIDENCE merge in binary and stay separate in multiclass."""
    # Synthetic flags triggering entropy and margin
    flags = [
        {"type": "high_entropy"},
        {"type": "low_margin"},
    ]
    features = {"max_prob": 0.52}

    # 1. Binary target (num_classes=2): all collapse to a single LOW_CONFIDENCE warning
    w_bin = generate_warnings(flags, features=features, num_classes=2)
    assert len(w_bin) == 1
    assert w_bin[0]["code"] == "LOW_CONFIDENCE"
    assert "52.0%" in w_bin[0]["message"]

    # 2. Multiclass target (num_classes=3): all three remain separate distinct warnings
    w_multi = generate_warnings(flags, features=features, num_classes=3)
    codes_multi = [w["code"] for w in w_multi]
    assert len(codes_multi) == 3
    assert "HIGH_ENTROPY" in codes_multi
    assert "LOW_MARGIN" in codes_multi
    assert "LOW_CONFIDENCE" in codes_multi


def test_decision_band_metadata_structure_and_json_serialization():
    """Verify decision band metadata structure, required fields, disclaimer note, and JSON serialization."""
    oof = np.concatenate([np.linspace(0.80, 0.95, 100), np.linspace(0.40, 0.60, 50)])
    y_correct = np.concatenate([np.ones(100, dtype=int), np.array([1]*25 + [0]*25)])

    thresh = tune_mitigation_thresholds(
        oof_trustworthiness=oof,
        y_correct=y_correct,
        target_risk=0.05,
        min_coverage=0.30,
        reject_min_error=0.30,
        reject_min_rows=30,
    )

    assert "band_metadata" in thresh
    band_meta = thresh["band_metadata"]
    assert len(band_meta) == 3

    band_names = [b["name"] for b in band_meta]
    assert band_names == ["TRUST", "REVIEW", "REJECT"]

    required_keys = ["name", "threshold_range", "count", "share", "observed_error_rate", "estimate_source", "note"]
    for b in band_meta:
        for k in required_keys:
            assert k in b, f"Missing key {k} in band {b['name']}"
        assert b["estimate_source"] == "calibration_oof"
        assert "optimistic" in b["note"]
        assert "Confirm them on held-out data" in b["note"]

    # JSON serializability of band_metadata
    json_str = json.dumps(band_meta)
    assert len(json_str) > 0
    loaded = json.loads(json_str)
    assert len(loaded) == 3

    # Verification through MitigationEngine
    engine = MitigationEngine(
        trust_threshold=thresh["trust_threshold"],
        reject_threshold=thresh["reject_threshold"],
        band_metadata=band_meta,
    )
    engine_dict = engine.to_dict()
    assert "band_metadata" in engine_dict
    assert engine_dict["band_metadata"] == band_meta
    json_engine = json.dumps(engine_dict)
    assert len(json_engine) > 0
