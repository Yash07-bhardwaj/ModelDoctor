"""Unit and integration tests for ReliabilityFeatureExtractor (Task 4)."""

import inspect
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


@pytest.fixture
def fast_dataset_and_models(monkeypatch):
    """
    Load a compact 600-row sample of loans.csv and provide fast candidate models
    (n_estimators=10 for RF/XGB) for testing.
    """
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

    # Fast override for test execution speed
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

    return splits, num_cols, cat_cols


def test_extractor_output_columns_and_no_nans(fast_dataset_and_models):
    """Assert output DataFrame contains all expected features without any NaNs or infinities."""
    splits, num_cols, cat_cols = fast_dataset_and_models

    model1_res = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model="Logistic Regression",
    )

    extractor = ReliabilityFeatureExtractor(
        n_bootstraps=3,
        k_neighbors=5,
        isolation_n_estimators=15,
        random_state=42,
    )
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=model1_res["base_pipeline"],
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    features_df = extractor.transform(splits.X_calib_raw, model1=model1_res["model"])

    expected_cols = [
        "max_prob",
        "entropy",
        "margin",
        "predicted_class",
        "bootstrap_disagreement",
        "bootstrap_std",
        "knn_distance",
        "knn_distance_percentile",
        "isolation_score",
        "missing_count",
        "missing_fraction",
        "out_of_range_count",
        "unseen_category_count",
    ]

    for col in expected_cols:
        assert col in features_df.columns, f"Missing feature: {col}"

    assert len(features_df) == len(splits.X_calib_raw)
    assert not features_df.isna().any().any(), "Reliability features contain NaNs!"
    assert np.isfinite(features_df.values).all(), "Reliability features contain non-finite values!"
    assert np.all(features_df["max_prob"] >= 0.0) and np.all(features_df["max_prob"] <= 1.0)
    assert np.all(features_df["entropy"] >= 0.0) and np.all(features_df["entropy"] <= 1.0)
    assert np.all(features_df["margin"] >= 0.0) and np.all(features_df["margin"] <= 1.0)
    assert np.all(features_df["knn_distance_percentile"] >= 0.0) and np.all(features_df["knn_distance_percentile"] <= 100.0)


def test_transform_has_no_label_parameter_and_invariant_to_labels(fast_dataset_and_models):
    """
    Assert that transform signature has no label/y parameter, and outputs do not change
    regardless of any external label changes.
    """
    sig = inspect.signature(ReliabilityFeatureExtractor.transform)
    forbidden = ["y", "label", "labels", "target"]
    for p in sig.parameters:
        assert p.lower() not in forbidden, f"Forbidden parameter '{p}' in transform signature!"

    splits, num_cols, cat_cols = fast_dataset_and_models

    model1_res = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model="Logistic Regression",
    )

    extractor = ReliabilityFeatureExtractor(n_bootstraps=2, isolation_n_estimators=10)
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=model1_res["base_pipeline"],
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    # Transform without any true labels
    feats1 = extractor.transform(splits.X_calib_raw, model1=model1_res["model"])
    feats2 = extractor.transform(splits.X_calib_raw, model1=model1_res["model"])

    pd.testing.assert_frame_equal(feats1, feats2)


def test_extractor_fit_only_on_train_split(fast_dataset_and_models):
    """Verify that fit signature only accepts train data and only train statistics are stored."""
    sig = inspect.signature(ReliabilityFeatureExtractor.fit)
    forbidden = ["calib", "test", "val"]
    for p in sig.parameters:
        for term in forbidden:
            assert term not in p.lower(), f"Suspicious parameter '{p}' in fit signature!"

    splits, num_cols, cat_cols = fast_dataset_and_models

    model1_res = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model="Decision Tree",
    )

    extractor = ReliabilityFeatureExtractor(n_bootstraps=2, isolation_n_estimators=10)
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=model1_res["base_pipeline"],
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    # Check numeric ranges match train split exactly, not calib or combined
    for col in num_cols:
        expected_min = float(splits.X_train_raw[col].dropna().min())
        expected_max = float(splits.X_train_raw[col].dropna().max())
        assert extractor.train_min_[col] == expected_min
        assert extractor.train_max_[col] == expected_max


def test_works_for_all_candidate_algorithms(fast_dataset_and_models):
    """Loop over all 5 candidate models using force_model and verify feature extraction works for each."""
    splits, num_cols, cat_cols = fast_dataset_and_models
    candidates = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "KNN"]

    for candidate_name in candidates:
        model1_res = select_model(
            X_train_raw=splits.X_train_raw,
            y_train=splits.y_train,
            numeric_features=num_cols,
            categorical_features=cat_cols,
            force_model=candidate_name,
        )

        extractor = ReliabilityFeatureExtractor(
            n_bootstraps=2,
            k_neighbors=3,
            isolation_n_estimators=10,
            random_state=42,
        )
        extractor.fit(
            X_train_raw=splits.X_train_raw,
            base_pipeline=model1_res["base_pipeline"],
            y_train=splits.y_train,
            numeric_features=num_cols,
            categorical_features=cat_cols,
        )

        feats = extractor.transform(splits.X_calib_raw.iloc[:10], model1=model1_res["model"])
        assert len(feats) == 10
        assert not feats.isna().any().any(), f"Extraction failed with NaNs for candidate {candidate_name}"


def test_anomalous_record_gets_higher_counts_and_flags(fast_dataset_and_models):
    """Verify an input with unseen categories, out-of-range numbers, and missing fields gets higher counts and flags."""
    splits, num_cols, cat_cols = fast_dataset_and_models

    model1_res = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model="Logistic Regression",
    )

    extractor = ReliabilityFeatureExtractor(n_bootstraps=2, isolation_n_estimators=10)
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=model1_res["base_pipeline"],
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    # 1. Normal record from calibration set
    normal_record = splits.X_calib_raw.iloc[[0]].copy()

    # 2. Synthetic anomalous record
    anom_record = splits.X_calib_raw.iloc[[0]].copy()
    anom_record["age"] = 350.0  # Out of range numeric
    anom_record["annual_income"] = 5_000_000.0  # Out of range numeric
    anom_record["loan_purpose"] = "space_expedition"  # Unseen category
    anom_record["credit_score"] = np.nan  # Missing value

    test_batch = pd.concat([normal_record, anom_record], ignore_index=True)
    feats_df, flags = extractor.transform_with_flags(test_batch, model1=model1_res["model"])

    norm_feats = feats_df.iloc[0]
    anom_feats = feats_df.iloc[1]

    # Anomalous record must have higher counts
    assert anom_feats["out_of_range_count"] >= 2
    assert anom_feats["unseen_category_count"] >= 1
    assert anom_feats["missing_count"] >= 1

    assert anom_feats["out_of_range_count"] > norm_feats["out_of_range_count"]
    assert anom_feats["unseen_category_count"] > norm_feats["unseen_category_count"]

    # Flags for anomalous record must contain structured reasons
    anom_flags = flags[1]
    flag_types = [f["type"] for f in anom_flags]
    assert "out_of_range" in flag_types
    assert "unseen_category" in flag_types
    assert "missing_feature" in flag_types


def test_single_record_input_and_joblib_roundtrip(fast_dataset_and_models):
    """Verify that single-record input works and joblib save/load round-trip yields identical output."""
    splits, num_cols, cat_cols = fast_dataset_and_models

    model1_res = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model="Decision Tree",
    )

    extractor = ReliabilityFeatureExtractor(
        n_bootstraps=2,
        k_neighbors=3,
        isolation_n_estimators=10,
        random_state=42,
    )
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=model1_res["base_pipeline"],
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    single_row = splits.X_calib_raw.iloc[[0]]
    feats_original = extractor.transform(single_row, model1=model1_res["model"])
    assert len(feats_original) == 1

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        joblib.dump(extractor, tmp_path)
        loaded_extractor = joblib.load(tmp_path)

        feats_loaded = loaded_extractor.transform(single_row, model1=model1_res["model"])
        pd.testing.assert_frame_equal(feats_original, feats_loaded)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
