"""Unit tests for Model 1 candidate selection, calibration, and leakage prevention."""

import inspect
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

import ml.models.candidates
import ml.models.selector
from ml.data.loader import load_dataset
from ml.data.profiler import profile_dataset
from ml.data.splitter import split_data
from ml.models.selector import select_model, evaluate_candidate_cv


@pytest.fixture(autouse=True)
def fast_models_for_tests(monkeypatch):
    """Test-only override: use fewer trees (e.g. n_estimators=20) for RF and XGBoost to speed up test execution."""
    orig_get_candidates = ml.models.candidates.get_candidate_models

    def _fast_get_candidates(*args, **kwargs):
        candidates = orig_get_candidates(*args, **kwargs)
        if "Random Forest" in candidates:
            candidates["Random Forest"].named_steps["classifier"].set_params(n_estimators=20)
        if "XGBoost" in candidates:
            candidates["XGBoost"].named_steps["classifier"].set_params(n_estimators=20)
        return candidates

    monkeypatch.setattr(ml.models.candidates, "get_candidate_models", _fast_get_candidates)
    monkeypatch.setattr(ml.models.selector, "get_candidate_models", _fast_get_candidates)


@pytest.fixture
def synthetic_train_data():
    """Load a small sample (about 600 rows) from sample_data/loans.csv."""
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
    return splits, num_cols, cat_cols


def test_leaderboard_contains_all_candidates_and_json_serializable(synthetic_train_data):
    """Verify leaderboard has all candidate models and is strictly JSON serializable."""
    splits, num_cols, cat_cols = synthetic_train_data
    result = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    leaderboard = result["leaderboard"]
    model_names = [entry["model"] for entry in leaderboard]
    expected_models = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "KNN"]

    for expected in expected_models:
        assert expected in model_names, f"Missing candidate: {expected}"

    # Must be valid JSON
    json_str = json.dumps(leaderboard)
    assert len(json_str) > 0

    # Sorted descending by primary metric
    metric = result["primary_metric"]
    metric_values = [entry[metric] for entry in leaderboard]
    assert metric_values == sorted(metric_values, reverse=True)


def test_same_folds_used_for_all_candidates(synthetic_train_data):
    """Verify that every candidate is evaluated on the exact same cross-validation fold splits."""
    splits, num_cols, cat_cols = synthetic_train_data

    recorded_folds = []

    def spy_evaluate_candidate_cv(*args, **kwargs):
        cv_splits = kwargs.get("cv_splits", args[4] if len(args) > 4 else None)
        recorded_folds.append(cv_splits)
        return evaluate_candidate_cv(*args, **kwargs)

    with patch("ml.models.selector.evaluate_candidate_cv", side_effect=spy_evaluate_candidate_cv):
        select_model(
            X_train_raw=splits.X_train_raw,
            y_train=splits.y_train,
            numeric_features=num_cols,
            categorical_features=cat_cols,
        )

    assert len(recorded_folds) >= 5, "Did not record folds for all candidates"
    # Compare fold 0 across all candidates
    first_candidate_folds = recorded_folds[0]
    for candidate_folds in recorded_folds[1:]:
        assert len(candidate_folds) == len(first_candidate_folds)
        for fold_idx in range(len(first_candidate_folds)):
            # Compare train indices
            assert np.array_equal(
                first_candidate_folds[fold_idx][0],
                candidate_folds[fold_idx][0],
            ), f"Train fold indices mismatch in fold {fold_idx}"
            # Compare validation indices
            assert np.array_equal(
                first_candidate_folds[fold_idx][1],
                candidate_folds[fold_idx][1],
            ), f"Validation fold indices mismatch in fold {fold_idx}"


def test_calibrated_model_predict_proba(synthetic_train_data):
    """Assert predict_proba returns values in [0, 1] that sum to 1.0 with correct column shape."""
    splits, num_cols, cat_cols = synthetic_train_data
    result = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    model = result["model"]
    probs = model.predict_proba(splits.X_calib_raw)

    assert probs.shape == (len(splits.X_calib_raw), 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_predicted_labels_map_back_to_class_names(synthetic_train_data):
    """Verify integer predictions map back accurately to original string class names."""
    splits, num_cols, cat_cols = synthetic_train_data
    result = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    model = result["model"]
    preds = model.predict(splits.X_calib_raw)

    mapped_labels = [splits.inverse_class_mapping[p] for p in preds]
    assert all(label in splits.classes for label in mapped_labels)


def test_force_model(synthetic_train_data):
    """Verify force_model works for candidate names, skips other candidates, and raises ValueError for unknown names."""
    splits, num_cols, cat_cols = synthetic_train_data

    # 1. Force Decision Tree (only Decision Tree should be cross-validated)
    res_dt = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model="Decision Tree",
    )
    assert res_dt["winner_name"] == "Decision Tree"
    assert len(res_dt["leaderboard"]) == 1
    assert res_dt["leaderboard"][0]["model"] == "Decision Tree"
    assert res_dt["leaderboard"][0]["status"] == "forced"
    assert res_dt["leaderboard"][0]["forced"] is True

    # 2. Force Logistic Regression (only Logistic Regression should be cross-validated)
    res_lr = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model="Logistic Regression",
    )
    assert res_lr["winner_name"] == "Logistic Regression"
    assert len(res_lr["leaderboard"]) == 1
    assert res_lr["leaderboard"][0]["model"] == "Logistic Regression"
    assert res_lr["leaderboard"][0]["status"] == "forced"
    assert res_lr["leaderboard"][0]["forced"] is True

    # 3. Invalid candidate name raises ValueError
    with pytest.raises(ValueError) as exc_info:
        select_model(
            X_train_raw=splits.X_train_raw,
            y_train=splits.y_train,
            numeric_features=num_cols,
            categorical_features=cat_cols,
            force_model="DeepNeuralNet",
        )
    assert "Unknown candidate 'DeepNeuralNet'" in str(exc_info.value)


def test_failing_candidate_does_not_crash_selection(synthetic_train_data):
    """Verify that a failing model is captured on the leaderboard and does not abort selection."""
    splits, num_cols, cat_cols = synthetic_train_data

    def failing_evaluate(name, pipeline, *args, **kwargs):
        if name == "KNN":
            raise RuntimeError("Simulated crash in KNN")
        return evaluate_candidate_cv(name, pipeline, *args, **kwargs)

    with patch("ml.models.selector.evaluate_candidate_cv", side_effect=failing_evaluate):
        result = select_model(
            X_train_raw=splits.X_train_raw,
            y_train=splits.y_train,
            numeric_features=num_cols,
            categorical_features=cat_cols,
        )

    leaderboard = result["leaderboard"]
    knn_entry = next(item for item in leaderboard if item["model"] == "KNN")
    assert knn_entry["status"] == "failed"
    assert "Simulated crash" in knn_entry["error_message"]
    # Another model should be selected as winner
    assert result["winner_name"] != "KNN"
    assert result["model"] is not None


def test_leakage_prevention_signature_and_indices(synthetic_train_data):
    """
    Assert that:
    1. select_model signature strictly lacks calibration or test arguments.
    2. Zero indices from calibration or test splits appear in the training split.
    """
    params = inspect.signature(select_model).parameters
    forbidden_terms = ["calib", "test", "val"]
    for param_name in params:
        for term in forbidden_terms:
            assert term not in param_name.lower(), f"Suspicious parameter name '{param_name}' in select_model!"

    splits, _, _ = synthetic_train_data
    train_idx = set(splits.X_train_raw.index)
    calib_idx = set(splits.X_calib_raw.index)
    test_idx = set(splits.X_test_raw.index)

    assert len(train_idx & calib_idx) == 0
    assert len(train_idx & test_idx) == 0


def test_multiclass_selection():
    """Verify automatic selection works cleanly on a multiclass dataset (Iris)."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    target_col = "target"

    splits = split_data(df, target_column=target_col, random_state=42)
    numeric_features = [c for c in df.columns if c != target_col]
    categorical_features = []

    result = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    assert result["num_classes"] == 3
    assert result["primary_metric"] == "weighted_f1_mean"
    assert len(result["leaderboard"]) == 5

    # Check predictions on calibration split
    probs = result["model"].predict_proba(splits.X_calib_raw)
    assert probs.shape == (len(splits.X_calib_raw), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_leaderboard_sorting_metrics(synthetic_train_data):
    """Verify leaderboard sorts ascending for log_loss and descending for other metrics."""
    splits, num_cols, cat_cols = synthetic_train_data

    # 1. Metric: log_loss_mean (ascending: lower loss is better)
    res_loss = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        primary_metric="log_loss_mean",
    )
    losses = [item["log_loss_mean"] for item in res_loss["leaderboard"] if item.get("status") == "success"]
    assert losses == sorted(losses), "Leaderboard not sorted ascending for log_loss_mean"

    # 2. Metric: accuracy_mean (descending: higher accuracy is better)
    res_acc = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        primary_metric="accuracy_mean",
    )
    accs = [item["accuracy_mean"] for item in res_acc["leaderboard"] if item.get("status") == "success"]
    assert accs == sorted(accs, reverse=True), "Leaderboard not sorted descending for accuracy_mean"
