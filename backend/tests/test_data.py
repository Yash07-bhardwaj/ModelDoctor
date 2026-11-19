"""Unit tests for the data layer: loader, profiler, splitter, and preprocessing."""

import json
import numpy as np
import pandas as pd
import pytest

from ml.data.loader import (
    load_dataset,
    DatasetValidationError,
    MissingTargetError,
    InsufficientRowsError,
    InvalidTargetError,
    UnsupportedTaskError,
)
from ml.data.profiler import profile_dataset
from ml.data.splitter import split_data, DataSplits
from ml.data.preprocessing import (
    build_preprocessor,
    fit_preprocessor,
    transform_data,
)


from pathlib import Path


@pytest.fixture
def sample_loans_df():
    """Load the generated loans dataset."""
    candidates = [
        Path("sample_data/loans.csv"),
        Path("../sample_data/loans.csv"),
        Path(__file__).resolve().parents[2] / "sample_data" / "loans.csv",
    ]
    for p in candidates:
        if p.exists():
            return load_dataset(str(p), target_column="default")
    raise FileNotFoundError("Could not locate sample_data/loans.csv")


def test_splits_no_index_overlap_and_proportions(sample_loans_df):
    """
    Assert that train (60%), calib (20%), and test (20%) splits:
    1. Have zero index overlap.
    2. Sum to total dataset rows.
    3. Do NOT leak the target column into feature DataFrames.
    """
    total_rows = len(sample_loans_df)
    splits: DataSplits = split_data(sample_loans_df, target_column="default", random_state=42)

    train_idx = set(splits.X_train_raw.index)
    calib_idx = set(splits.X_calib_raw.index)
    test_idx = set(splits.X_test_raw.index)

    # 1. Zero index overlap
    assert len(train_idx & calib_idx) == 0, "Train and Calibration splits share indices!"
    assert len(train_idx & test_idx) == 0, "Train and Test splits share indices!"
    assert len(calib_idx & test_idx) == 0, "Calibration and Test splits share indices!"

    # 2. Split sizes are approx 60% / 20% / 20%
    assert len(splits.X_train_raw) == int(total_rows * 0.60)
    assert len(splits.X_calib_raw) == int(total_rows * 0.20)
    assert len(splits.X_test_raw) == int(total_rows * 0.20)
    assert len(train_idx) + len(calib_idx) + len(test_idx) == total_rows

    # 3. Target column is NOT in raw feature splits
    assert "default" not in splits.X_train_raw.columns
    assert "default" not in splits.X_calib_raw.columns
    assert "default" not in splits.X_test_raw.columns


def test_stratification_preserves_class_proportions(sample_loans_df):
    """Assert that class proportions are preserved across train, calibration, and test."""
    splits: DataSplits = split_data(sample_loans_df, target_column="default", random_state=42)

    overall_rate = float(sample_loans_df["default"].mean())
    train_rate = float(np.mean(splits.y_train))
    calib_rate = float(np.mean(splits.y_calib))
    test_rate = float(np.mean(splits.y_test))

    # All splits should be within 1% of the overall default rate
    assert abs(train_rate - overall_rate) < 0.01
    assert abs(calib_rate - overall_rate) < 0.01
    assert abs(test_rate - overall_rate) < 0.01


def test_preprocessor_fit_on_train_only():
    """
    Verify that ColumnTransformer statistics are derived STRICTLY from the
    train split and not the calibration, test, or combined data.
    """
    # Create controlled dataset with intentional shift between train and test
    rng = np.random.RandomState(42)
    n = 600
    df = pd.DataFrame({
        "num_feat": np.concatenate([np.ones(360) * 10.0, np.ones(240) * 100.0]),
        "cat_feat": ["A"] * 360 + ["B"] * 240,
        "target": ["class_0"] * 300 + ["class_1"] * 300,
    })

    splits = split_data(df, target_column="target", random_state=42)
    num_cols = ["num_feat"]
    cat_cols = ["cat_feat"]

    preprocessor = build_preprocessor(numeric_features=num_cols, categorical_features=cat_cols)
    bundle = fit_preprocessor(
        preprocessor=preprocessor,
        X_train_raw=splits.X_train_raw,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )

    # Check fitted mean from StandardScaler inside numeric pipeline
    scaler = bundle.transformer.named_transformers_["numeric"].named_steps["scaler"]
    train_mean = splits.X_train_raw["num_feat"].mean()
    full_mean = df["num_feat"].mean()

    # The scaler's mean must match the train split, NOT the full dataset
    assert np.isclose(scaler.mean_[0], train_mean)
    assert not np.isclose(scaler.mean_[0], full_mean)

    # Check numeric ranges match train split
    assert bundle.numeric_ranges["num_feat"] == (
        float(splits.X_train_raw["num_feat"].min()),
        float(splits.X_train_raw["num_feat"].max()),
    )


def test_preprocessor_handles_unseen_categories():
    """Verify that handle_unknown='ignore' works when calibration/test has new categories."""
    df_train = pd.DataFrame({
        "color": ["red", "blue", "green"] * 200,
        "score": np.linspace(1, 10, 600),
    })
    df_test = pd.DataFrame({
        "color": ["yellow", "purple", "red"],
        "score": [2.0, 5.0, 8.0],
    })

    num_cols = ["score"]
    cat_cols = ["color"]
    preprocessor = build_preprocessor(num_cols, cat_cols)
    bundle = fit_preprocessor(preprocessor, df_train, num_cols, cat_cols)

    transformed_test = transform_data(bundle, df_test)
    assert transformed_test.shape[0] == 3
    assert not np.isnan(transformed_test).any()


def test_validation_errors():
    """Verify custom exceptions for missing target, row count, task type, and rare classes."""
    # 1. Missing target column
    df_valid = pd.DataFrame({
        "f1": np.random.randn(500),
        "f2": np.random.choice(["a", "b"], size=500),
        "target": np.random.choice([0, 1], size=500),
    })
    with pytest.raises(MissingTargetError) as exc_info:
        load_dataset(df_valid.to_csv(index=False), target_column="nonexistent")
    assert "not found" in str(exc_info.value)

    # 2. Too few rows (<500)
    df_small = df_valid.iloc[:400]
    with pytest.raises(InsufficientRowsError) as exc_info:
        load_dataset(df_small.to_csv(index=False), target_column="target")
    assert "At least 500 rows are required" in str(exc_info.value)

    # 3. Continuous numeric target (Regression rejection)
    df_continuous = pd.DataFrame({
        "f1": np.random.randn(550),
        "target": np.linspace(0.1, 99.9, 550),  # 550 distinct continuous float values
    })
    with pytest.raises(UnsupportedTaskError) as exc_info:
        load_dataset(df_continuous.to_csv(index=False), target_column="target")
    assert "Only classification targets are supported in this version." in str(exc_info.value)

    # 4. Single class target
    df_single = pd.DataFrame({
        "f1": np.random.randn(500),
        "target": [1] * 500,
    })
    with pytest.raises(InvalidTargetError) as exc_info:
        load_dataset(df_single.to_csv(index=False), target_column="target")
    assert "At least 2 distinct classes are required" in str(exc_info.value)

    # 5. Class with fewer than 10 samples
    df_rare = pd.DataFrame({
        "f1": np.random.randn(500),
        "target": [0] * 495 + [1] * 5,  # class 1 has only 5 samples
    })
    with pytest.raises(InvalidTargetError) as exc_info:
        load_dataset(df_rare.to_csv(index=False), target_column="target")
    assert "fewer than 10 samples" in str(exc_info.value)


def test_profiler_output_json_serializable(sample_loans_df):
    """Assert that profile_dataset returns a JSON-serializable dictionary with correct structure."""
    profile = profile_dataset(sample_loans_df, target_column="default")

    # Must serialize without error
    json_str = json.dumps(profile)
    assert len(json_str) > 0

    assert profile["row_count"] == 5000
    assert profile["column_count"] == 11
    assert profile["target_column"] == "default"
    assert profile["detected_task"] == "binary_classification"
    assert "imbalance_ratio" in profile
    assert isinstance(profile["warnings"], list)
    assert len(profile["numeric_columns"]) > 0
    assert len(profile["categorical_columns"]) > 0


def test_profiler_detects_id_and_constant_warnings():
    """Verify profiler flags ID-like and constant columns."""
    df_warn = pd.DataFrame({
        "id_col": [f"ID_{i}" for i in range(500)],  # 100% unique
        "const_col": [1.0] * 500,                    # Constant
        "feat": np.random.randn(500),
        "target": [0] * 250 + [1] * 250,
    })
    profile = profile_dataset(df_warn, target_column="target")
    warning_text = " ".join(profile["warnings"])
    assert "id_col" in warning_text and "ID column" in warning_text
    assert "const_col" in warning_text and "constant" in warning_text.lower()
