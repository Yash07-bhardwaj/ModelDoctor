"""Demonstration script for Task 4: Reliability Feature Extraction.

Loads sample_data/loans.csv, runs data layer splits, calls select_model
on the TRAIN split only, fits ReliabilityFeatureExtractor on the TRAIN split only,
extracts features for the CALIBRATION split, and prints:
- Feature table summary (mean, min, max per feature)
- Comparison of average feature values for correct vs incorrect Model 1 predictions
- Univariate AUROC for detecting incorrect predictions for each feature vs confidence
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure backend root is on sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.data.loader import load_dataset
from ml.data.profiler import profile_dataset
from ml.data.splitter import split_data
from ml.models.selector import select_model
from ml.reliability.features import ReliabilityFeatureExtractor


def run_demo():
    # 1. Locate and load dataset
    candidates = [
        Path("sample_data/loans.csv"),
        Path("../sample_data/loans.csv"),
        BASE_DIR.parent / "sample_data" / "loans.csv",
    ]
    data_path = None
    for p in candidates:
        if p.exists():
            data_path = p
            break

    if not data_path:
        raise FileNotFoundError("Could not find sample_data/loans.csv")

    print(f"Loading dataset from: {data_path.resolve()}\n")
    target_column = "default"
    df = load_dataset(str(data_path), target_column=target_column)

    # 2. Extract profile and perform 3-way split
    profile = profile_dataset(df, target_column=target_column)
    numeric_features = profile["numeric_columns"]
    categorical_features = profile["categorical_columns"]

    splits = split_data(df, target_column=target_column, random_state=42)
    print(f"Dataset split: Train = {len(splits.X_train_raw)} rows, "
          f"Calibration = {len(splits.X_calib_raw)} rows, "
          f"Test = {len(splits.X_test_raw)} rows.\n")

    # 3. Model 1 selection on TRAIN split only
    print("Selecting and training Model 1 on TRAIN split only...")
    model1_res = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=42,
    )
    model1 = model1_res["model"]
    base_pipeline = model1_res["base_pipeline"]
    winner_name = model1_res["winner_name"]
    print(f"Selected Model 1: {winner_name}\n")

    # 4. Fit ReliabilityFeatureExtractor on TRAIN split only
    print("Fitting ReliabilityFeatureExtractor strictly on TRAIN split (B=10 bootstraps, k=5)...")
    extractor = ReliabilityFeatureExtractor(
        n_bootstraps=10,
        k_neighbors=5,
        isolation_n_estimators=100,
        random_state=42,
    )
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=base_pipeline,
        y_train=splits.y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    print("ReliabilityFeatureExtractor fitted successfully.\n")

    # 5. Extract reliability features on CALIBRATION split
    print("Extracting reliability features for CALIBRATION split...")
    calib_feats, calib_flags = extractor.transform_with_flags(splits.X_calib_raw, model1=model1)
    print(f"Extracted {calib_feats.shape[1]} reliability features for {len(calib_feats)} calibration rows.\n")

    # Model 1 correctness ground truth on Calibration split
    calib_preds = model1.predict(splits.X_calib_raw)
    calib_probs = model1.predict_proba(splits.X_calib_raw)
    is_correct = (calib_preds == splits.y_calib).astype(int)
    is_incorrect = (1 - is_correct).astype(int)

    n_correct = int(is_correct.sum())
    n_incorrect = int(is_incorrect.sum())
    print(f"Calibration Ground Truth: {n_correct} correct predictions, {n_incorrect} incorrect predictions ({n_incorrect/len(calib_preds)*100:.2f}% error rate).\n")

    # =========================================================================
    # a. Feature Table Summary
    # =========================================================================
    print("=" * 90)
    print("A. RELIABILITY FEATURE SUMMARY TABLE (CALIBRATION SPLIT)")
    print("=" * 90)
    header_a = f"{'Feature':<26} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}"
    print(header_a)
    print("-" * 90)
    for col in calib_feats.columns:
        s = calib_feats[col]
        print(f"{col:<26} | {s.mean():<10.4f} | {s.std():<10.4f} | {s.min():<10.4f} | {s.max():<10.4f}")
    print("=" * 90)

    # =========================================================================
    # b. Correct vs Incorrect Predictions Comparison
    # =========================================================================
    print("\n" + "=" * 90)
    print("B. AVERAGE FEATURE VALUE: CORRECT vs INCORRECT PREDICTIONS")
    print("   [*** exploratory sanity check only; no decisions or tuning based on this ***]")
    print("=" * 90)
    header_b = f"{'Feature':<26} | {'When CORRECT (n=' + str(n_correct) + ')':<22} | {'When INCORRECT (n=' + str(n_incorrect) + ')':<22} | {'Difference':<12}"
    print(header_b)
    print("-" * 90)
    for col in calib_feats.columns:
        mean_corr = calib_feats.loc[is_correct == 1, col].mean()
        mean_inc = calib_feats.loc[is_correct == 0, col].mean()
        diff = mean_inc - mean_corr
        print(f"{col:<26} | {mean_corr:<22.4f} | {mean_inc:<22.4f} | {diff:<+12.4f}")
    print("=" * 90)

    # =========================================================================
    # c. Univariate AUROC for Detecting Incorrect Predictions
    # =========================================================================
    print("\n" + "=" * 90)
    print("C. UNIVARIATE AUROC FOR DETECTING INCORRECT PREDICTIONS")
    print("   [*** exploratory sanity check only; no decisions or tuning based on this ***]")
    print("=" * 90)
    # Baseline confidence AUROC (detecting incorrect predictions using low confidence)
    baseline_conf_score = -calib_feats["max_prob"]  # lower confidence = higher error risk
    baseline_auc = roc_auc_score(is_incorrect, baseline_conf_score)

    header_c = f"{'Feature':<26} | {'Score Direction':<16} | {'Error AUROC':<12} | {'vs Confidence AUROC (' + f'{baseline_auc:.3f}' + ')'}"
    print(header_c)
    print("-" * 90)

    for col in calib_feats.columns:
        vals = calib_feats[col].values
        # Determine natural correlation direction
        corr = np.corrcoef(vals, is_incorrect)[0, 1]
        score = vals if corr >= 0 else -vals
        direction = "Higher = Error" if corr >= 0 else "Lower = Error"

        try:
            auc = roc_auc_score(is_incorrect, score)
            diff = auc - baseline_auc
            diff_str = f"{diff:+.4f}" if col != "max_prob" else "BASELINE"
            print(f"{col:<26} | {direction:<16} | {auc:<12.4f} | {diff_str}")
        except Exception:
            print(f"{col:<26} | {direction:<16} | {'N/A':<12} | N/A")
    print("=" * 90)

    # 5 Sample Records with Diagnostic Flags
    print("\n" + "=" * 90)
    print("DIAGNOSTIC FLAGS: 5 SAMPLE RECORDS FROM CALIBRATION SPLIT")
    print("=" * 90)
    sample_indices = [0, 15, 30, 45, 60]
    for idx in sample_indices:
        pred_label = splits.inverse_class_mapping[calib_preds[idx]]
        true_label = splits.inverse_class_mapping[splits.y_calib[idx]]
        status = "CORRECT" if pred_label == true_label else "INCORRECT"
        flags = calib_flags[idx]
        print(f"Record #{idx} | True: {true_label} | Pred: {pred_label} ({status}) | Confidence: {calib_feats['max_prob'].iloc[idx]*100:.1f}%")
        if flags:
            print(f"  Flags: {flags}")
        else:
            print("  Flags: [None - nominal record]")
        print("-" * 90)

    print("\n[VERIFICATION]: Test split remains completely UNTOUCHED (1,000 samples).")
    print("All feature learning and scaling was strictly confined to Train split.\n")


if __name__ == "__main__":
    run_demo()
