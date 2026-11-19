"""Demonstration script for Task 2 (Data Layer).

Loads sample_data/loans.csv and prints:
- Profile report
- Split sizes
- Default rates per split
- Verification of zero index overlap
- List of feature columns (confirming target is excluded)
"""

import json
from pathlib import Path
import numpy as np

# Adjust python path if necessary to find backend modules
import sys
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.data.loader import load_dataset
from ml.data.profiler import profile_dataset
from ml.data.splitter import split_data


def run_demo():
    # 1. Locate dataset
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

    # 2. Profile dataset
    print("=" * 60)
    print("1. DATASET PROFILE")
    print("=" * 60)
    profile = profile_dataset(df, target_column=target_column)
    print(json.dumps(profile, indent=2))
    print()

    # 3. Stratified 3-way split
    splits = split_data(df, target_column=target_column, random_state=42)

    train_n = len(splits.X_train_raw)
    calib_n = len(splits.X_calib_raw)
    test_n = len(splits.X_test_raw)
    total_n = train_n + calib_n + test_n

    print("=" * 60)
    print("2. SPLIT SIZES")
    print("=" * 60)
    print(f"Total Rows:        {total_n}")
    print(f"Train Split:       {train_n} rows ({train_n / total_n * 100:.1f}%)")
    print(f"Calibration Split: {calib_n} rows ({calib_n / total_n * 100:.1f}%)")
    print(f"Test Split:        {test_n} rows ({test_n / total_n * 100:.1f}%)\n")

    # 4. Default rate per split
    rate_train = float(np.mean(splits.y_train)) * 100
    rate_calib = float(np.mean(splits.y_calib)) * 100
    rate_test = float(np.mean(splits.y_test)) * 100
    rate_overall = float(df[target_column].mean()) * 100

    print("=" * 60)
    print("3. DEFAULT RATE IN EACH SPLIT (STRATIFICATION)")
    print("=" * 60)
    print(f"Overall Dataset:   {rate_overall:.2f}%")
    print(f"Train Split:       {rate_train:.2f}%")
    print(f"Calibration Split: {rate_calib:.2f}%")
    print(f"Test Split:        {rate_test:.2f}%\n")

    # 5. Index overlap check
    train_idx = set(splits.X_train_raw.index)
    calib_idx = set(splits.X_calib_raw.index)
    test_idx = set(splits.X_test_raw.index)

    overlap_train_calib = len(train_idx & calib_idx)
    overlap_train_test = len(train_idx & test_idx)
    overlap_calib_test = len(calib_idx & test_idx)
    total_overlap = overlap_train_calib + overlap_train_test + overlap_calib_test

    print("=" * 60)
    print("4. INDEX OVERLAP VERIFICATION")
    print("=" * 60)
    print(f"Shared indices between Train & Calibration: {overlap_train_calib}")
    print(f"Shared indices between Train & Test:        {overlap_train_test}")
    print(f"Shared indices between Calibration & Test:  {overlap_calib_test}")
    if total_overlap == 0:
        print("RESULT: PASS - No index appears in more than one split (Zero data leakage).\n")
    else:
        print("RESULT: FAIL - Index overlap detected!\n")

    # 6. Feature columns inspection
    feature_cols = splits.feature_names
    target_in_features = target_column in feature_cols or target_column in splits.X_train_raw.columns

    print("=" * 60)
    print("5. FEATURE COLUMNS (EXCLUDING TARGET)")
    print("=" * 60)
    print(f"Feature count: {len(feature_cols)}")
    print(f"Features:      {feature_cols}")
    print(f"Target column '{target_column}' in feature list? {'YES (LEAKAGE ERROR!)' if target_in_features else 'NO (CONFIRMED EXCLUDED)'}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
