"""Stratified 3-way dataset splitting module for ModelDoctor."""

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class DataSplits:
    """
    Container for raw data splits, encoded targets, and label metadata.
    Preserves original index on raw DataFrames.
    """
    X_train_raw: pd.DataFrame
    X_calib_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: np.ndarray
    y_calib: np.ndarray
    y_test: np.ndarray
    target_encoder: LabelEncoder
    class_mapping: Dict[str, int]
    inverse_class_mapping: Dict[int, str]
    classes: List[str]
    feature_names: List[str]
    target_column: str

    def get_split_summary(self) -> Dict[str, Any]:
        """Return counts and class distributions across all three splits."""
        total = len(self.X_train_raw) + len(self.X_calib_raw) + len(self.X_test_raw)
        return {
            "total_rows": total,
            "train": {
                "rows": len(self.X_train_raw),
                "percentage": round(len(self.X_train_raw) / total * 100.0, 1),
                "class_counts": {
                    self.classes[c]: int((self.y_train == c).sum())
                    for c in range(len(self.classes))
                },
            },
            "calibration": {
                "rows": len(self.X_calib_raw),
                "percentage": round(len(self.X_calib_raw) / total * 100.0, 1),
                "class_counts": {
                    self.classes[c]: int((self.y_calib == c).sum())
                    for c in range(len(self.classes))
                },
            },
            "test": {
                "rows": len(self.X_test_raw),
                "percentage": round(len(self.X_test_raw) / total * 100.0, 1),
                "class_counts": {
                    self.classes[c]: int((self.y_test == c).sum())
                    for c in range(len(self.classes))
                },
            },
        }


def split_data(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
) -> DataSplits:
    """
    Perform a strict 60/20/20 stratified split into Train, Calibration,
    and Test sets.
    
    DataFrames returned are RAW (untransformed) with original indices preserved.
    The target column is completely removed from feature matrices.

    Parameters
    ----------
    df : pd.DataFrame
        Validated input DataFrame.
    target_column : str
        Target column name.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    DataSplits
        Dataclass containing raw feature splits and encoded targets.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # 1. Separate raw features and target
    X_raw = df.drop(columns=[target_column]).copy()
    y_raw = df[target_column].copy()

    # 2. Fit label encoder on target column
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw.astype(str))
    classes = [str(cls) for cls in encoder.classes_]
    class_mapping = {cls: idx for idx, cls in enumerate(classes)}
    inverse_class_mapping = {idx: cls for idx, cls in enumerate(classes)}

    # 3. First split: 60% Train, 40% Temp (Calibration + Test)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw,
        y_encoded,
        train_size=0.60,
        stratify=y_encoded,
        random_state=random_state,
    )

    # 4. Second split: Split Temp 50/50 -> 20% Calibration, 20% Test of total
    X_calib_raw, X_test_raw, y_calib, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return DataSplits(
        X_train_raw=X_train_raw,
        X_calib_raw=X_calib_raw,
        X_test_raw=X_test_raw,
        y_train=np.asarray(y_train),
        y_calib=np.asarray(y_calib),
        y_test=np.asarray(y_test),
        target_encoder=encoder,
        class_mapping=class_mapping,
        inverse_class_mapping=inverse_class_mapping,
        classes=classes,
        feature_names=list(X_raw.columns),
        target_column=target_column,
    )
