"""Feature preprocessing pipelines for ModelDoctor.

Rules:
- Numeric features: median imputation + standard scaling.
- Categorical features: most-frequent imputation + one-hot encoding (handle_unknown='ignore').
- Must be FIT STRICTLY ON TRAIN SPLIT ONLY, then used to transform calibration and test data.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessorBundle:
    """
    Encapsulates a fitted ColumnTransformer and raw training feature metadata
    (ranges and categories) needed for downstream reliability features.
    """
    transformer: ColumnTransformer
    numeric_features: List[str]
    categorical_features: List[str]
    numeric_ranges: Dict[str, Tuple[float, float]]  # (min, max) on train split
    feature_names_out: List[str]


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Construct an unfitted ColumnTransformer pipeline.

    Parameters
    ----------
    numeric_features : list[str]
        List of numeric column names.
    categorical_features : list[str]
        List of categorical column names.

    Returns
    -------
    ColumnTransformer
        Configured scikit-learn ColumnTransformer.
    """
    transformers = []

    if numeric_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", cat_pipeline, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train_raw: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> PreprocessorBundle:
    """
    Strictly fit the preprocessor on the TRAIN SPLIT ONLY and extract
    training feature ranges for raw input out-of-range checks.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Unfitted preprocessor pipeline.
    X_train_raw : pd.DataFrame
        Raw training features (never calibration or test data).
    numeric_features : list[str]
        List of numeric column names.
    categorical_features : list[str]
        List of categorical column names.

    Returns
    -------
    PreprocessorBundle
        Fitted transformer bundle with train metadata.
    """
    # Fit the pipeline on training data only
    preprocessor.fit(X_train_raw)

    # Compute raw numeric min and max strictly on train split
    numeric_ranges: Dict[str, Tuple[float, float]] = {}
    for col in numeric_features:
        if col in X_train_raw.columns:
            series = X_train_raw[col].dropna()
            if len(series) > 0:
                numeric_ranges[col] = (float(series.min()), float(series.max()))
            else:
                numeric_ranges[col] = (0.0, 0.0)

    # Extract output feature names
    try:
        feature_names_out = list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out fails
        feature_names_out = [f"feature_{i}" for i in range(preprocessor.transform(X_train_raw[:1]).shape[1])]

    return PreprocessorBundle(
        transformer=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_ranges=numeric_ranges,
        feature_names_out=feature_names_out,
    )


def transform_data(bundle: PreprocessorBundle, X_raw: pd.DataFrame) -> np.ndarray:
    """
    Transform raw data using the fitted preprocessor.

    Parameters
    ----------
    bundle : PreprocessorBundle
        The fitted preprocessor bundle.
    X_raw : pd.DataFrame
        Raw features DataFrame (train, calibration, test, or new inference data).

    Returns
    -------
    np.ndarray
        Transformed 2D numpy array of preprocessed features.
    """
    transformed = bundle.transformer.transform(X_raw)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=np.float64)
