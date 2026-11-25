"""Candidate regression models (Model 1) for ModelDoctor.

Each candidate is a full scikit-learn Pipeline:

    preprocessor (median impute + scale / mode impute + one-hot, unknown -> zeros)
      -> regressor: TransformedTargetRegressor(estimator, transformer=StandardScaler())

Because preprocessing and the target scaler live inside the pipeline, cloning and
fitting per CV fold re-fits them on that fold's training rows only (no leakage).

Why the target is standardized inside the model: it makes every candidate behave the
same for targets of any scale or offset (1e-9, 1e12, negative, ...). XGBoost works in
float32, so a target like 5e12 + small variation would otherwise lose its variation
to rounding. Predictions are returned in the original units.

The set mirrors the classification candidates (same dependencies, one model per
family) with regression counterparts:
- Ridge Regression : linear baseline; Ridge rather than plain least squares because
                     one-hot features are collinear (alpha=1 is a mild, stable penalty).
- Decision Tree    : interpretable, non-linear (depth-limited like the classifier).
- Random Forest    : robust non-linear ensemble.
- XGBoost          : gradient boosting (already a project dependency).
- KNN              : local, non-parametric. Skipped above KNN_MAX_TRAIN_ROWS rows, where
                     brute-force neighbour search makes CV impractically slow.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ml.data.preprocessing import build_preprocessor

REGRESSION_CANDIDATES = ("Ridge Regression", "Decision Tree", "Random Forest", "XGBoost", "KNN")
KNN_MAX_TRAIN_ROWS = 50_000


class Float64TargetScaler(StandardScaler):
    """
    StandardScaler for the target that always un-standardizes in float64.

    XGBoost predicts in float32; inverse-transforming a float32 array keeps float32, which
    cannot represent the variation of a target with a large offset (e.g. 5e12 + small
    changes: float32 spacing there is ~5e5). Casting first keeps full precision.
    """

    def inverse_transform(self, X, copy=None):
        return super().inverse_transform(np.asarray(X, dtype=np.float64), copy=copy)


def _wrap(estimator, numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("regressor", TransformedTargetRegressor(regressor=estimator, transformer=Float64TargetScaler())),
    ])


def get_regression_candidates(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int = 42,
    fast: bool = False,
    n_train_rows: Optional[int] = None,
) -> Dict[str, Pipeline]:
    """Unfitted candidate pipelines, in order of increasing complexity (ties go to the earlier one)."""
    trees = 30 if fast else 200
    boost = 30 if fast else 200
    est = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=trees, min_samples_leaf=2, random_state=random_state, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=boost, max_depth=4, learning_rate=0.1, subsample=0.9,
            objective="reg:squarederror", random_state=random_state, n_jobs=-1,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1),
    }
    if n_train_rows is not None and n_train_rows > KNN_MAX_TRAIN_ROWS:
        est.pop("KNN")
    return {name: _wrap(e, numeric_features, categorical_features) for name, e in est.items()}
