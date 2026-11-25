"""Reliability features for REGRESSION (inputs to regression Model 2).

Every feature is computable at prediction time from the raw input record and fitted
artifacts only; the record's own target is never used.

Input-space signals (shared with classification, same code path):
    knn_distance, knn_distance_percentile, isolation_score, missing_count,
    out_of_range_count, unseen_category_count
Output-space signals (regression specific, unitless: divided by the training target scale s):
    bootstrap_std            std of B bootstrap refits' predictions (Model 1 variability)
    knn_target_std           spread of the k nearest TRAIN rows' targets (local noise)
    knn_oof_residual         mean |out-of-fold residual| of the k nearest TRAIN rows
                             (how wrong Model 1 tends to be in this neighbourhood)
    prediction_extrapolation how far Model 1's prediction lies outside the train target range

LEAKAGE RULES:
- fit() uses the TRAIN partition only: inputs, targets, and Model 1's OUT-OF-FOLD
  predictions on train (from model selection CV). In-sample train residuals are never
  used: a model that has fitted a row under-states its error there.
- Neighbours are always TRAIN rows, so calibration / test / inference records are scored
  against data that excludes them.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone

from ml.reliability.features import ReliabilityFeatureExtractor

REGRESSION_FEATURES = [
    "bootstrap_std",
    "knn_target_std",
    "knn_oof_residual",
    "prediction_extrapolation",
    "knn_distance",
    "knn_distance_percentile",
    "isolation_score",
    "missing_count",
    "out_of_range_count",
    "unseen_category_count",
]


class RegressionReliabilityFeatureExtractor(ReliabilityFeatureExtractor):
    """Regression counterpart of ReliabilityFeatureExtractor (reuses its input-space signals)."""

    def __init__(
        self,
        n_bootstraps: int = 10,
        k_neighbors: int = 5,
        k_local: int = 20,
        isolation_n_estimators: int = 100,
        random_state: int = 42,
    ):
        super().__init__(
            n_bootstraps=n_bootstraps,
            k_neighbors=k_neighbors,
            isolation_n_estimators=isolation_n_estimators,
            random_state=random_state,
        )
        self.k_local = k_local
        self.target_scale_: float = 1.0
        self.y_train_: np.ndarray = np.array([])
        self.train_abs_residual_: np.ndarray = np.array([])
        self.global_abs_residual_: float = 0.0
        self.y_min_: float = 0.0
        self.y_max_: float = 0.0

    def fit(  # type: ignore[override]
        self,
        X_train_raw: pd.DataFrame,
        base_pipeline,
        y_train: np.ndarray,
        train_oof_pred: np.ndarray,
        target_scale: float,
        numeric_features: List[str],
        categorical_features: List[str],
    ) -> "RegressionReliabilityFeatureExtractor":
        """Fit on the TRAIN partition only (see module docstring)."""
        y = np.asarray(y_train, dtype=float)
        oof = np.asarray(train_oof_pred, dtype=float)
        if len(y) != len(X_train_raw) or len(oof) != len(y):
            raise ValueError("X_train_raw, y_train and train_oof_pred must have the same length.")
        if not (np.isfinite(target_scale) and target_scale > 0):
            raise ValueError("target_scale must be a positive finite number.")
        resid = np.abs(y - oof)  # NaN where a row never had an out-of-fold prediction
        if not np.isfinite(resid).any():
            raise ValueError("No out-of-fold train predictions available for residual features.")

        self.numeric_features_ = list(numeric_features)
        self.categorical_features_ = list(categorical_features)
        self.feature_names_in_ = list(X_train_raw.columns)
        self.classes_ = np.array([])
        self.num_classes_ = 0
        self._fit_input_space(X_train_raw)

        self.target_scale_ = float(target_scale)
        self.y_train_ = y
        self.y_min_, self.y_max_ = float(y.min()), float(y.max())
        self.train_abs_residual_ = resid
        self.global_abs_residual_ = float(np.nanmean(resid))

        # Plain row bootstrap of the train partition (no classes to stratify on).
        self.bootstrap_models_ = []
        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_bootstraps):
            idx = rng.choice(len(y), size=len(y), replace=True)
            model = clone(base_pipeline)
            model.fit(X_train_raw.iloc[idx], y[idx])
            self.bootstrap_models_.append(model)
        return self

    def transform_with_flags(  # type: ignore[override]
        self, X_raw: pd.DataFrame, model1
    ) -> Tuple[pd.DataFrame, List[List[Dict[str, Any]]]]:
        if not isinstance(X_raw, pd.DataFrame):
            X_raw = pd.DataFrame(X_raw)
        n = len(X_raw)
        if n == 0:
            return pd.DataFrame(columns=REGRESSION_FEATURES), []
        s = self.target_scale_

        pred = np.asarray(model1.predict(X_raw), dtype=float).ravel()

        if self.bootstrap_models_:
            boot = np.vstack([np.asarray(m.predict(X_raw), dtype=float).ravel() for m in self.bootstrap_models_])
            bootstrap_std = boot.std(axis=0) / s
        else:
            bootstrap_std = np.zeros(n)

        inp, flags = self._input_space_signals(X_raw)

        Xp = self.preprocessor_.transform(X_raw)
        if hasattr(Xp, "toarray"):
            Xp = Xp.toarray()
        k = int(min(self.k_local, len(self.y_train_)))
        _, idx = self.knn_.kneighbors(np.asarray(Xp, dtype=np.float64), n_neighbors=k)
        knn_target_std = self.y_train_[idx].std(axis=1) / s
        local_resid = self.train_abs_residual_[idx]
        with np.errstate(invalid="ignore"):
            counts = np.isfinite(local_resid).sum(axis=1)
            sums = np.nansum(local_resid, axis=1)
        knn_oof_residual = np.where(counts > 0, sums / np.maximum(counts, 1), self.global_abs_residual_) / s

        extrap = (np.maximum(0.0, self.y_min_ - pred) + np.maximum(0.0, pred - self.y_max_)) / s
        for i in np.where(extrap > 0)[0]:
            flags[i].append({
                "type": "prediction_outside_training_range",
                "prediction": float(pred[i]),
                "train_min": self.y_min_,
                "train_max": self.y_max_,
            })

        features = pd.DataFrame({
            "bootstrap_std": bootstrap_std,
            "knn_target_std": knn_target_std,
            "knn_oof_residual": knn_oof_residual,
            "prediction_extrapolation": extrap,
            "knn_distance": inp["knn_distance"],
            "knn_distance_percentile": inp["knn_distance_percentile"],
            "isolation_score": inp["isolation_score"],
            "missing_count": inp["missing_count"],
            "out_of_range_count": inp["out_of_range_count"],
            "unseen_category_count": inp["unseen_category_count"],
        }, index=X_raw.index)[REGRESSION_FEATURES]
        features.attrs["flags"] = flags
        self.last_flags_ = flags
        return features, flags

    def _get_feature_names(self) -> List[str]:
        return list(REGRESSION_FEATURES)
