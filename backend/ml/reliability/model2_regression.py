"""Model 2 for REGRESSION: an estimator of Model 1's expected absolute error.

RELIABILITY TARGET
    e_scaled = |y - y_hat| / s
where y_hat is Model 1's prediction for a row it was NOT trained on (calibration rows;
Model 1 is fit on train only) and s is a robust scale of the TRAIN targets
(robust_target_scale). Model 2 predicts E[e_scaled | reliability features].

OUTPUT
    estimated_abs_error = s * Model2(x)   (target units, >= 0)
Larger = Model 1's prediction is expected to be further from the truth = LESS reliable.
It is an expected error, NOT a probability and NOT a bound on the error.

Why this target:
- Absolute (not squared) error: same units as the target; a fitted mean of squared error
  is dominated by a few large residuals.
- Not relative/percentage error: |e|/|y| is undefined at y = 0 and explodes near 0.
- Dividing by one TRAIN-derived constant changes no ranking; it makes values unit-free so
  defaults and reports behave the same for targets of any scale or sign.

CANDIDATES (scored on the same calibration CV folds, out-of-fold):
- "Constant (mean error)"  : no-skill reference (calibration mean error).
- "Bootstrap spread"       : linear, non-negative map of bootstrap_std (Model 1's own
                             variability; the regression analogue of "calibrated confidence").
- "Linear (all features)"  : ridge regression on all reliability features.
- "Gradient Boosting"      : small, regularised HistGradientBoostingRegressor.
Winner: lowest out-of-fold MSE of the error estimate (the proper loss for a conditional mean),
ties to the simpler candidate. If the constant wins, no feature carries usable error signal
and Model 2 is NOT deployed (status "no_signal" -> constant fallback).

FALLBACKS (no Model 2 artifact is created):
- "insufficient_data": fewer than MIN_MODEL2_ROWS calibration rows with error targets.
- "constant_error"   : calibration errors (scaled) have ~zero variance.
- "no_signal"        : the constant candidate won (or every other candidate failed).
In all three the estimate is the calibration mean error for every record.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.reliability.model2 import find_support_violations

MIN_MODEL2_ROWS = 50
CONSTANT_ERROR_TOL = 1e-12
CONSTANT_CANDIDATE = "Constant (mean error)"
SPREAD_CANDIDATE = "Bootstrap spread"
FALLBACK_STATUSES = ("insufficient_data", "constant_error", "no_signal")


class TargetScaleError(ValueError):
    """Raised when no positive scale can be derived from the training targets."""


def robust_target_scale(y_train: np.ndarray) -> Tuple[float, str]:
    """
    Positive scale of the TRAIN targets: 1.4826 * MAD, else IQR / 1.349, else std.
    (MAD and IQR can be 0 when most targets share one value, e.g. zero-inflated data.)
    """
    y = np.asarray(y_train, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        raise TargetScaleError("No finite training targets.")
    med = np.median(y)
    for value, name in (
        (1.4826 * np.median(np.abs(y - med)), "mad"),
        ((np.percentile(y, 75) - np.percentile(y, 25)) / 1.349, "iqr"),
        (np.std(y), "std"),
    ):
        if np.isfinite(value) and value > 0:
            return float(value), name
    raise TargetScaleError("Training targets are constant; no error scale can be derived.")


def scaled_abs_error(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> np.ndarray:
    """|y - y_hat| / s. Safe for zero / negative / tiny / huge targets (no division by y)."""
    return np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) / float(scale)


def _finite_frame(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(float).replace([np.inf, -np.inf], np.nan)


class RegressionErrorEstimator:
    """Deployed regression Model 2 (joblib-serializable)."""

    def __init__(self, pipeline, feature_names: List[str], winner_name: str, target_scale: float):
        self.pipeline = pipeline
        self.feature_names = list(feature_names)
        self.winner_name = winner_name
        self.target_scale = float(target_scale)
        self.support_feature_names_: List[str] = []
        self.feature_min_: Dict[str, float] = {}
        self.feature_max_: Dict[str, float] = {}

    def set_support(self, X: pd.DataFrame) -> "RegressionErrorEstimator":
        """Calibration range of EVERY reliability feature (not only the winner's inputs)."""
        Xf = _finite_frame(X)
        self.support_feature_names_ = list(X.columns)
        self.feature_min_ = {c: float(np.nanmin(Xf[c])) for c in X.columns}
        self.feature_max_ = {c: float(np.nanmax(Xf[c])) for c in X.columns}
        return self

    def _prepare(self, features_df: pd.DataFrame) -> pd.DataFrame:
        X = _finite_frame(features_df[self.feature_names])
        lo = pd.Series({c: self.feature_min_.get(c, -np.inf) for c in self.feature_names})
        hi = pd.Series({c: self.feature_max_.get(c, np.inf) for c in self.feature_names})
        X = X.clip(lower=lo, upper=hi, axis=1)
        # Non-finite input: use the riskiest supported value (all signals: higher = riskier).
        # Such rows are also reported by support_violations.
        for c in self.feature_names:
            if X[c].isna().any():
                X[c] = X[c].fillna(hi[c] if np.isfinite(hi[c]) else 0.0)
        return X

    def predict_scaled_error(self, features_df: pd.DataFrame) -> np.ndarray:
        """E[|y - y_hat| / s], clipped to >= 0."""
        out = np.asarray(self.pipeline.predict(self._prepare(features_df)), dtype=float).ravel()
        if not np.all(np.isfinite(out)):
            raise ValueError("Model 2 produced non-finite error estimates.")
        return np.maximum(out, 0.0)

    def predict_error(self, features_df: pd.DataFrame) -> np.ndarray:
        """Expected absolute error in target units (>= 0)."""
        return self.predict_scaled_error(features_df) * self.target_scale

    def support_violations(self, features_df: pd.DataFrame) -> List[List[str]]:
        cols = [c for c in self.support_feature_names_ if c in features_df.columns]
        return find_support_violations(features_df, cols, self.feature_min_, self.feature_max_)


def error_estimate_metrics(observed: np.ndarray, estimated: np.ndarray) -> Dict[str, Optional[float]]:
    """How well estimated errors match observed errors (same units)."""
    obs = np.asarray(observed, dtype=float)
    est = np.asarray(estimated, dtype=float)
    if len(obs) < 2:
        return {"mse": None, "mae": None, "spearman": None, "mean_estimated": None, "mean_observed": None}
    spear = None
    def varies(v):
        return np.std(v) > 1e-12 * (np.abs(np.mean(v)) + 1e-12)

    if varies(obs) and varies(est):
        spear = float(spearmanr(obs, est).correlation)
    return {
        "mse": float(np.mean((est - obs) ** 2)),
        "mae": float(np.mean(np.abs(est - obs))),
        "spearman": spear,
        "mean_estimated": float(np.mean(est)),
        "mean_observed": float(np.mean(obs)),
    }


def _candidates(feature_cols: List[str], random_state: int) -> List[Tuple[str, Any, List[str]]]:
    spread = ["bootstrap_std"] if "bootstrap_std" in feature_cols else feature_cols[:1]
    return [
        (CONSTANT_CANDIDATE, DummyRegressor(strategy="mean"), spread),
        (SPREAD_CANDIDATE, Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", LinearRegression(positive=True)),  # more spread never means less error
        ]), spread),
        ("Linear (all features)", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]), feature_cols),
        ("Gradient Boosting", HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1.0, random_state=random_state,
        ), feature_cols),
    ]


def _fallback(status: str, message: str, target: np.ndarray, n: int, **extra) -> Dict[str, Any]:
    finite = target[np.isfinite(target)] if len(target) else target
    mean_err = float(np.mean(finite)) if len(finite) else None
    return {
        "status": status,
        "model2": None,
        "winner_name": None,
        "message": message,
        "fallback_scaled_error": mean_err,
        "oof_scaled_error": np.full(n, mean_err if mean_err is not None else np.nan),
        "candidates_scores": extra.get("candidates_scores", {}),
        "n_rows": n,
        "feature_names": extra.get("feature_names", []),
    }


def train_regression_model2(
    features_df: pd.DataFrame,
    scaled_error: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    target_scale: float,
    random_state: int = 42,
    min_rows: int = MIN_MODEL2_ROWS,
) -> Dict[str, Any]:
    """
    Train regression Model 2 on CALIBRATION rows.

    features_df  : reliability features of the calibration rows (no targets in it).
    scaled_error : |y - y_hat| / s for the same rows (Model 1 never trained on them).
    cv_splits    : positional folds over these rows (group / time aware upstream).

    Returns status ("success" or a FALLBACK_STATUSES value), model2 (or None), winner_name,
    oof_scaled_error (winner's out-of-fold estimates; NaN where a row was never validated),
    candidates_scores (out-of-fold metrics per candidate), fallback_scaled_error, message.
    """
    y = np.asarray(scaled_error, dtype=float)
    n = len(y)
    if n != len(features_df):
        raise ValueError("features_df and scaled_error have different lengths.")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("scaled_error must be finite and non-negative.")
    feature_cols = list(features_df.columns)
    if n < min_rows:
        return _fallback("insufficient_data",
                         f"Only {n} calibration rows (< {min_rows}); Model 2 is not trained.", y, n)
    if np.std(y) <= CONSTANT_ERROR_TOL:
        return _fallback("constant_error",
                         "Calibration errors are (near) constant; there is nothing for Model 2 to rank.", y, n)

    X = _finite_frame(features_df)
    scores: Dict[str, Dict[str, Any]] = {}
    oofs: Dict[str, np.ndarray] = {}
    cands = _candidates(feature_cols, random_state)
    for order, (name, est, cols) in enumerate(cands):
        oof = np.full(n, np.nan)
        try:
            for tr, va in cv_splits:
                m = clone(est).fit(X.iloc[tr][cols], y[tr])
                oof[va] = np.maximum(np.asarray(m.predict(X.iloc[va][cols]), dtype=float), 0.0)
            ok = np.isfinite(oof)
            if not ok.any():
                raise ValueError("no out-of-fold predictions")
            met = error_estimate_metrics(y[ok], oof[ok])
            scores[name] = {"status": "success", "order": order, "features": cols,
                            "n_oof": int(ok.sum()), **met}
            oofs[name] = oof
        except Exception as exc:  # a failed candidate never stops the others
            scores[name] = {"status": "failed", "order": order, "features": cols,
                            "error_message": f"{type(exc).__name__}: {exc}"}

    good = [k for k, v in scores.items() if v["status"] == "success" and v["mse"] is not None]
    if not good:
        return _fallback("no_signal", "Every Model 2 candidate failed; using the mean error.", y, n,
                         candidates_scores=scores, feature_names=feature_cols)
    winner = min(good, key=lambda k: (scores[k]["mse"], scores[k]["order"]))
    for k in scores:
        scores[k]["selected"] = k == winner
    if winner == CONSTANT_CANDIDATE:
        return _fallback(
            "no_signal",
            "No reliability feature predicted the error better than the calibration mean error "
            "(out-of-fold MSE); Model 2 is not deployed.",
            y, n, candidates_scores=scores, feature_names=feature_cols,
        )

    _, est, cols = next(c for c in cands if c[0] == winner)
    final = clone(est).fit(X[cols], y)
    model = RegressionErrorEstimator(final, cols, winner, target_scale).set_support(features_df)
    return {
        "status": "success",
        "model2": model,
        "winner_name": winner,
        "message": f"{winner} had the lowest out-of-fold MSE of the error estimate on the calibration split.",
        "fallback_scaled_error": float(np.mean(y)),
        "oof_scaled_error": oofs[winner],
        "candidates_scores": scores,
        "n_rows": n,
        "feature_names": feature_cols,
    }
