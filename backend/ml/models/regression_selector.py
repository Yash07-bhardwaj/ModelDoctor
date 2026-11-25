"""Model 1 selection for regression.

LEAKAGE RULES:
- Receives ONLY the train partition. Calibration and test data are never passed here.
- Every candidate is scored on the SAME cross-validation folds of the train partition;
  preprocessing and target scaling are re-fit inside every fold (they live in the pipeline).
- The CV scheme follows the split strategy: KFold for IID data, GroupKFold when a group
  column was declared, and forward-chaining (expanding window) folds when a time column
  was declared, so validation rows are never in a training fold's past or group.
- The winner is re-fit on the whole train partition. There is no probability calibration.

SELECTION METRIC (configurable, default "mae"): mean over folds of
- "mae"  : mean absolute error (lower is better). Default because it is in the target's
           units, robust to a few large residuals, and matches the absolute-error target
           planned for Model 2.
- "rmse" : root mean squared error (lower is better); penalises large errors more.
- "r2"   : coefficient of determination (higher is better). Undefined on a fold whose
           target is constant; such folds are skipped (nan-mean) and counted.
Ties keep candidate order (simpler models first).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

from ml.models.regression_candidates import get_regression_candidates

REGRESSION_METRICS = {"mae": "min", "rmse": "min", "r2": "max"}
DEFAULT_REGRESSION_METRIC = "mae"
N_CV_SPLITS = 5


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """MAE, RMSE and R² (R² is None when y_true is constant: the metric is undefined)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    err = y_pred - y_true
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": None if ss_tot == 0.0 else float(1.0 - np.sum(err ** 2) / ss_tot),
    }


def make_regression_cv_splits(
    n_rows: int,
    strategy: str = "random",
    keys: Optional[np.ndarray] = None,
    n_splits: int = N_CV_SPLITS,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Positional (train_idx, val_idx) folds over `n_rows` rows of ONE partition.

    - random: shuffled KFold.
    - group : GroupKFold on `keys` (groups never span train and validation).
    - time  : forward chaining on `keys` (timestamps). Unique times are cut into
              n_splits + 1 consecutive blocks of ~equal row count; fold k trains on
              blocks 0..k and validates on block k+1. Equal timestamps stay in one block.
    """
    if strategy == "random":
        return list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.zeros(n_rows)))
    if keys is None or len(keys) != n_rows:
        raise ValueError(f"CV strategy '{strategy}' needs one group/time key per row.")
    keys = np.asarray(keys)
    if strategy == "group":
        n_groups = len(pd.unique(keys))
        if n_groups < 2:
            raise ValueError(f"Group cross-validation needs at least 2 groups in this partition (found {n_groups}).")
        k = min(n_splits, n_groups)
        return list(GroupKFold(n_splits=k).split(np.zeros(n_rows), groups=keys))
    if strategy == "time":
        order = pd.Series(keys)
        counts = order.value_counts().sort_index()
        cum = counts.cumsum().to_numpy()
        uniq = counts.index.to_numpy()
        n_blocks = min(n_splits + 1, len(uniq))
        if n_blocks < 2:
            raise ValueError("Time cross-validation needs at least 2 distinct time points.")
        # block id per unique time: by cumulative row share
        block_of_time = np.minimum((cum - 1) * n_blocks // n_rows, n_blocks - 1)
        block = pd.Series(block_of_time, index=uniq).reindex(order.to_numpy()).to_numpy()
        folds = []
        for b in range(1, n_blocks):
            tr, va = np.where(block < b)[0], np.where(block == b)[0]
            if len(tr) and len(va):
                folds.append((tr, va))
        if not folds:
            raise ValueError("Could not build forward-chaining folds from the time column.")
        return folds
    raise ValueError(f"Unknown CV strategy '{strategy}'.")


def _evaluate_candidate(
    pipeline, X: pd.DataFrame, y: np.ndarray, cv_splits
) -> Tuple[Dict[str, Any], np.ndarray]:
    fold_scores: Dict[str, List[float]] = {"mae": [], "rmse": [], "r2": []}
    oof = np.full(len(y), np.nan)
    t0 = time.perf_counter()
    for tr, va in cv_splits:
        fold = clone(pipeline)
        fold.fit(X.iloc[tr], y[tr])
        pred = np.asarray(fold.predict(X.iloc[va]), dtype=float)
        if not np.all(np.isfinite(pred)):
            raise ValueError("candidate produced non-finite predictions")
        oof[va] = pred
        sc = regression_scores(y[va], pred)
        for k in fold_scores:
            fold_scores[k].append(np.nan if sc[k] is None else sc[k])
    row: Dict[str, Any] = {"status": "success", "train_seconds": round(time.perf_counter() - t0, 3)}
    for k, vals in fold_scores.items():
        arr = np.asarray(vals, dtype=float)
        ok = arr[np.isfinite(arr)]
        # Unrounded: rounding would tie every candidate for targets on a tiny scale (e.g. 1e-9).
        row[f"{k}_mean"] = float(ok.mean()) if len(ok) else None
        row[f"{k}_std"] = float(ok.std()) if len(ok) else None
    row["r2_undefined_folds"] = int(np.sum(~np.isfinite(np.asarray(fold_scores["r2"], dtype=float))))
    row["error_message"] = None
    return row, oof


def select_regression_model(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    selection_metric: str = DEFAULT_REGRESSION_METRIC,
    force_model: Optional[str] = None,
    random_state: int = 42,
    fast: bool = False,
    cv_strategy: str = "random",
    cv_keys: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Cross-validate regression candidates on the TRAIN partition, pick the best by
    `selection_metric`, and re-fit it on the whole train partition.

    Returns dict with: model (fitted Pipeline), base_pipeline (unfitted clone),
    winner_name, selection_metric, leaderboard (sorted, includes a median-predictor
    reference row), cv_strategy, n_folds, oof_predictions (winner, positional on
    X_train_raw; NaN where a row is never validated, e.g. the first time block).
    """
    metric = str(selection_metric).lower()
    if metric not in REGRESSION_METRICS:
        raise ValueError(f"Unknown regression selection metric '{selection_metric}'. "
                         f"Use one of {sorted(REGRESSION_METRICS)}.")
    y = np.asarray(y_train, dtype=float)
    if len(y) != len(X_train_raw):
        raise ValueError("X_train_raw and y_train have different lengths.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_train contains non-finite values.")
    if np.unique(y).size < 2:
        raise ValueError("y_train is constant; a regression model cannot be selected.")

    candidates = get_regression_candidates(
        numeric_features, categorical_features, random_state=random_state, fast=fast,
        n_train_rows=len(y),
    )
    if force_model is not None and force_model not in candidates:
        raise ValueError(f"Unknown regression candidate '{force_model}'. Available: {list(candidates)}")
    to_eval = {force_model: candidates[force_model]} if force_model else candidates

    cv_splits = make_regression_cv_splits(len(y), cv_strategy, cv_keys, random_state=random_state)

    leaderboard: List[Dict[str, Any]] = []
    oof_by_model: Dict[str, np.ndarray] = {}
    for order, (name, pipe) in enumerate(to_eval.items()):
        try:
            row, oof = _evaluate_candidate(pipe, X_train_raw, y, cv_splits)
            oof_by_model[name] = oof
        except Exception as exc:  # fault tolerant: record and continue
            row = {"status": "failed", "train_seconds": 0.0, "error_message": f"{type(exc).__name__}: {exc}"}
            for k in REGRESSION_METRICS:
                row[f"{k}_mean"] = row[f"{k}_std"] = None
        row.update({"model": name, "order": order, "forced": force_model is not None})
        leaderboard.append(row)

    key = f"{metric}_mean"
    sign = 1.0 if REGRESSION_METRICS[metric] == "min" else -1.0
    ok = [r for r in leaderboard if r["status"] == "success" and r.get(key) is not None]
    if not ok:
        raise RuntimeError("All regression candidates failed during cross-validation.")
    ok.sort(key=lambda r: (sign * r[key], r["order"]))
    winner = ok[0]["model"]
    leaderboard = ok + [r for r in leaderboard if r not in ok]

    # No-skill reference: predict the training-fold median (same folds).
    ref = {"mae": [], "rmse": [], "r2": []}
    for tr, va in cv_splits:
        sc = regression_scores(y[va], np.full(len(va), np.median(y[tr])))
        for k in ref:
            ref[k].append(np.nan if sc[k] is None else sc[k])
    reference = {f"{k}_mean": float(np.nanmean(v)) if np.isfinite(v).any() else None
                 for k, v in ((k, np.asarray(v, dtype=float)) for k, v in ref.items())}

    final = clone(candidates[winner])
    final.fit(X_train_raw, y)
    return {
        "model": final,
        "base_pipeline": clone(candidates[winner]),
        "winner_name": winner,
        "selection_metric": metric,
        "primary_metric": key,
        "leaderboard": leaderboard,
        "median_baseline_cv": reference,
        "cv_strategy": cv_strategy,
        "n_folds": len(cv_splits),
        "oof_predictions": oof_by_model.get(winner),
    }
