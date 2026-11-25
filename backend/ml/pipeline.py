"""End-to-end ModelDoctor training and inference orchestration.

Phase A (train_modeldoctor):
    profiling -> splitting -> model_selection -> reliability_features ->
    reliability_training -> evaluation  (saving is done by the caller)

Phase B (predict_records):
    validate raw record(s) -> Model 1 prediction + confidence -> reliability features
    -> Model 2 trustworthiness -> TRUST / REVIEW / REJECT + warnings

LEAKAGE RULES (see project_spec.md section 4/5):
- Model 1, its preprocessing, its probability calibration and the reliability
  feature extractor are fit on the TRAIN split only.
- Model 2 and the mitigation thresholds use the CALIBRATION split only
  (thresholds from out-of-fold Model 2 predictions).
- The TEST split is touched exactly once, in the evaluation stage, after
  everything has been fixed.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import uuid

import numpy as np
import pandas as pd

from ml.data.loader import DatasetValidationError
from ml.data.profiler import profile_dataset
from ml.data.splitter import RegressionSplits, split_data, split_regression_data
from ml.models.regression_selector import (
    DEFAULT_REGRESSION_METRIC,
    make_regression_cv_splits,
    regression_scores,
    select_regression_model,
)
from ml.models.selector import select_model
from ml.reliability.evaluation import evaluate_system
from ml.reliability.features import ReliabilityFeatureExtractor
from ml.reliability.mitigation import MitigationEngine, tune_mitigation_thresholds
from ml.reliability.model2 import ReliabilitySupport, compute_reliability_table, train_model2
from ml.reliability.model2_regression import (
    error_estimate_metrics,
    robust_target_scale,
    scaled_abs_error,
    train_regression_model2,
)
from ml.reliability.regression_features import RegressionReliabilityFeatureExtractor
from ml.reliability.audit_regression import leakage_audit, reproducibility_record
from ml.reliability.evaluation_regression import (
    error_estimation_metrics,
    evaluate_regression_mitigation,
    model1_metrics,
)
from ml.reliability.mitigation_regression import (
    DEFAULT_TOLERANCE_QUANTILE,
    RegressionMitigationEngine,
    resolve_error_tolerance,
    tune_regression_thresholds,
)
from ml.reliability.monitoring import ANOMALY_CODES
from ml.task import CLASSIFICATION, REGRESSION, resolve_task_type

STAGES = [
    "profiling",
    "splitting",
    "model_selection",
    "reliability_features",
    "reliability_training",
    "evaluation",
    "saving",
]

MAX_FORM_CATEGORIES = 200


class RecordValidationError(ValueError):
    """Raised when an inference record does not match the trained feature schema."""


@dataclass
class TrainingConfig:
    """User-tunable training options."""
    target_risk: float = 0.05
    min_coverage: float = 0.30
    force_model: Optional[str] = None
    n_bootstraps: int = 10
    force_review_on_anomalies: bool = False
    fast: bool = False
    random_state: int = 42
    # Explicit task type ("classification" default | "regression"); never inferred from names.
    task_type: str = CLASSIFICATION
    # Regression only: declared dependence structure for leakage-safe splitting.
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    # Regression only: Model 1 CV selection metric ("mae" | "rmse" | "r2").
    regression_metric: str = DEFAULT_REGRESSION_METRIC
    # Regression only: a prediction "fails" when |y - y_hat| > error_tolerance (target units).
    # When None, the tolerance is this quantile of Model 1's out-of-fold TRAIN absolute errors.
    error_tolerance: Optional[float] = None
    error_tolerance_quantile: float = DEFAULT_TOLERANCE_QUANTILE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_training_frame(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, List[str], List[str], List[Dict[str, str]]]:
    """
    Normalise raw training columns and drop columns with no usable signal.

    - Boolean feature columns become 0/1 integers (numeric).
    - Categorical values are stored as strings (missing stays NaN), so values
      typed later as numbers (e.g. 1 vs "1") match training categories.
    - Constant columns and string identifier columns (100% unique) are dropped.

    Returns (df, numeric_features, categorical_features, dropped_columns).
    """
    df = df.copy()
    numeric: List[str] = []
    categorical: List[str] = []
    dropped: List[Dict[str, str]] = []
    n = len(df)

    for col in df.columns:
        if col == target_column:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            df[col] = series.astype(int)
            series = df[col]

        non_null_unique = series.nunique(dropna=True)
        if non_null_unique <= 1:
            dropped.append({"column": str(col), "reason": "constant or empty column"})
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric.append(col)
        else:
            if non_null_unique == n:
                dropped.append({"column": str(col), "reason": "identifier-like (every value unique)"})
                continue
            df[col] = series.where(series.isna(), series.astype(str))
            categorical.append(col)

    drop_cols = [d["column"] for d in dropped]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if not numeric and not categorical:
        raise DatasetValidationError("No usable feature columns remain after removing constant / ID columns.")

    return df, numeric, categorical, dropped


def build_feature_schema(
    X_train_raw: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[Dict[str, Any]]:
    """Per-feature metadata (from the TRAIN split) used to auto-build the React form."""
    schema: List[Dict[str, Any]] = []
    for col in X_train_raw.columns:
        series = X_train_raw[col]
        entry: Dict[str, Any] = {
            "name": str(col),
            "missing_rate": round(float(series.isna().mean()), 4),
        }
        if col in numeric_features:
            s = pd.to_numeric(series, errors="coerce").dropna()
            is_int = bool(len(s) and np.allclose(s, np.round(s)))
            entry.update({
                "type": "numeric",
                "integer": is_int,
                "min": float(s.min()) if len(s) else None,
                "max": float(s.max()) if len(s) else None,
                "median": float(s.median()) if len(s) else None,
            })
        else:
            counts = series.dropna().astype(str).value_counts()
            entry.update({
                "type": "categorical",
                "categories": sorted(counts.index[:MAX_FORM_CATEGORIES].tolist()),
                "most_frequent": str(counts.index[0]) if len(counts) else None,
                "n_categories": int(len(counts)),
            })
        schema.append(entry)
    return schema


def coerce_records(
    records: pd.DataFrame,
    feature_schema: List[Dict[str, Any]],
    allow_extra_columns: bool = False,
) -> pd.DataFrame:
    """
    Validate and coerce raw inference records against the trained schema.

    - Unknown columns -> error (or silently ignored for batch CSVs when allowed).
    - Absent feature columns -> error (send null / leave blank for a missing value).
    - Non-numeric or infinite values in numeric columns -> error naming the column and rows.
    Missing values (null / NaN / empty string) are ALLOWED: they are a reliability signal.
    """
    expected = [f["name"] for f in feature_schema]
    cols = [str(c) for c in records.columns]
    records = records.copy()
    records.columns = cols

    unknown = [c for c in cols if c not in expected]
    if unknown and not allow_extra_columns:
        raise RecordValidationError(f"Unknown column(s): {unknown}. Expected features: {expected}")
    missing = [c for c in expected if c not in cols]
    if missing:
        raise RecordValidationError(
            f"Missing required field(s): {missing}. Send null for a value that is unknown."
        )

    out = pd.DataFrame(index=records.index)
    for f in feature_schema:
        name = f["name"]
        series = records[name].replace({"": np.nan})
        series = series.where(series.notna(), np.nan)
        if f["type"] == "numeric":
            if series.dtype == bool:
                series = series.astype(int)
            converted = pd.to_numeric(series.map(lambda v: int(v) if isinstance(v, bool) else v),
                                      errors="coerce")
            bad = series.notna() & converted.isna()
            if bad.any():
                bad_rows = [int(i) for i in np.where(bad.values)[0][:5]]
                bad_vals = [str(v) for v in series[bad].head(3).tolist()]
                raise RecordValidationError(
                    f"Field '{name}' must be numeric; got {bad_vals} (row positions {bad_rows})."
                )
            converted = converted.astype(float)
            infinite = np.isinf(converted.to_numpy())
            if infinite.any():
                bad_rows = [int(i) for i in np.where(infinite)[0][:5]]
                raise RecordValidationError(
                    f"Field '{name}' must be a finite number; got infinity (row positions {bad_rows}). "
                    "Send null for an unknown value."
                )
            out[name] = converted
        else:
            out[name] = series.map(lambda v: v if pd.isna(v) else str(v)).astype(object)
    return out


# ---------------------------------------------------------------------------
# Regression data preparation
# ---------------------------------------------------------------------------

def prepare_regression_data(
    df: pd.DataFrame,
    target_column: str,
    config: Optional[TrainingConfig] = None,
) -> Tuple[RegressionSplits, List[str], List[str], List[Dict[str, str]]]:
    """
    Regression data pipeline: feature preparation + leakage-safe 3-way split.

    `df` must already be validated with load_dataset(..., task_type="regression").
    Declared group / time columns are split metadata: they are excluded from the
    features (so they are never fitted by preprocessing) and only drive the split.
    No preprocessing is fitted here; scalers / imputers / encoders are fitted later,
    inside Model 1 pipelines, on the TRAIN partition only.

    Returns (splits, numeric_features, categorical_features, dropped_columns).
    """
    cfg = config or TrainingConfig(task_type=REGRESSION)
    meta_cols = [c for c in (cfg.group_column, cfg.time_column) if c is not None]
    missing = [c for c in meta_cols if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Split column(s) {missing} not found in the dataset.")
    if target_column in meta_cols:
        raise DatasetValidationError("The target column cannot be used as a group or time column.")

    frame, num_cols, cat_cols, dropped = prepare_training_frame(df.drop(columns=meta_cols), target_column)
    for c in meta_cols:
        frame[c] = df[c]
    splits = split_regression_data(
        frame,
        target_column,
        random_state=cfg.random_state,
        group_column=cfg.group_column,
        time_column=cfg.time_column,
    )
    return splits, num_cols, cat_cols, dropped


REGRESSION_ARTIFACT_VERSION = 1


def train_regression_model1(
    df: pd.DataFrame,
    target_column: str,
    config: Optional[TrainingConfig] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Regression Model 1 only (see _fit_regression_model1). Returns (artifacts, report)."""
    artifacts, report, _ = _fit_regression_model1(df, target_column, config)
    return artifacts, report


def _fit_regression_model1(
    df: pd.DataFrame,
    target_column: str,
    config: Optional[TrainingConfig] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Regression Model 1: prepare + split, cross-validate candidates on TRAIN only, refit the
    winner on TRAIN, and score it once on the CALIBRATION partition as an untuned holdout
    check (calibration labels are not used for any choice here).

    The TEST partition is not touched: test metrics are produced only by the final
    evaluation, after Model 2 and the mitigation thresholds are fixed.
    """
    cfg = config or TrainingConfig(task_type=REGRESSION)
    if resolve_task_type(cfg.task_type) != REGRESSION:
        raise ValueError("train_regression_model1 requires task_type='regression'.")
    t0 = time.perf_counter()
    splits, num_cols, cat_cols, dropped = prepare_regression_data(df, target_column, cfg)
    keys = None
    if splits.split_keys is not None:
        keys = splits.split_keys.loc[splits.X_train_raw.index].to_numpy()

    m1 = select_regression_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        selection_metric=cfg.regression_metric,
        force_model=cfg.force_model,
        random_state=cfg.random_state,
        fast=cfg.fast,
        cv_strategy=splits.strategy,
        cv_keys=keys,
    )
    calib_pred = predict_regression({"model1": m1["model"], "feature_schema": None}, splits.X_calib_raw)
    calib_scores = regression_scores(splits.y_calib, calib_pred)
    median_ref = regression_scores(splits.y_calib, np.full(len(splits.y_calib), np.median(splits.y_train)))

    artifacts: Dict[str, Any] = {
        "task_type": REGRESSION,
        "artifact_version": REGRESSION_ARTIFACT_VERSION,
        "model1": m1["model"],
        "model1_name": m1["winner_name"],
        "extractor": None,
        "model2": None,
        "engine": None,
        "reliability_mode": "model1_only",
        "target_column": target_column,
        "feature_schema": build_feature_schema(splits.X_train_raw, num_cols, cat_cols),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "training_config": cfg.to_dict(),
    }
    report: Dict[str, Any] = {
        "task": REGRESSION,
        "target_column": target_column,
        "selected_model": m1["winner_name"],
        "selection_metric": m1["selection_metric"],
        "cv_strategy": m1["cv_strategy"],
        "n_folds": m1["n_folds"],
        "leaderboard": m1["leaderboard"],
        "median_baseline_cv": m1["median_baseline_cv"],
        "calibration_holdout": {
            "model1": calib_scores,
            "median_baseline": median_ref,
            "note": "Calibration partition, not used for model selection. Test metrics are deferred "
                    "until Model 2 and mitigation are finalized.",
        },
        "features": {"numeric": num_cols, "categorical": cat_cols},
        "dropped_columns": dropped,
        "splits": splits.get_split_summary(),
        "test_partition": "untouched",
        "config": cfg.to_dict(),
        "total_seconds": round(time.perf_counter() - t0, 2),
    }
    context = {"splits": splits, "m1": m1, "cfg": cfg, "calib_pred": calib_pred}
    return artifacts, report, context


REGRESSION_RELIABILITY_MODES = ("model1_only", "model2", "constant_fallback")


def train_regression_reliability(
    df: pd.DataFrame,
    target_column: str,
    config: Optional[TrainingConfig] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Regression Model 1 + reliability features + Model 2 (no mitigation thresholds yet).
    Returns (artifacts, JSON-serializable report, in-memory calibration out-of-fold data).

    Data use:
      TRAIN       Model 1 (CV selection + refit), target scale s, feature extractor
                  (incl. bootstrap refits and Model 1's out-of-fold train residuals)
      CALIBRATION Model 1 predictions on rows it never saw -> error targets
                  |y - y_hat| / s -> Model 2 (cross-validated, out-of-fold metrics)
      TEST        not touched
    """
    t0 = time.perf_counter()
    artifacts, report, ctx = _fit_regression_model1(df, target_column, config)
    splits, m1, cfg = ctx["splits"], ctx["m1"], ctx["cfg"]
    model1 = m1["model"]

    scale, scale_method = robust_target_scale(splits.y_train)
    extractor = RegressionReliabilityFeatureExtractor(n_bootstraps=cfg.n_bootstraps, random_state=cfg.random_state)
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=m1["base_pipeline"],
        y_train=splits.y_train,
        train_oof_pred=m1["oof_predictions"],
        target_scale=scale,
        numeric_features=artifacts["numeric_features"],
        categorical_features=artifacts["categorical_features"],
    )
    calib_feats, _ = extractor.transform_with_flags(splits.X_calib_raw, model1)
    calib_err = scaled_abs_error(splits.y_calib, ctx["calib_pred"], scale)

    calib_keys = None
    if splits.split_keys is not None:
        calib_keys = splits.split_keys.loc[splits.X_calib_raw.index].to_numpy()
    cv = make_regression_cv_splits(len(calib_err), splits.strategy, calib_keys, random_state=cfg.random_state)
    m2 = train_regression_model2(calib_feats, calib_err, cv, target_scale=scale, random_state=cfg.random_state)

    mode = "model2" if m2["status"] == "success" else "constant_fallback"
    oof = m2["oof_scaled_error"]
    ok = np.isfinite(oof)
    baselines = {}
    for name, sc in m2["candidates_scores"].items():
        if sc.get("status") == "success":
            baselines[name] = {k: sc[k] for k in ("mse", "mae", "spearman", "mean_estimated", "mean_observed")}

    artifacts.update({
        "extractor": extractor,
        "model2": m2["model2"],
        "reliability_mode": mode,
        "target_scale": scale,
        "target_scale_method": scale_method,
        "fallback_scaled_error": m2["fallback_scaled_error"],
        # Fallback only: calibration range of the reliability features (out-of-support guard).
        "reliability_support": ReliabilitySupport(calib_feats) if m2["model2"] is None else None,
    })
    report.update({
        "reliability_mode": mode,
        "target_scale": {"value": scale, "method": scale_method},
        "model2": {
            "status": m2["status"],
            "message": m2["message"],
            "winner": m2["winner_name"],
            "selection_metric": "oof_mse_of_scaled_abs_error",
            "output": "estimated absolute error of Model 1 (target units; larger = less reliable; "
                      "an expected value, not a probability or a bound)",
            "n_calibration_rows": m2["n_rows"],
            "feature_names": list(m2["model2"].feature_names) if m2["model2"] is not None else [],
            "candidate_feature_names": m2["feature_names"],
            "candidates": m2["candidates_scores"],
            "oof_winner_metrics_scaled": error_estimate_metrics(calib_err[ok], oof[ok]) if ok.sum() > 1 else None,
            "n_oof_rows": int(ok.sum()),
            "calibration_error_scaled": {
                "mean": float(calib_err.mean()),
                "median": float(np.median(calib_err)),
                "p90": float(np.percentile(calib_err, 90)),
                "max": float(calib_err.max()),
            },
        },
        "test_partition": "untouched",
        "total_seconds": round(time.perf_counter() - t0, 2),
    })
    # In memory only, never saved or reported: calibration out-of-fold estimates for the
    # calibration-only threshold tuning of the mitigation phase.
    calibration_dev = {"scaled_error": calib_err, "oof_estimate": oof, "features": calib_feats, "splits": splits,
                       "model1_cv_rows": int(len(m1["oof_predictions"])), "model2_rows": int(m2["n_rows"])}
    return artifacts, report, calibration_dev


def predict_regression_reliability(artifacts: Dict[str, Any], X_raw: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Model 1 prediction + Model 2 estimated absolute error for coerced records.
    Model 2 failures never alter Model 1's prediction: the record falls back to the
    calibration mean error and is flagged.
    """
    if len(X_raw) == 0:
        return []
    X_raw = X_raw[[f["name"] for f in artifacts["feature_schema"]]]
    pred = predict_regression(artifacts, X_raw)
    extractor = artifacts["extractor"]
    scale = float(artifacts["target_scale"])
    fallback = float(artifacts["fallback_scaled_error"])
    model2 = artifacts.get("model2")
    est_scaled = np.full(len(X_raw), fallback)
    source = "calibration_mean_error"
    try:
        feats, flags = extractor.transform_with_flags(X_raw, artifacts["model1"])
    except Exception:
        # Reliability signals unavailable: keep Model 1's prediction, use the default estimate,
        # and flag the record so it is never auto-TRUSTED.
        n = len(X_raw)
        return [{
            "prediction": float(pred[i]),
            "estimated_abs_error": fallback * scale,
            "estimated_error_scaled": fallback,
            "error_estimate_source": "calibration_mean_error",
            "flags": [{"type": "reliability_features_unavailable"}],
            "reliability_features": {},
        } for i in range(n)]
    if model2 is not None:
        try:
            est_scaled = model2.predict_scaled_error(feats)
            source = "model2"
            outside = model2.support_violations(feats)
        except Exception:
            flags = [list(f) + [{"type": "model2_unavailable"}] for f in flags]
            outside = [[] for _ in range(len(X_raw))]
    else:
        support = artifacts.get("reliability_support")
        outside = support.support_violations(feats) if support is not None else [[] for _ in range(len(X_raw))]
    for i, o in enumerate(outside):
        if o:
            flags[i] = list(flags[i]) + [{"type": "outside_reliability_support", "features": o}]

    out = []
    recs = feats.to_dict(orient="records")
    for i in range(len(X_raw)):
        out.append({
            "prediction": float(pred[i]),
            "estimated_abs_error": float(est_scaled[i] * scale),
            "estimated_error_scaled": float(est_scaled[i]),
            "error_estimate_source": source,
            "flags": flags[i],
            "reliability_features": {k: float(v) for k, v in recs[i].items()},
        })
    return out


def predict_regression(artifacts: Dict[str, Any], X_raw: pd.DataFrame) -> np.ndarray:
    """
    Model 1 numeric predictions for records already coerced to the training schema
    (see coerce_records). Columns are put in training order first. Never uses targets.
    """
    if len(X_raw) == 0:
        return np.array([], dtype=float)
    schema = artifacts.get("feature_schema")
    if schema:
        X_raw = X_raw[[f["name"] for f in schema]]
    pred = np.asarray(artifacts["model1"].predict(X_raw), dtype=float).ravel()
    if not np.all(np.isfinite(pred)):
        raise ValueError("Model 1 produced non-finite predictions.")
    return pred


def predict_regression_records(artifacts: Dict[str, Any], X_raw: pd.DataFrame) -> List[Dict[str, Any]]:
    """Full regression scoring: prediction, estimated error, TRUST / REVIEW / REJECT, warnings, explanation."""
    engine = artifacts.get("engine")
    if not isinstance(engine, RegressionMitigationEngine):
        raise ValueError("Regression bundle has no mitigation engine (thresholds were not tuned).")
    out = []
    for r in predict_regression_reliability(artifacts, X_raw):
        ev = engine.evaluate_record(r["estimated_error_scaled"], r["flags"])
        r.update(ev)
        r.update(engine.explain(r["estimated_error_scaled"], ev["decision"], ev["warnings"]))
        r["reliability_mode"] = artifacts["reliability_mode"]
        out.append(r)
    return out


def _train_regression_modeldoctor(
    df: pd.DataFrame,
    target_column: str,
    cfg: TrainingConfig,
    on_stage: Callable[[str, str], None],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Regression Phase A:
      TRAIN       Model 1, target scale, feature extractor, error tolerance (default)
      CALIBRATION Model 2 + its out-of-fold estimates -> TRUST / REJECT thresholds
      TEST        scored exactly once, after every model and threshold is fixed
    """
    on_stage("model_selection", "Selecting Model 1 and training Model 2 (train / calibration splits)")
    artifacts, report, dev = train_regression_reliability(df, target_column, cfg)
    splits, scale = dev["splits"], float(artifacts["target_scale"])

    on_stage("reliability_training", "Tuning TRUST / REVIEW / REJECT thresholds on calibration out-of-fold estimates")
    tol = resolve_error_tolerance(
        artifacts["extractor"].train_abs_residual_, scale,
        error_tolerance=cfg.error_tolerance, quantile=cfg.error_tolerance_quantile,
    )
    thresholds = tune_regression_thresholds(
        dev["oof_estimate"], dev["scaled_error"], tol["scaled"], target_scale=scale,
        target_risk=cfg.target_risk, min_coverage=cfg.min_coverage,
    )
    engine = RegressionMitigationEngine(thresholds, scale, tol, cfg.force_review_on_anomalies)
    artifacts["engine"] = engine
    artifacts["error_tolerance"] = tol
    # Shared identifier tying this report's metrics to these artifacts.
    artifacts["run_id"] = uuid.uuid4().hex

    on_stage("evaluation", "Evaluating on the held-out test split (once)")
    results = predict_regression_records(artifacts, splits.X_test_raw)
    test_pred = np.array([r["prediction"] for r in results])
    test_est = np.array([r["estimated_abs_error"] for r in results])
    feats = [r["reliability_features"] for r in results]
    baselines = {
        "Model 1 bootstrap spread": np.array([f.get("bootstrap_std", np.nan) for f in feats]),
        "Local training error (kNN out-of-fold residual)": np.array([f.get("knn_oof_residual", np.nan) for f in feats]),
        "Distance to training data (kNN)": np.array([f.get("knn_distance", np.nan) for f in feats]),
    }
    evaluation = evaluate_regression_mitigation(
        splits.y_test, test_pred, test_est, [r["decision"] for r in results], tol["value"], baselines,
    )
    test_abs_err = np.abs(splits.y_test - test_pred)
    trust_mask = np.array([r["decision"] == "TRUST" for r in results])
    artifacts["monitoring_baseline"] = {
        "task_type": REGRESSION,
        "test_coverage": round(float(trust_mask.mean()), 4) if len(results) else None,
        "test_trust_failure_rate": (round(float((test_abs_err[trust_mask] > tol["value"]).mean()), 4)
                                    if trust_mask.any() else None),
        "test_mae": float(test_abs_err.mean()) if len(results) else None,
        "test_mean_estimated_error": float(test_est.mean()) if len(results) else None,
        "test_anomaly_rate": round(float(np.mean([
            any(w["code"] in ANOMALY_CODES for w in r["warnings"]) for r in results
        ])), 4) if results else 0.0,
        "error_tolerance": tol["value"],
    }
    report.update({
        "thresholds": engine.to_dict(),
        "error_tolerance": tol,
        # Test-split evaluation, separated by question (Phase 5).
        "test_evaluation": {
            "model1": model1_metrics(splits.y_test, test_pred, float(np.median(splits.y_train))),
            "model2": error_estimation_metrics(
                test_abs_err, test_est,
                constant_estimate=float(artifacts["fallback_scaled_error"]) * scale,
                ranking_baselines=baselines,
            ),
            "uncertainty": {
                "prediction_intervals": None,
                "note": "This version provides an estimated absolute error per prediction, not a prediction "
                        "interval; no interval coverage is claimed.",
            },
        },
        "mitigation_evaluation": evaluation,
        "leakage_audit": leakage_audit(
            splits, artifacts,
            model1_cv_rows=dev["model1_cv_rows"],
            model2_rows=dev["model2_rows"],
            threshold_rows=int(thresholds["calibration_rows"]),
            test_rows_evaluated=len(results),
        ),
        "reproducibility": reproducibility_record(df, splits, cfg.to_dict(), artifacts),
        "test_partition": "evaluated once, after Model 1, Model 2 and thresholds were fixed",
    })
    return artifacts, report


# ---------------------------------------------------------------------------
# Training (Phase A)
# ---------------------------------------------------------------------------

def _noop_stage(stage: str, message: str = "") -> None:
    return None


def train_modeldoctor(
    df: pd.DataFrame,
    target_column: str,
    config: Optional[TrainingConfig] = None,
    on_stage: Callable[[str, str], None] = _noop_stage,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run the full ModelDoctor setup on a validated DataFrame.

    Returns
    -------
    (artifacts, report)
        artifacts: fitted objects + metadata needed for inference (joblib-able)
        report: JSON-serializable system report for the dashboard
    """
    cfg = config or TrainingConfig()
    if resolve_task_type(cfg.task_type) == REGRESSION:
        return _train_regression_modeldoctor(df, target_column, cfg, on_stage)
    t_start = time.perf_counter()
    timings: Dict[str, float] = {}

    def stage(name: str, msg: str) -> float:
        on_stage(name, msg)
        return time.perf_counter()

    # 1. Profiling -----------------------------------------------------------
    t0 = stage("profiling", "Profiling dataset")
    profile = profile_dataset(df, target_column)
    df_clean, num_cols, cat_cols, dropped = prepare_training_frame(df, target_column)
    timings["profiling"] = time.perf_counter() - t0

    # 2. Splitting -----------------------------------------------------------
    t0 = stage("splitting", "Stratified 60/20/20 train / calibration / test split")
    splits = split_data(df_clean, target_column, random_state=cfg.random_state)
    num_classes = len(splits.classes)
    timings["splitting"] = time.perf_counter() - t0

    # 3. Model 1 selection (TRAIN split only) -------------------------------
    t0 = stage("model_selection", "Cross-validating candidate models on the train split")
    m1 = select_model(
        X_train_raw=splits.X_train_raw,
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        force_model=cfg.force_model,
        random_state=cfg.random_state,
        fast=cfg.fast,
    )
    model1 = m1["model"]
    timings["model_selection"] = time.perf_counter() - t0

    # 4. Reliability features (extractor fit on TRAIN; applied to calibration) --
    t0 = stage("reliability_features", "Fitting reliability signals and scoring the calibration split")
    extractor = ReliabilityFeatureExtractor(
        n_bootstraps=cfg.n_bootstraps,
        random_state=cfg.random_state,
    )
    extractor.fit(
        X_train_raw=splits.X_train_raw,
        base_pipeline=m1["base_pipeline"],
        y_train=splits.y_train,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )
    calib_feats, _ = extractor.transform_with_flags(splits.X_calib_raw, model1)
    calib_pred = np.argmax(model1.predict_proba(splits.X_calib_raw), axis=1)
    calib_correct = (calib_pred == splits.y_calib).astype(int)
    timings["reliability_features"] = time.perf_counter() - t0

    # 5. Model 2 + thresholds (CALIBRATION split only) ----------------------
    t0 = stage("reliability_training", "Training Model 2 and tuning thresholds on calibration out-of-fold scores")
    m2 = train_model2(
        calib_feats,
        calib_correct,
        num_classes=num_classes,
        random_state=cfg.random_state,
    )
    if m2["status"] == "success":
        reliability_mode = "model2"
        model2 = m2["model2"]
        threshold_scores = m2["oof_trustworthiness"]
    else:
        # Fallback: Model 1 never saw the calibration split, so its confidence there
        # is an honest out-of-sample score for tuning thresholds.
        reliability_mode = "confidence_fallback"
        model2 = None
        threshold_scores = calib_feats["max_prob"].to_numpy()
    # In fallback mode the thresholds are only validated on the calibration range of the
    # reliability signals; inputs outside it must not be auto-TRUSTED (Model 2 mode tracks
    # the same range inside Model2Detector).
    reliability_support = ReliabilitySupport(calib_feats) if model2 is None else None

    thresholds = tune_mitigation_thresholds(
        threshold_scores,
        calib_correct,
        target_risk=cfg.target_risk,
        min_coverage=cfg.min_coverage,
    )
    engine = MitigationEngine(
        trust_threshold=thresholds["trust_threshold"],
        reject_threshold=thresholds["reject_threshold"],
        target_risk=thresholds["target_risk"],
        min_coverage=thresholds["min_coverage"],
        target_risk_met=thresholds["target_risk_met"],
        message=thresholds["message"],
        force_review_on_anomalies=cfg.force_review_on_anomalies,
        calibration_coverage=thresholds["calibration_coverage"],
        calibration_selective_risk=thresholds["calibration_selective_risk"],
        num_classes=num_classes,
        band_metadata=thresholds["band_metadata"],
    )
    timings["reliability_training"] = time.perf_counter() - t0

    artifacts: Dict[str, Any] = {
        "model1": model1,
        "extractor": extractor,
        "model2": model2,
        "engine": engine,
        "reliability_mode": reliability_mode,
        "reliability_support": reliability_support,
        "classes": list(splits.classes),
        "num_classes": num_classes,
        "target_column": target_column,
        "feature_schema": build_feature_schema(splits.X_train_raw, num_cols, cat_cols),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
    }

    # 6. Evaluation (TEST split, touched exactly once) ----------------------
    t0 = stage("evaluation", "Evaluating ModelDoctor vs confidence baseline on the held-out test split")
    test_results = predict_records(artifacts, splits.X_test_raw)
    test_pred = np.array([splits.class_mapping[r["prediction"]] for r in test_results])
    test_correct = (test_pred == splits.y_test).astype(int)
    evaluation = evaluate_system(
        correct=test_correct,
        confidence=np.array([r["confidence"] for r in test_results]),
        trustworthiness=np.array([r["trustworthiness"] for r in test_results]),
        decisions=[r["decision"] for r in test_results],
        reliability_mode=reliability_mode,
        model2_winner=m2["winner_name"],
    )
    timings["evaluation"] = time.perf_counter() - t0

    # Baseline rates of input-anomaly warnings on test data, used by monitoring drift checks
    test_anomaly_rate = float(np.mean([
        any(w["code"] in ANOMALY_CODES for w in r["warnings"]) for r in test_results
    ])) if test_results else 0.0
    artifacts["monitoring_baseline"] = {
        "test_anomaly_rate": round(test_anomaly_rate, 4),
        "test_trust_accuracy": round(1.0 - evaluation["comparison"][2]["selective_risk"], 4),
        "test_coverage": evaluation["comparison"][2]["coverage"],
        "test_accuracy": round(1.0 - evaluation["overall_error_rate"], 4),
    }

    leaderboard = [
        {
            "model": row["model"],
            "score": row.get(m1["primary_metric"]),
            "train_seconds": row.get("train_seconds"),
            **{k: v for k, v in row.items() if k not in ("model",)},
        }
        for row in m1["leaderboard"]
    ]

    model2_effects = None
    final_clf = model2.pipeline.named_steps.get("classifier") if model2 is not None else None
    if final_clf is not None and hasattr(final_clf, "coef_") and final_clf.coef_.shape[1] == len(model2.feature_names):
        # Standardized coefficients; for the probability candidates max_prob enters as logit(max_prob).
        coefs = final_clf.coef_[0]
        model2_effects = sorted(
            [{"feature": f, "coefficient": round(float(c), 4)} for f, c in zip(model2.feature_names, coefs)],
            key=lambda d: -abs(d["coefficient"]),
        )

    report: Dict[str, Any] = {
        "task": "classification",
        "task_detail": profile["detected_task"],
        "target_column": target_column,
        "classes": list(splits.classes),
        "selected_model": m1["winner_name"],
        "primary_metric": m1["primary_metric"],
        "calibration_method": m1["calibration_method"],
        "leaderboard": leaderboard,
        "profile": profile,
        "dropped_columns": dropped,
        "features": {"numeric": num_cols, "categorical": cat_cols},
        "splits": splits.get_split_summary(),
        "reliability_mode": reliability_mode,
        "model2": {
            "status": m2["status"],
            "message": m2["message"],
            "winner": m2["winner_name"],
            "feature_names": list(model2.feature_names) if model2 is not None else [],
            "candidate_feature_names": m2["feature_names"],
            "selection_metric": m2.get("selection_metric"),
            "candidates": m2["candidates_scores"],
            "calibration_ece_oof": m2["ece"] if reliability_mode == "model2" else None,
            "calibration_brier_oof": m2["brier_score"] if reliability_mode == "model2" else None,
            "n_correct_calibration": m2["n_correct"],
            "n_incorrect_calibration": m2["n_incorrect"],
            "reliability_table_oof": m2["reliability_table"] if reliability_mode == "model2"
            else compute_reliability_table(calib_correct, threshold_scores),
            "feature_effects": model2_effects,
        },
        "thresholds": engine.to_dict(),
        **evaluation,
        "config": cfg.to_dict(),
        "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
        "total_seconds": round(time.perf_counter() - t_start, 2),
    }
    return artifacts, report


# ---------------------------------------------------------------------------
# Inference (Phase B)
# ---------------------------------------------------------------------------

def predict_records(artifacts: Dict[str, Any], X_raw: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Score raw records (already coerced to the training schema).
    True outcomes are never used here.
    """
    if artifacts.get("task_type", CLASSIFICATION) != CLASSIFICATION:
        raise ValueError("predict_records serves classification bundles only; use predict_regression.")
    if len(X_raw) == 0:
        return []
    feature_order = [f["name"] for f in artifacts["feature_schema"]]
    X_raw = X_raw[feature_order]

    model1 = artifacts["model1"]
    extractor: ReliabilityFeatureExtractor = artifacts["extractor"]
    engine: MitigationEngine = artifacts["engine"]
    classes: List[str] = artifacts["classes"]

    probs = np.asarray(model1.predict_proba(X_raw), dtype=float)
    pred_idx = np.argmax(probs, axis=1)
    feats, flags = extractor.transform_with_flags(X_raw, model1)

    if artifacts.get("model2") is not None:
        model2 = artifacts["model2"]
        trust = model2.predict_trustworthiness(feats)
        for i, outside in enumerate(model2.support_violations(feats)):
            if outside:
                flags[i] = list(flags[i]) + [{"type": "outside_reliability_support", "features": outside}]
    else:
        trust = feats["max_prob"].to_numpy()
        support = artifacts.get("reliability_support")  # absent in artifacts saved before it existed
        if support is not None:
            for i, outside in enumerate(support.support_violations(feats)):
                if outside:
                    flags[i] = list(flags[i]) + [{"type": "outside_reliability_support", "features": outside}]

    results: List[Dict[str, Any]] = []
    feat_records = feats.to_dict(orient="records")
    for i in range(len(X_raw)):
        rec_feats = {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                     for k, v in feat_records[i].items()}
        out = engine.evaluate_record(float(trust[i]), flags[i], rec_feats)
        results.append({
            "prediction": classes[int(pred_idx[i])],
            "confidence": round(float(probs[i, pred_idx[i]]), 4),
            "trustworthiness": out["trustworthiness"],
            "decision": out["decision"],
            "warnings": out["warnings"],
            "class_probabilities": {classes[j]: round(float(probs[i, j]), 4) for j in range(len(classes))},
            "reliability_features": {k: round(v, 4) if isinstance(v, float) else v for k, v in rec_feats.items()},
        })
    return results
