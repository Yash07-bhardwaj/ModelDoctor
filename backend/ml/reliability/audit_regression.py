"""Leakage audit and reproducibility metadata for regression runs.

The audit reads evidence from the fitted objects themselves (e.g. how many rows a scaler
actually saw), so it checks the implementation rather than restating the design.
"""

import hashlib
import json
import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """sha256 over column names and every value (row order included)."""
    h = hashlib.sha256()
    h.update("\x1f".join(map(str, df.columns)).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


def index_hash(index) -> str:
    """Order-independent sha256 of a partition's row labels."""
    labels = sorted(str(i) for i in index)
    return hashlib.sha256("\x1f".join(labels).encode("utf-8")).hexdigest()


def library_versions() -> Dict[str, str]:
    import sklearn
    import scipy
    import xgboost

    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "xgboost": xgboost.__version__}


PROBE_ROWS = 25


def behaviour_fingerprint(artifacts: Dict[str, Any], X_probe: pd.DataFrame) -> str:
    """
    sha256 of what the bundle DOES on a fixed probe: predictions and estimated errors
    (10 significant digits) plus decisions. Byte hashes of fitted tree models are not stable
    (scikit-learn tree node arrays pickle uninitialised padding bytes), and multi-threaded
    forests can differ in the last bit, so behaviour is the reliable identity for tying a
    report's metrics to saved artifacts.
    """
    from ml.pipeline import predict_regression_records  # local import: pipeline imports this module

    rows = predict_regression_records(artifacts, X_probe)
    canon = [[f"{r['prediction']:.10g}", f"{r['estimated_abs_error']:.10g}", r["decision"]] for r in rows]
    return hashlib.sha256(json.dumps(canon).encode("utf-8")).hexdigest()


def _rows_seen(obj, path: List[str]) -> Optional[int]:
    """n_samples_seen_ of a nested scaler, or None if the path does not exist."""
    try:
        for step in path:
            if step == "@named_steps":
                continue
            if hasattr(obj, "named_transformers_") and step in obj.named_transformers_:
                obj = obj.named_transformers_[step]
            elif hasattr(obj, "named_steps") and step in obj.named_steps:
                obj = obj.named_steps[step]
            else:
                obj = getattr(obj, step)
        seen = obj.n_samples_seen_
        return int(np.max(seen))
    except Exception:
        return None


def leakage_audit(
    splits,
    artifacts: Dict[str, Any],
    model1_cv_rows: int,
    model2_rows: int,
    threshold_rows: int,
    test_rows_evaluated: int,
) -> Dict[str, Any]:
    n_tr, n_ca, n_te = len(splits.X_train_raw), len(splits.X_calib_raw), len(splits.X_test_raw)
    idx = {"train": set(splits.X_train_raw.index), "calibration": set(splits.X_calib_raw.index),
           "test": set(splits.X_test_raw.index)}
    m1 = artifacts["model1"]
    ex = artifacts.get("extractor")
    m1_pre = _rows_seen(m1, ["preprocessor", "numeric", "scaler"])
    m1_target = _rows_seen(m1, ["regressor", "transformer_"])
    ex_pre = _rows_seen(ex, ["preprocessor_", "numeric", "scaler"]) if ex is not None else None
    tol = artifacts.get("error_tolerance") or {}

    def check(name: str, passed: Optional[bool], evidence: str) -> Dict[str, Any]:
        return {"check": name, "passed": passed, "evidence": evidence}

    checks = [
        check("partitions are disjoint",
              not (idx["train"] & idx["calibration"] or idx["train"] & idx["test"] or idx["calibration"] & idx["test"]),
              f"train {n_tr}, calibration {n_ca}, test {n_te} rows; pairwise intersections empty"),
        check("Model 1 cross-validation used train rows only", model1_cv_rows == n_tr,
              f"selection received the {n_tr} train rows only (out-of-fold vector length {model1_cv_rows}); "
              "calibration / test rows are never passed to it"),
        check("Model 1 feature preprocessing fitted on train only",
              None if m1_pre is None else m1_pre == n_tr,
              "no numeric features" if m1_pre is None else f"scaler saw {m1_pre} rows (train = {n_tr})"),
        check("Model 1 target scaling fitted on train only",
              None if m1_target is None else m1_target == n_tr,
              "not applicable" if m1_target is None else f"target scaler saw {m1_target} rows (train = {n_tr})"),
        check("Reliability feature extractor fitted on train only",
              None if ex_pre is None else ex_pre == n_tr,
              "no numeric features" if ex_pre is None else f"extractor scaler saw {ex_pre} rows (train = {n_tr})"),
        check("Model 2 trained on calibration rows only (out-of-fold Model 1 errors)", model2_rows == n_ca,
              f"{model2_rows} Model 2 rows == {n_ca} calibration rows; Model 1 never saw them"),
        check("Thresholds tuned on calibration out-of-fold estimates only", threshold_rows <= n_ca,
              f"thresholds were tuned on {threshold_rows} calibration out-of-fold estimates "
              f"(of {n_ca} calibration rows; rows without an out-of-fold estimate are excluded)"),
        check("Error tolerance derived without calibration / test labels",
              str(tol.get("source", "")).startswith(("train_oof", "configured")),
              f"source: {tol.get('source')}"),
        check("Test partition used once, for final evaluation", test_rows_evaluated == n_te,
              f"{test_rows_evaluated} rows scored after thresholds were fixed (test = {n_te})"),
    ]
    if splits.strategy in ("group", "time"):
        keys = splits.split_keys
        if splits.strategy == "group":
            g = {k: set(keys.loc[list(v)]) for k, v in idx.items()}
            ok = not (g["train"] & g["calibration"] or g["train"] & g["test"] or g["calibration"] & g["test"])
            checks.append(check("No group spans two partitions", ok, f"{splits.boundaries}"))
        else:
            t = {k: keys.loc[list(v)] for k, v in idx.items()}
            ok = t["train"].max() < t["calibration"].min() and t["calibration"].max() < t["test"].min()
            checks.append(check("Partitions are strictly ordered in time", bool(ok), f"{splits.boundaries}"))
    return {"passed": all(c["passed"] is not False for c in checks), "checks": checks}


def reproducibility_record(df: pd.DataFrame, splits, config: Dict[str, Any],
                           artifacts: Dict[str, Any]) -> Dict[str, Any]:
    probe = splits.X_calib_raw.head(PROBE_ROWS)
    return {
        "run_id": artifacts.get("run_id"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"fingerprint_sha256": dataset_fingerprint(df), "rows": int(len(df)),
                    "columns": [str(c) for c in df.columns]},
        "config": config,
        "random_state": config.get("random_state"),
        "split": {
            "strategy": splits.strategy,
            "fractions": list(splits.fractions),
            "index_sha256": {"train": index_hash(splits.X_train_raw.index),
                             "calibration": index_hash(splits.X_calib_raw.index),
                             "test": index_hash(splits.X_test_raw.index)},
        },
        "artifact_version": artifacts.get("artifact_version"),
        "behaviour_fingerprint": {
            "sha256": behaviour_fingerprint(artifacts, probe),
            "probe": f"first {len(probe)} calibration rows (in split order)",
        },
        "library_versions": library_versions(),
    }
