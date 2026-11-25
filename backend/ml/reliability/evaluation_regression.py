"""Regression mitigation evaluation (TEST split, after every model and threshold is fixed).

CONVENTIONS
- abs_error = |y - y_hat| (target units). failure = abs_error > error tolerance delta.
- Every score here is "lower = more reliable" (estimated error, bootstrap spread, distance).
- Matched coverage: a baseline TRUSTs exactly as many rows as ModelDoctor TRUSTs (its lowest
  scores) and REJECTs exactly as many as ModelDoctor REJECTs (its highest scores); the rest
  is REVIEW. Ties are broken by row order (stable sort), identically for every method.
- All methods are evaluated on exactly the same test rows.
"""

from typing import Any, Dict, List, Optional

import numpy as np

DECISIONS = ("TRUST", "REVIEW", "REJECT")

REGRESSION_METRIC_DEFINITIONS: Dict[str, str] = {
    "abs_error": "|y - y_hat| in target units.",
    "failure": "abs_error > error tolerance (configured, or a quantile of Model 1's out-of-fold train errors).",
    "failure_rate": "Share of failures in a band (the regression 'risk'). None for an empty band.",
    "mae / rmse": "Mean absolute / root mean squared error of Model 1 within a band (target units).",
    "coverage": "Share of test rows in a band; TRUST coverage = automatic-use rate.",
    "matched_coverage": "Baselines get the same TRUST and REJECT band sizes as ModelDoctor (lowest scores "
                        "TRUSTed, highest REJECTed).",
    "risk_coverage_curve": "Rows accepted in increasing score order; selective MAE and failure rate at each coverage.",
    "aurc_mae": "Mean selective MAE over all coverage levels (lower is better).",
}


def _band_stats(abs_err: np.ndarray, fail: np.ndarray, mask: np.ndarray, n: int) -> Dict[str, Any]:
    k = int(mask.sum())
    return {
        "count": k,
        "coverage": round(k / n, 4) if n else 0.0,
        "failures": int(fail[mask].sum()),
        "failure_rate": round(float(fail[mask].mean()), 4) if k else None,
        "mae": float(abs_err[mask].mean()) if k else None,
        "rmse": float(np.sqrt(np.mean(abs_err[mask] ** 2))) if k else None,
    }


def band_table(abs_err: np.ndarray, fail: np.ndarray, decisions: np.ndarray) -> List[Dict[str, Any]]:
    n = len(abs_err)
    return [{"decision": d, **_band_stats(abs_err, fail, decisions == d, n)} for d in DECISIONS]


def matched_decisions(score: np.ndarray, n_trust: int, n_reject: int) -> np.ndarray:
    """TRUST the n_trust lowest scores, REJECT the n_reject highest, REVIEW the rest."""
    n = len(score)
    order = np.argsort(np.asarray(score, dtype=float), kind="mergesort")
    out = np.full(n, "REVIEW", dtype=object)
    out[order[:n_trust]] = "TRUST"
    if n_reject > 0:
        out[order[n - n_reject:]] = "REJECT"
    return out


def regression_risk_coverage(abs_err: np.ndarray, fail: np.ndarray, score: np.ndarray,
                             n_points: int = 50) -> Dict[str, Any]:
    n = len(abs_err)
    if n == 0:
        return {"curve": [], "aurc_mae": None}
    order = np.argsort(np.asarray(score, dtype=float), kind="mergesort")
    cum_err = np.cumsum(abs_err[order])
    cum_fail = np.cumsum(fail[order])
    ks = np.unique(np.clip(np.round(np.linspace(1, n, n_points)).astype(int), 1, n))
    curve = [{"coverage": round(k / n, 4), "selective_mae": float(cum_err[k - 1] / k),
              "selective_failure_rate": round(float(cum_fail[k - 1] / k), 4)} for k in ks]
    aurc = float(np.mean(cum_err / np.arange(1, n + 1)))
    return {"curve": curve, "aurc_mae": aurc}


def evaluate_regression_mitigation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    est_error: np.ndarray,
    decisions: List[str],
    error_tolerance: float,
    baselines: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Mitigation quality on the TEST split. est_error is ModelDoctor's estimated absolute error
    (target units); baselines maps a name to an alternative score (lower = more reliable).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    est = np.asarray(est_error, dtype=float)
    dec = np.asarray(decisions, dtype=object)
    n = len(y_true)
    abs_err = np.abs(y_true - y_pred)
    fail = abs_err > error_tolerance
    all_rows = np.ones(n, dtype=bool)

    md_bands = band_table(abs_err, fail, dec)
    n_trust = md_bands[0]["count"]
    n_reject = md_bands[2]["count"]
    md_trust = md_bands[0]

    comparisons = []
    for name, score in (baselines or {}).items():
        bdec = matched_decisions(np.asarray(score, dtype=float), n_trust, n_reject)
        bb = band_table(abs_err, fail, bdec)
        bt = bb[0]
        comparisons.append({
            "baseline": name,
            "bands": bb,
            "trust_failures_reduction": bt["failures"] - md_trust["failures"],
            "trust_mae_reduction": (None if bt["mae"] is None or md_trust["mae"] is None
                                    else bt["mae"] - md_trust["mae"]),
            "trust_mae_reduction_relative": (None if not bt["mae"] or md_trust["mae"] is None
                                             else (bt["mae"] - md_trust["mae"]) / bt["mae"]),
            "risk_coverage": regression_risk_coverage(abs_err, fail, score),
        })

    return {
        "n_test": int(n),
        "error_tolerance": float(error_tolerance),
        "no_filtering": _band_stats(abs_err, fail, all_rows, n),
        "modeldoctor_bands": md_bands,
        "overall_coverage": md_trust["coverage"],
        "matched_coverage_baselines": comparisons,
        "risk_coverage": {"modeldoctor": regression_risk_coverage(abs_err, fail, est)},
        "metric_definitions": REGRESSION_METRIC_DEFINITIONS,
    }


# ---------------------------------------------------------------------------
# Model 1 and Model 2 test metrics (Phase 5)
# ---------------------------------------------------------------------------

def _check_pair(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{name_a} and {name_b} have different lengths ({len(a)} vs {len(b)}).")


def model1_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train_median: float) -> Dict[str, Any]:
    """
    Model 1 prediction quality on held-out rows: MAE, RMSE, R², plus a no-skill reference
    (predict the TRAIN median). Non-finite predictions are an error (never silently dropped).
    R² is None when the evaluated targets are constant (undefined); all metrics are None
    for an empty set.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    _check_pair(y, p, "y_true", "y_pred")
    if not np.all(np.isfinite(y)):
        raise ValueError("Evaluation targets contain non-finite values.")
    bad = int((~np.isfinite(p)).sum())
    if bad:
        raise ValueError(f"Model 1 produced {bad} non-finite prediction(s); metrics would be meaningless.")
    n = len(y)
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "r2_note": "empty evaluation set",
                "median_baseline": {"mae": None, "rmse": None}, "mae_skill_vs_median": None}
    err = p - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    base = np.full(n, float(y_train_median)) - y
    mae, base_mae = float(np.mean(np.abs(err))), float(np.mean(np.abs(base)))
    return {
        "n": n,
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": None if ss_tot == 0.0 else float(1 - np.sum(err ** 2) / ss_tot),
        "r2_note": "undefined: evaluated targets are constant" if ss_tot == 0.0 else None,
        "median_baseline": {"mae": base_mae, "rmse": float(np.sqrt(np.mean(base ** 2)))},
        # 1 - MAE / MAE(train-median predictor): 0 = no better than the median, 1 = perfect
        "mae_skill_vs_median": None if base_mae == 0 else float(1 - mae / base_mae),
    }


def _varies(v: np.ndarray) -> bool:
    """True when v is not (numerically) constant; float dust such as 1e-17 does not count."""
    return len(v) > 1 and np.std(v) > 1e-12 * (np.abs(np.mean(v)) + 1e-12)


def _ranks_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    from scipy.stats import spearmanr

    if not (_varies(a) and _varies(b)):
        return None
    return float(spearmanr(a, b).correlation)


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not (_varies(a) and _varies(b)):
        return None
    return float(np.corrcoef(a, b)[0, 1])


def error_estimation_metrics(
    abs_error: np.ndarray,
    est_error: np.ndarray,
    constant_estimate: Optional[float] = None,
    ranking_baselines: Optional[Dict[str, np.ndarray]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    How well Model 2's estimated absolute error matches the observed absolute error
    (same units). Rows with a missing / non-finite estimate are excluded and counted.

    - mae / rmse of the estimate, and the same for a constant estimate (the calibration
      mean error) as the no-skill reference
    - spearman (error ranking, primary) and pearson correlations
    - calibration by equal-count bins of the estimate: mean estimated vs mean observed
    - spearman of alternative ranking signals (lower = more reliable) for comparison
    """
    obs_all = np.asarray(abs_error, dtype=float)
    est_all = np.asarray(est_error, dtype=float)
    _check_pair(obs_all, est_all, "abs_error", "est_error")
    ok = np.isfinite(est_all) & np.isfinite(obs_all)
    obs, est = obs_all[ok], est_all[ok]
    n = len(obs)
    out: Dict[str, Any] = {"n": n, "n_missing_estimates": int((~ok).sum())}
    if n == 0:
        out.update({"mae": None, "rmse": None, "spearman": None, "pearson": None,
                    "mean_estimated": None, "mean_observed": None, "calibration_bins": [],
                    "constant_reference": None, "ranking_baselines": {}})
        return out
    out.update({
        "mae": float(np.mean(np.abs(est - obs))),
        "rmse": float(np.sqrt(np.mean((est - obs) ** 2))),
        "spearman": _ranks_corr(est, obs),
        "pearson": _pearson(est, obs),
        "mean_estimated": float(est.mean()),
        "mean_observed": float(obs.mean()),
    })
    if constant_estimate is not None and np.isfinite(constant_estimate):
        c = np.full(n, float(constant_estimate))
        out["constant_reference"] = {"estimate": float(constant_estimate),
                                     "mae": float(np.mean(np.abs(c - obs))),
                                     "rmse": float(np.sqrt(np.mean((c - obs) ** 2)))}
    else:
        out["constant_reference"] = None
    bins = []
    if n >= n_bins:
        order = np.argsort(est, kind="mergesort")
        for chunk in np.array_split(order, n_bins):
            bins.append({"count": int(len(chunk)), "mean_estimated": float(est[chunk].mean()),
                         "mean_observed": float(obs[chunk].mean())})
    out["calibration_bins"] = bins
    out["ranking_baselines"] = {
        name: _ranks_corr(np.asarray(score, dtype=float)[ok], obs)
        for name, score in (ranking_baselines or {}).items()
    }
    return out
