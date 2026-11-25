"""TRUST / REVIEW / REJECT mitigation for REGRESSION.

SCORE: Model 2's estimated absolute error, scaled by the training target scale s
(e_hat = estimated |y - y_hat| / s). LOWER = more reliable (the opposite direction of the
classification P(correct) score, so classification thresholds are never reused).

FAILURE: a prediction fails when its absolute error exceeds the error tolerance delta:
    failure = |y - y_hat| > delta
delta comes from configuration (`error_tolerance`, target units) or, by default, the
`error_tolerance_quantile` (0.9) of Model 1's OUT-OF-FOLD absolute errors on the TRAIN
partition: "an error larger than 90% of Model 1's typical errors". It never comes from
calibration or test labels, and it is stored in target units and scaled units.

DECISIONS (thresholds tuned on CALIBRATION out-of-fold estimates only):
- TRUST : e_hat <= trust_threshold. The largest such band (maximum coverage) whose
          calibration failure rate is <= target_risk with coverage >= min_coverage.
          Meaning: estimated risk of a large error is low enough to use automatically.
- REJECT: e_hat > reject_threshold. The largest high-score band with >= reject_min_rows
          rows whose calibration failure rate is >= max(reject_min_error,
          min(reject_error_cap, reject_relative_factor * base failure rate)).
          Meaning: a large error is more likely than the system supports; abstain.
- REVIEW: everything else (moderate or uncertain risk), plus overrides below.

If the target risk cannot be met, TRUST is the min_coverage band with the lowest failure
rate, but only if that band is genuinely better than accepting everything; otherwise TRUST
is disabled (no meaningful separation). With constant scores (fallback mode) TRUST covers
everything only if the overall calibration failure rate already meets the target.
With fewer than MIN_THRESHOLD_ROWS usable calibration rows, TRUST and REJECT are disabled.

OVERRIDES (never TRUST): reliability features outside the calibration range, Model 2 or the
feature extractor unavailable, an invalid (non-finite / negative) estimate, and, when
configured, out-of-range / unseen input values.

The calibration band rates are optimistic (thresholds were tuned on them); the TRUST band
also reports a Wilson 95% upper bound to show the effect of calibration sample size.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ml.reliability.mitigation import DOMAIN_ANOMALY_FLAGS, RISK_LEVELS, generate_warnings

MIN_THRESHOLD_ROWS = 50
DEFAULT_TOLERANCE_QUANTILE = 0.9
NEVER_TRUST_FLAGS = ("outside_reliability_support", "model2_unavailable", "reliability_features_unavailable")


def wilson_upper(failures: int, n: int, z: float = 1.96) -> Optional[float]:
    """Upper end of the Wilson score interval for a failure rate (None when n == 0)."""
    if n <= 0:
        return None
    p = failures / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(min(1.0, (centre + half) / denom))


def resolve_error_tolerance(
    train_abs_residuals: np.ndarray,
    target_scale: float,
    error_tolerance: Optional[float] = None,
    quantile: float = DEFAULT_TOLERANCE_QUANTILE,
) -> Dict[str, Any]:
    """delta in target units and scaled units, from config or train out-of-fold residuals."""
    if error_tolerance is not None:
        delta = float(error_tolerance)
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("error_tolerance must be a positive finite number (target units).")
        source = "configured"
    else:
        if not 0 < quantile < 1:
            raise ValueError("error_tolerance_quantile must be strictly between 0 and 1.")
        r = np.asarray(train_abs_residuals, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) == 0:
            raise ValueError("No out-of-fold train residuals to derive the error tolerance from.")
        delta = float(np.quantile(r, quantile))
        source = f"train_oof_abs_error_q{quantile:g}"
        if delta <= 0:  # Model 1 is (near) exact on most train rows: any error counts as failure
            pos = r[r > 0]
            delta = float(pos.min()) if len(pos) else float(target_scale) * 1e-9
            source += "_min_positive"
    return {"value": delta, "scaled": delta / float(target_scale), "source": source}


def _band(mask: np.ndarray, fail: np.ndarray, err: np.ndarray, scale: float, n_total: int) -> Dict[str, Any]:
    n = int(mask.sum())
    f = int(fail[mask].sum())
    return {
        "count": n,
        "share": round(n / n_total, 4) if n_total else 0.0,
        "failure_rate": round(f / n, 4) if n else None,
        "failures": f,
        "mae": float(err[mask].mean() * scale) if n else None,
        "rmse": float(np.sqrt(np.mean((err[mask] * scale) ** 2))) if n else None,
    }


def calibration_bands(
    scores: np.ndarray, scaled_err: np.ndarray, tol_scaled: float, scale: float,
    trust_threshold: Optional[float], reject_threshold: Optional[float],
) -> List[Dict[str, Any]]:
    fail = scaled_err > tol_scaled
    n = len(scores)
    trust = scores <= trust_threshold if trust_threshold is not None else np.zeros(n, bool)
    reject = scores > reject_threshold if reject_threshold is not None else np.zeros(n, bool)
    reject &= ~trust
    review = ~(trust | reject)
    out = []
    for name, m in (("TRUST", trust), ("REVIEW", review), ("REJECT", reject)):
        row = {"name": name, **_band(m, fail, scaled_err, scale, n), "estimate_source": "calibration_oof"}
        if name == "TRUST":
            row["failure_rate_upper95"] = wilson_upper(row["failures"], row["count"])
        out.append(row)
    return out


def tune_regression_thresholds(
    oof_scores: np.ndarray,
    scaled_err: np.ndarray,
    tol_scaled: float,
    target_scale: float = 1.0,
    target_risk: float = 0.05,
    min_coverage: float = 0.30,
    reject_min_error: float = 0.30,
    reject_min_rows: int = 30,
    reject_relative_factor: float = 2.0,
    reject_error_cap: float = 0.50,
    min_rows: int = MIN_THRESHOLD_ROWS,
) -> Dict[str, Any]:
    """
    Tune trust / reject thresholds on CALIBRATION out-of-fold estimates (scaled units).
    Rows without an out-of-fold estimate (NaN) are ignored.
    """
    s_all = np.asarray(oof_scores, dtype=float)
    e_all = np.asarray(scaled_err, dtype=float)
    ok = np.isfinite(s_all) & np.isfinite(e_all)
    s, e = s_all[ok], e_all[ok]
    n = len(s)
    fail = e > tol_scaled
    base = float(fail.mean()) if n else None

    def result(trust, reject, met, msg, cov=0.0, risk=None, status="ok"):
        return {
            "status": status,
            "trust_threshold": None if trust is None else float(trust),
            "reject_threshold": None if reject is None else float(reject),
            "error_tolerance_scaled": float(tol_scaled),
            "target_risk": float(target_risk),
            "min_coverage": float(min_coverage),
            "target_risk_met": bool(met),
            "message": msg,
            "calibration_rows": n,
            "calibration_base_failure_rate": None if base is None else round(base, 4),
            "calibration_coverage": round(float(cov), 4),
            "calibration_failure_rate": None if risk is None else round(float(risk), 4),
            "band_metadata": calibration_bands(s, e, tol_scaled, target_scale, trust, reject) if n else [],
        }

    if n < min_rows:
        return result(None, None, False, f"Only {n} calibration rows with error estimates (< {min_rows}); "
                      "TRUST and REJECT are disabled and every prediction goes to REVIEW.",
                      status="insufficient_calibration")

    order = np.argsort(s, kind="mergesort")
    s_sorted, fail_sorted = s[order], fail[order]
    cum_fail = np.cumsum(fail_sorted)
    uniq = np.unique(s_sorted)
    # rows accepted at threshold t = all rows with score <= t
    n_acc = np.searchsorted(s_sorted, uniq, side="right")
    risk = cum_fail[n_acc - 1] / n_acc
    cover = n_acc / n

    # ---- TRUST
    trust = None
    met = False
    feasible = np.where((cover >= min_coverage) & (risk <= target_risk))[0]
    if len(uniq) == 1:
        if risk[0] <= target_risk:
            trust, met = uniq[0], True
            msg = (f"Error estimates do not vary; the overall calibration failure rate {risk[0]:.1%} meets the "
                   f"{target_risk:.1%} target, so every supported prediction is TRUSTED.")
        else:
            msg = (f"Error estimates do not vary (no separation) and the overall calibration failure rate "
                   f"{risk[0]:.1%} exceeds the {target_risk:.1%} target; TRUST is disabled.")
    elif len(feasible):
        i = feasible[-1]  # largest threshold = maximum coverage that still meets the target
        trust, met = uniq[i], True
        msg = (f"Target failure rate {target_risk:.1%} met at estimated error <= {uniq[i]:.4g} (scaled) "
               f"with {cover[i]:.1%} calibration coverage.")
    else:
        cand = np.where(cover >= min_coverage)[0]
        i = cand[np.lexsort((-cover[cand], risk[cand]))[0]]  # lowest risk, then larger coverage
        if risk[i] < base:
            trust = uniq[i]
            msg = (f"Target failure rate {target_risk:.1%} not reachable with >= {min_coverage:.0%} coverage; best "
                   f"is {risk[i]:.1%} at {cover[i]:.1%} coverage (base {base:.1%}).")
        else:
            msg = (f"No threshold with >= {min_coverage:.0%} coverage lowers the failure rate below the base rate "
                   f"{base:.1%} (no meaningful separation); TRUST is disabled.")

    # ---- REJECT (scan from the largest high-score band downwards)
    reject = None
    required = max(reject_min_error, min(reject_error_cap, reject_relative_factor * base))
    lower = trust if trust is not None else -np.inf
    for t in uniq:  # ascending threshold -> descending REJECT band size
        if t < lower:
            continue
        band = s > t
        nb = int(band.sum())
        if nb < reject_min_rows:
            break
        if fail[band].mean() >= required:
            reject = t
            break
    # reject == trust is valid: every non-TRUST row is REJECT (perfect separation, empty REVIEW band).
    msg_r = (f" REJECT when estimated error > {reject:.4g} (scaled; band failure rate >= {required:.0%})."
             if reject is not None else
             f" No band of >= {reject_min_rows} rows reaches a {required:.0%} failure rate; REJECT is disabled.")

    if trust is not None:
        acc = s <= trust
        cov_t, risk_t = acc.mean(), fail[acc].mean()
    else:
        cov_t, risk_t = 0.0, None
    return result(trust, reject, met, msg + msg_r, cov_t, risk_t)


def regression_warnings(flags: List[Dict[str, Any]], decision: str) -> List[Dict[str, str]]:
    """Input/OOD warnings shared with classification plus regression-specific ones."""
    warnings = [w for w in generate_warnings(flags, None, decision=None, num_classes=0)]
    for f in flags or []:
        t = f.get("type")
        if t == "prediction_outside_training_range":
            warnings.append({"code": "EXTRAPOLATION",
                             "message": "the prediction lies outside the range of target values seen in training"})
        elif t == "model2_unavailable":
            warnings.append({"code": "ESTIMATE_UNAVAILABLE",
                             "message": "the reliability model could not score this record; a default error estimate was used"})
        elif t == "reliability_features_unavailable":
            warnings.append({"code": "ESTIMATE_UNAVAILABLE",
                             "message": "reliability signals could not be computed for this record"})
        elif t == "invalid_error_estimate":
            warnings.append({"code": "INVALID_ESTIMATE",
                             "message": "the error estimate was invalid (non-finite or negative)"})
    if decision in ("REVIEW", "REJECT") and not warnings:
        warnings.append({"code": "HIGH_ESTIMATED_ERROR",
                         "message": "the estimated error is above the level accepted for automatic use"})
    return warnings


class RegressionMitigationEngine:
    """Thresholds (scaled estimated error) + decision rules for regression. joblib-serializable."""

    task_type = "regression"

    def __init__(self, thresholds: Dict[str, Any], target_scale: float, error_tolerance: Dict[str, Any],
                 force_review_on_anomalies: bool = False):
        self.trust_threshold = thresholds["trust_threshold"]
        self.reject_threshold = thresholds["reject_threshold"]
        if (self.trust_threshold is not None and self.reject_threshold is not None
                and self.reject_threshold < self.trust_threshold):
            raise ValueError("reject_threshold must not be below trust_threshold (estimated error).")
        self.target_scale = float(target_scale)
        self.error_tolerance = dict(error_tolerance)
        self.force_review_on_anomalies = bool(force_review_on_anomalies)
        self.thresholds = dict(thresholds)

    def decide(self, est_scaled: float, flags: Optional[List[Dict[str, Any]]] = None) -> str:
        flags = flags or []
        if est_scaled is None or not np.isfinite(est_scaled) or est_scaled < 0:
            return "REVIEW"
        if self.trust_threshold is not None and est_scaled <= self.trust_threshold:
            decision = "TRUST"
        elif self.reject_threshold is not None and est_scaled > self.reject_threshold:
            decision = "REJECT"
        else:
            decision = "REVIEW"
        types = {f.get("type") for f in flags}
        if decision == "TRUST" and types & set(NEVER_TRUST_FLAGS):
            decision = "REVIEW"
        if decision == "TRUST" and self.force_review_on_anomalies and types & set(DOMAIN_ANOMALY_FLAGS):
            decision = "REVIEW"
        return decision

    def evaluate_record(self, est_scaled: float, flags: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        flags = list(flags or [])
        if est_scaled is None or not np.isfinite(est_scaled) or est_scaled < 0:
            flags.append({"type": "invalid_error_estimate"})
        decision = self.decide(est_scaled, flags)
        return {"decision": decision, "warnings": regression_warnings(flags, decision)}

    def explain(self, est_scaled: float, decision: str, warnings: List[Dict[str, str]]) -> Dict[str, Any]:
        s = self.target_scale
        est = "unavailable" if est_scaled is None or not np.isfinite(est_scaled) else f"{est_scaled * s:.4g}"
        tol = self.error_tolerance["value"]
        reasons = []
        if decision == "TRUST":
            reasons.append(f"Estimated absolute error {est} is at or below the trust level "
                           f"{self.trust_threshold * s:.4g}.")
            summary = "Accepted: the estimated error is low enough for automatic use."
        elif decision == "REJECT":
            reasons.append(f"Estimated absolute error {est} is above the reject level {self.reject_threshold * s:.4g}.")
            summary = "Rejected: ModelDoctor abstains because a large error is too likely."
        else:
            if self.trust_threshold is None:
                reasons.append("Automatic TRUST is disabled for this model (no reliable separation on calibration data).")
            else:
                reasons.append(f"Estimated absolute error {est} is above the trust level {self.trust_threshold * s:.4g}, "
                               "or the estimate is not supported for this input.")
            summary = "Needs human review: a person should verify this prediction before it is used."
        reasons.append(f"A prediction counts as a failure when its error exceeds {tol:.4g} "
                       f"({self.error_tolerance['source']}). The estimate is an expected error, not a guaranteed bound.")
        for w in warnings:
            if w["code"] != "HIGH_ESTIMATED_ERROR":
                m = w["message"].rstrip(".")
                reasons.append(m[:1].upper() + m[1:] + ".")
        return {"risk_level": RISK_LEVELS[decision], "explanation": summary, "reasons": reasons}

    def to_dict(self) -> Dict[str, Any]:
        s = self.target_scale
        d = dict(self.thresholds)
        d.update({
            "score": "estimated absolute error (lower is more reliable)",
            "trust_threshold_units": None if self.trust_threshold is None else self.trust_threshold * s,
            "reject_threshold_units": None if self.reject_threshold is None else self.reject_threshold * s,
            "error_tolerance": self.error_tolerance,
            "target_scale": s,
            "force_review_on_anomalies": self.force_review_on_anomalies,
        })
        return d
