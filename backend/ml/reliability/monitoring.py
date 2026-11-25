"""Phase C monitoring: compare estimated reliability with real outcomes once they arrive."""

from typing import Any, Dict, List, Optional

import numpy as np

ANOMALY_CODES = {
    "OUT_OF_RANGE", "UNSEEN_CATEGORY", "OOD_DISTANCE", "ANOMALOUS_INPUT",
    "MISSING_FIELD", "MISSING_FIELDS", "UNSUPPORTED_ESTIMATE",
}
DECISIONS = ["TRUST", "REVIEW", "REJECT"]
HISTOGRAM_BINS = 10


def _histogram(values: np.ndarray, decisions: np.ndarray) -> List[Dict[str, Any]]:
    """Counts per equal-width bin on [0, 1], split by decision (for stacked bar charts)."""
    edges = np.linspace(0.0, 1.0, HISTOGRAM_BINS + 1)
    idx = np.clip(np.digitize(values, edges[1:-1], right=False), 0, HISTOGRAM_BINS - 1) if len(values) else values
    out = []
    for b in range(HISTOGRAM_BINS):
        in_bin = idx == b
        row: Dict[str, Any] = {"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "lower": round(float(edges[b]), 2),
                               "count": int(in_bin.sum())}
        for d in DECISIONS:
            row[d] = int((in_bin & (decisions == d)).sum())
        out.append(row)
    return out


def _window_size(n: int) -> int:
    if n <= 40:
        return 10
    return int(np.ceil(n / 10))


def compute_monitoring(
    predictions: List[Dict[str, Any]],
    feedback: List[Dict[str, Any]],
    baseline: Optional[Dict[str, Any]] = None,
    min_labeled_for_alerts: int = 20,
) -> Dict[str, Any]:
    """
    Summarise served predictions and submitted outcomes.

    predictions: rows logged at prediction time (id, timestamp, prediction, confidence,
                 trustworthiness, decision, warning_codes)
    feedback:    rows {prediction_id, actual, timestamp}; the latest entry per id wins
    baseline:    test-split reference rates saved at training time
    """
    baseline = baseline or {}
    actual_by_id: Dict[str, str] = {}
    for fb in feedback:
        actual_by_id[str(fb["prediction_id"])] = str(fb["actual"])

    n = len(predictions)
    decisions = np.array([p["decision"] for p in predictions]) if n else np.array([])
    trust = np.array([float(p["trustworthiness"]) for p in predictions]) if n else np.array([])
    conf = np.array([float(p["confidence"]) for p in predictions]) if n else np.array([])
    labeled_mask = np.array([p["prediction_id"] in actual_by_id for p in predictions], dtype=bool)
    correct = np.array([
        1 if actual_by_id.get(p["prediction_id"]) == str(p["prediction"]) else 0 for p in predictions
    ], dtype=int)
    anomalous = np.array([
        any(code in ANOMALY_CODES for code in p.get("warning_codes", [])) for p in predictions
    ], dtype=bool)

    distribution = []
    for d in DECISIONS:
        mask = decisions == d
        lab = mask & labeled_mask
        distribution.append({
            "decision": d,
            "count": int(mask.sum()),
            "share": round(float(mask.mean()), 4) if n else 0.0,
            "labeled": int(lab.sum()),
            "mean_trustworthiness": round(float(trust[mask].mean()), 4) if mask.any() else None,
            "observed_accuracy": round(float(correct[lab].mean()), 4) if lab.any() else None,
        })

    # Coverage / reliability over time, in sequential windows of predictions
    timeline = []
    if n:
        w = _window_size(n)
        for start in range(0, n, w):
            sl = slice(start, min(start + w, n))
            lab = labeled_mask[sl]
            timeline.append({
                "window": len(timeline) + 1,
                "from_index": start + 1,
                "to_index": min(start + w, n),
                "start_time": predictions[start].get("timestamp"),
                "count": int(min(start + w, n) - start),
                "coverage": round(float(np.mean(decisions[sl] == "TRUST")), 4),
                "mean_trustworthiness": round(float(trust[sl].mean()), 4),
                "mean_confidence": round(float(conf[sl].mean()), 4),
                "anomaly_rate": round(float(anomalous[sl].mean()), 4),
                "labeled": int(lab.sum()),
                "observed_accuracy": round(float(correct[sl][lab].mean()), 4) if lab.any() else None,
            })

    # Observed vs estimated reliability (labeled predictions only)
    n_lab = int(labeled_mask.sum())
    observed_vs_estimated = {
        "labeled": n_lab,
        "mean_estimated_trustworthiness": round(float(trust[labeled_mask].mean()), 4) if n_lab else None,
        "mean_model_confidence": round(float(conf[labeled_mask].mean()), 4) if n_lab else None,
        "observed_accuracy": round(float(correct[labeled_mask].mean()), 4) if n_lab else None,
        "bins": [],
    }
    if n_lab:
        edges = np.linspace(0, 1, 6)
        t_lab, c_lab = trust[labeled_mask], correct[labeled_mask]
        for i in range(5):
            lo, hi = edges[i], edges[i + 1]
            m = (t_lab >= lo) & ((t_lab <= hi) if i == 4 else (t_lab < hi))
            if m.any():
                observed_vs_estimated["bins"].append({
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "count": int(m.sum()),
                    "mean_estimated": round(float(t_lab[m].mean()), 4),
                    "observed_accuracy": round(float(c_lab[m].mean()), 4),
                })

    # Alerts: degradation and drift
    alerts: List[Dict[str, str]] = []
    trust_lab = labeled_mask & (decisions == "TRUST") if n else labeled_mask
    expected_trust_acc = baseline.get("test_trust_accuracy")
    if trust_lab.sum() >= min_labeled_for_alerts and expected_trust_acc is not None:
        obs = float(correct[trust_lab].mean())
        if obs < expected_trust_acc - 0.10:
            alerts.append({
                "level": "critical",
                "code": "TRUST_DEGRADATION",
                "message": (
                    f"Accuracy of TRUST predictions is {obs * 100:.1f}% vs {expected_trust_acc * 100:.1f}% on the "
                    "test split. Reliability estimates may no longer hold; consider retraining."
                ),
            })
    if n_lab >= min_labeled_for_alerts:
        est = float(trust[labeled_mask].mean())
        obs = float(correct[labeled_mask].mean())
        if abs(est - obs) > 0.10:
            alerts.append({
                "level": "warning",
                "code": "MISCALIBRATION",
                "message": (
                    f"Estimated trustworthiness averages {est * 100:.1f}% but observed accuracy is "
                    f"{obs * 100:.1f}% on {n_lab} labeled predictions."
                ),
            })
    base_anom = baseline.get("test_anomaly_rate")
    if n >= min_labeled_for_alerts and base_anom is not None:
        recent = anomalous[-min(n, 100):]
        rate = float(recent.mean())
        if rate > max(2 * base_anom, base_anom + 0.10):
            alerts.append({
                "level": "warning",
                "code": "INPUT_DRIFT",
                "message": (
                    f"{rate * 100:.1f}% of recent inputs carry out-of-range / unseen / anomaly warnings vs "
                    f"{base_anom * 100:.1f}% on the test split. Incoming data may be drifting."
                ),
            })
    base_cov = baseline.get("test_coverage")
    if n >= min_labeled_for_alerts and base_cov is not None:
        cov = float(np.mean(decisions[-min(n, 100):] == "TRUST"))
        if cov < base_cov - 0.20:
            alerts.append({
                "level": "warning",
                "code": "COVERAGE_DROP",
                "message": f"Recent TRUST coverage is {cov * 100:.1f}% vs {base_cov * 100:.1f}% on the test split.",
            })

    predicted_classes: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    for p in predictions:
        predicted_classes[str(p["prediction"])] = predicted_classes.get(str(p["prediction"]), 0) + 1
        sources[str(p.get("source", "unknown"))] = sources.get(str(p.get("source", "unknown")), 0) + 1

    return {
        "total_predictions": int(n),
        "labeled_predictions": n_lab,
        "coverage": round(float(np.mean(decisions == "TRUST")), 4) if n else None,
        "mean_confidence": round(float(conf.mean()), 4) if n else None,
        "mean_trustworthiness": round(float(trust.mean()), 4) if n else None,
        "anomaly_rate": round(float(anomalous.mean()), 4) if n else None,
        "confidence_histogram": _histogram(conf, decisions) if n else [],
        "trustworthiness_histogram": _histogram(trust, decisions) if n else [],
        "predicted_class_distribution": [
            {"class": c, "count": k} for c, k in sorted(predicted_classes.items(), key=lambda kv: -kv[1])
        ],
        "source_distribution": sources,
        "first_prediction_at": predictions[0].get("timestamp") if n else None,
        "last_prediction_at": predictions[-1].get("timestamp") if n else None,
        "decision_distribution": distribution,
        "timeline": timeline,
        "observed_vs_estimated": observed_vs_estimated,
        "baseline": baseline,
        "alerts": alerts,
        "retraining_recommended": any(a["level"] == "critical" for a in alerts),
    }


def compute_regression_monitoring(
    predictions: List[Dict[str, Any]],
    feedback: List[Dict[str, Any]],
    baseline: Optional[Dict[str, Any]] = None,
    min_labeled_for_alerts: int = 20,
) -> Dict[str, Any]:
    """
    Regression monitoring. predictions: logged rows with a numeric `prediction`,
    `estimated_abs_error`, `decision`, `warning_codes`. feedback: {prediction_id, actual (number)}.
    Observed error = |actual - prediction|; failure = observed error > the training tolerance.
    """
    baseline = baseline or {}
    tol = baseline.get("error_tolerance")
    actual: Dict[str, float] = {}
    for fb in feedback:
        try:
            v = float(fb["actual"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            actual[str(fb["prediction_id"])] = v

    n = len(predictions)
    dec = np.array([p["decision"] for p in predictions]) if n else np.array([])
    pred = np.array([float(p["prediction"]) for p in predictions]) if n else np.array([])
    est = np.array([float(p["estimated_abs_error"]) for p in predictions]) if n else np.array([])
    lab = np.array([p["prediction_id"] in actual for p in predictions], dtype=bool)
    obs = np.array([abs(actual[p["prediction_id"]] - float(p["prediction"])) if p["prediction_id"] in actual
                    else np.nan for p in predictions]) if n else np.array([])
    fail = (obs > tol) if tol is not None else np.zeros(n, dtype=bool)
    anomalous = np.array([any(c in ANOMALY_CODES for c in p.get("warning_codes", [])) for p in predictions],
                         dtype=bool)

    def mean_or_none(v: np.ndarray, digits: int = 6) -> Optional[float]:
        return round(float(v.mean()), digits) if len(v) else None

    distribution = []
    for d in DECISIONS:
        m = dec == d
        ml = m & lab
        distribution.append({
            "decision": d,
            "count": int(m.sum()),
            "share": round(float(m.mean()), 4) if n else 0.0,
            "labeled": int(ml.sum()),
            "mean_estimated_error": mean_or_none(est[m]),
            "observed_mae": mean_or_none(obs[ml]),
            "observed_failure_rate": mean_or_none(fail[ml].astype(float), 4) if tol is not None else None,
        })

    timeline = []
    if n:
        w = _window_size(n)
        for start in range(0, n, w):
            sl = slice(start, min(start + w, n))
            l_ = lab[sl]
            timeline.append({
                "window": len(timeline) + 1, "from_index": start + 1, "to_index": min(start + w, n),
                "start_time": predictions[start].get("timestamp"), "count": int(min(start + w, n) - start),
                "coverage": round(float(np.mean(dec[sl] == "TRUST")), 4),
                "mean_estimated_error": float(est[sl].mean()),
                "anomaly_rate": round(float(anomalous[sl].mean()), 4),
                "labeled": int(l_.sum()),
                "observed_mae": float(obs[sl][l_].mean()) if l_.any() else None,
            })

    n_lab = int(lab.sum())
    ove = {
        "labeled": n_lab,
        "mean_estimated_abs_error": mean_or_none(est[lab]),
        "observed_mae": mean_or_none(obs[lab]),
        "observed_failure_rate": mean_or_none(fail[lab].astype(float), 4) if tol is not None else None,
        "error_tolerance": tol,
    }

    alerts: List[Dict[str, str]] = []
    trust_lab = lab & (dec == "TRUST") if n else lab
    base_fail = baseline.get("test_trust_failure_rate")
    if tol is not None and base_fail is not None and trust_lab.sum() >= min_labeled_for_alerts:
        rate = float(fail[trust_lab].mean())
        if rate > base_fail + 0.10:
            alerts.append({"level": "critical", "code": "TRUST_DEGRADATION",
                           "message": f"{rate * 100:.1f}% of labeled TRUST predictions exceed the error tolerance "
                                      f"vs {base_fail * 100:.1f}% on the test split; consider retraining."})
    if n_lab >= min_labeled_for_alerts:
        e_mean, o_mean = float(est[lab].mean()), float(obs[lab].mean())
        if e_mean > 0 and o_mean > 1.5 * e_mean:
            alerts.append({"level": "warning", "code": "ERROR_UNDERESTIMATED",
                           "message": f"Observed mean absolute error {o_mean:.4g} is more than 1.5x the estimated "
                                      f"{e_mean:.4g} on {n_lab} labeled predictions."})
    base_anom = baseline.get("test_anomaly_rate")
    if n >= min_labeled_for_alerts and base_anom is not None:
        rate = float(anomalous[-min(n, 100):].mean())
        if rate > max(2 * base_anom, base_anom + 0.10):
            alerts.append({"level": "warning", "code": "INPUT_DRIFT",
                           "message": f"{rate * 100:.1f}% of recent inputs carry out-of-range / unseen / anomaly "
                                      f"warnings vs {base_anom * 100:.1f}% on the test split."})
    base_cov = baseline.get("test_coverage")
    if n >= min_labeled_for_alerts and base_cov is not None:
        cov = float(np.mean(dec[-min(n, 100):] == "TRUST"))
        if cov < base_cov - 0.20:
            alerts.append({"level": "warning", "code": "COVERAGE_DROP",
                           "message": f"Recent TRUST coverage is {cov * 100:.1f}% vs {base_cov * 100:.1f}% on the test split."})

    sources: Dict[str, int] = {}
    for p in predictions:
        k = str(p.get("source", "unknown"))
        sources[k] = sources.get(k, 0) + 1

    return {
        "task_type": "regression",
        "total_predictions": int(n),
        "labeled_predictions": n_lab,
        "coverage": round(float(np.mean(dec == "TRUST")), 4) if n else None,
        "mean_estimated_error": mean_or_none(est),
        "anomaly_rate": round(float(anomalous.mean()), 4) if n else None,
        "prediction_summary": ({"min": float(pred.min()), "median": float(np.median(pred)), "max": float(pred.max())}
                               if n else None),
        "source_distribution": sources,
        "first_prediction_at": predictions[0].get("timestamp") if n else None,
        "last_prediction_at": predictions[-1].get("timestamp") if n else None,
        "decision_distribution": distribution,
        "timeline": timeline,
        "observed_vs_estimated": ove,
        "baseline": baseline,
        "alerts": alerts,
        "retraining_recommended": any(a["level"] == "critical" for a in alerts),
    }
