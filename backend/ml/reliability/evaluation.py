"""Test-split evaluation of ModelDoctor against the confidence-threshold baseline.

CRITICAL RULES:
- Everything here is computed on the TEST split only, AFTER all models and
  thresholds were fixed. Nothing in this module is used to tune anything.
- Every comparison includes the "No filtering" reference and the
  confidence-threshold baseline evaluated at the SAME coverage as ModelDoctor
  (fair comparison, spec section 8).
- All outputs are JSON-serializable (lists / dicts of Python floats).

CONVENTIONS (see METRIC_DEFINITIONS, which is also returned in the report):
- correct = 1 when Model 1's predicted class equals the true class, else 0.
- Error detection treats "Model 1 is WRONG" as the POSITIVE class.
- Scores: ModelDoctor uses trustworthiness = P(correct) from Model 2 (or Model 1
  confidence in fallback mode); the baseline uses Model 1's max predicted probability.
  Risk / error scores are 1 - score, so a higher value means more likely wrong.
- Both methods are always evaluated on exactly the same test rows.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.reliability.model2 import compute_brier_score, compute_ece


def _safe_auc(is_incorrect: np.ndarray, risk_score: np.ndarray) -> Optional[float]:
    """AUROC for ranking incorrect predictions above correct ones (None if undefined)."""
    if len(np.unique(is_incorrect)) < 2:
        return None
    return round(float(roc_auc_score(is_incorrect, risk_score)), 4)


def _safe_auprc(is_incorrect: np.ndarray, risk_score: np.ndarray) -> Optional[float]:
    """Average precision for detecting incorrect predictions (None if undefined)."""
    if len(np.unique(is_incorrect)) < 2:
        return None
    return round(float(average_precision_score(is_incorrect, risk_score)), 4)


METRIC_DEFINITIONS: Dict[str, str] = {
    "positive_class": "Model 1 prediction is incorrect (error). correct = 1[argmax P(y|x) == y].",
    "error_probability": "P(error) = 1 - trustworthiness for ModelDoctor; 1 - max_prob for the baseline.",
    "auroc": "ROC AUC of P(error) for separating incorrect (positive) from correct predictions. "
             "None when the test split has only one outcome.",
    "auprc": "Average precision of P(error) for incorrect predictions (positive). Random reference = error rate.",
    "brier": "Mean of (P(correct) - correct)^2; identical to the Brier score of P(error) vs incorrect.",
    "ece": "Expected Calibration Error: 10 equal-width bins on [0, 1] (last bin closed), "
           "sum_b |B_b|/N * |mean predicted - observed rate| of the stated probability.",
    "selective_risk": "Error rate among TRUST (accepted) predictions; 0.0 is reported when nothing is accepted "
                      "(see 'accepted').",
    "matched_coverage": "The confidence baseline accepts exactly as many test rows as ModelDoctor TRUSTs "
                        "(the highest-confidence ones) and rejects exactly as many as ModelDoctor REJECTs "
                        "(the lowest-confidence ones).",
    "error_detection": "A prediction is FLAGGED when it is not TRUSTED (REVIEW or REJECT). precision = "
                       "errors among flagged / flagged, recall = flagged errors / all errors. None when undefined "
                       "(nothing flagged, or no errors).",
}

MIN_ERRORS_FOR_STABLE_METRICS = 30


def selective_metrics(correct: np.ndarray, accepted: np.ndarray) -> Dict[str, Any]:
    """Coverage and selective risk (error among accepted) for a boolean acceptance mask."""
    correct = np.asarray(correct, dtype=int)
    accepted = np.asarray(accepted, dtype=bool)
    n = len(correct)
    n_acc = int(accepted.sum())
    risk = float(np.mean(1 - correct[accepted])) if n_acc > 0 else 0.0
    return {
        "coverage": round(n_acc / n, 4) if n else 0.0,
        "selective_risk": round(risk, 4),
        "accepted": n_acc,
        "total": int(n),
    }


def accept_top_k(score: np.ndarray, k: int) -> np.ndarray:
    """Accept exactly the k rows with the highest score (stable tie-breaking)."""
    score = np.asarray(score, dtype=float)
    mask = np.zeros(len(score), dtype=bool)
    if k <= 0:
        return mask
    order = np.argsort(-score, kind="mergesort")
    mask[order[:k]] = True
    return mask


def error_detection_metrics(correct: np.ndarray, flagged: np.ndarray) -> Dict[str, Any]:
    """
    Precision / recall of flagging (not accepting) a prediction, as a detector of Model 1 ERRORS.

    Positive class = Model 1 is incorrect. precision = TP / flagged, recall = TP / errors.
    Undefined values (nothing flagged -> precision; no errors -> recall) are None, not 0.
    """
    correct = np.asarray(correct, dtype=int)
    flagged = np.asarray(flagged, dtype=bool)
    incorrect = correct == 0
    tp = int(np.sum(flagged & incorrect))
    fp = int(np.sum(flagged & ~incorrect))
    fn = int(np.sum(~flagged & incorrect))
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def rnd(v):
        return None if v is None else round(float(v), 4)

    return {
        "positive_class": "model1_incorrect",
        "flag_rule": "decision != TRUST",
        "precision": rnd(precision),
        "recall": rnd(recall),
        "f1": rnd(f1),
        "flagged": int(flagged.sum()),
        "errors_caught": tp,
        "false_alarms": fp,
        "errors_missed": fn,
        "errors_total": int(incorrect.sum()),
    }


def error_probability_metrics(correct: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    """AUROC / AUPRC / Brier / ECE of P(error) = 1 - score, with incorrect as the positive class."""
    correct = np.asarray(correct, dtype=int)
    p_error = 1.0 - np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    is_incorrect = 1 - correct
    return {
        "auroc": _safe_auc(is_incorrect, p_error),
        "auprc": _safe_auprc(is_incorrect, p_error),
        "brier": compute_brier_score(is_incorrect, p_error),
        "ece": compute_ece(is_incorrect, p_error),
        "mean_predicted_error": round(float(p_error.mean()), 4) if len(p_error) else None,
        "observed_error_rate": round(float(is_incorrect.mean()), 4) if len(p_error) else None,
    }


def _band_rows(correct: np.ndarray, masks: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    n = len(correct)
    rows = []
    for name in ["TRUST", "REVIEW", "REJECT"]:
        m = masks[name]
        cnt = int(m.sum())
        errs = int(np.sum(1 - correct[m]))
        rows.append({
            "decision": name,
            "count": cnt,
            "coverage": round(cnt / n, 4) if n else 0.0,
            "errors": errs,
            "risk": round(errs / cnt, 4) if cnt else None,
        })
    return rows


def matched_band_comparison(
    correct: np.ndarray,
    confidence: np.ndarray,
    decisions: np.ndarray,
) -> Dict[str, Any]:
    """
    TRUST / REVIEW / REJECT counts and risks for ModelDoctor and for a confidence baseline
    whose bands have exactly the same sizes (TRUST = top-k confidence, REJECT = bottom-r).
    """
    correct = np.asarray(correct, dtype=int)
    decisions = np.asarray(decisions)
    n = len(correct)
    md = {d: decisions == d for d in ["TRUST", "REVIEW", "REJECT"]}
    k, r = int(md["TRUST"].sum()), int(md["REJECT"].sum())
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="mergesort")
    base_trust = np.zeros(n, dtype=bool)
    base_reject = np.zeros(n, dtype=bool)
    base_trust[order[:k]] = True
    if r > 0:
        base_reject[order[n - r:]] = True
    base = {"TRUST": base_trust, "REJECT": base_reject, "REVIEW": ~(base_trust | base_reject)}
    md_rows, base_rows = _band_rows(correct, md), _band_rows(correct, base)
    md_err, base_err = md_rows[0]["errors"], base_rows[0]["errors"]
    return {
        "modeldoctor": md_rows,
        "confidence_baseline": base_rows,
        "trust_errors_modeldoctor": md_err,
        "trust_errors_baseline": base_err,
        "trust_error_reduction": base_err - md_err,
        "trust_error_reduction_relative": round((base_err - md_err) / base_err, 4) if base_err else None,
        "reject_errors_modeldoctor": md_rows[2]["errors"],
        "reject_errors_baseline": base_rows[2]["errors"],
    }


def risk_coverage_curve(
    correct: np.ndarray,
    score: np.ndarray,
    n_points: int = 50,
) -> List[Dict[str, float]]:
    """
    Selective risk at evenly spaced coverage levels, accepting the highest-scored rows first.
    """
    correct = np.asarray(correct, dtype=int)
    n = len(correct)
    if n == 0:
        return []
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    cum_errors = np.cumsum(1 - correct[order])
    ks = np.unique(np.clip(np.round(np.linspace(1, n, n_points)).astype(int), 1, n))
    return [
        {
            "coverage": round(float(k / n), 4),
            "selective_risk": round(float(cum_errors[k - 1] / k), 4),
        }
        for k in ks
    ]


def area_under_risk_coverage(correct: np.ndarray, score: np.ndarray) -> float:
    """AURC: mean selective risk over all coverage levels (lower is better)."""
    correct = np.asarray(correct, dtype=int)
    n = len(correct)
    if n == 0:
        return 0.0
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    cum_errors = np.cumsum(1 - correct[order])
    risks = cum_errors / np.arange(1, n + 1)
    return round(float(np.mean(risks)), 4)


def calibration_curve(
    correct: np.ndarray,
    prob: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    """Equal-width reliability diagram bins: mean predicted P(correct) vs observed accuracy."""
    correct = np.asarray(correct, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (prob >= lo) & ((prob <= hi) if i == n_bins - 1 else (prob < hi))
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        out.append({
            "bin_low": round(float(lo), 2),
            "bin_high": round(float(hi), 2),
            "count": cnt,
            "mean_predicted": round(float(prob[mask].mean()), 4),
            "observed_accuracy": round(float(correct[mask].mean()), 4),
        })
    return out


def _delta_phrase(risk_delta: float) -> str:
    """risk_delta = baseline risk - ModelDoctor risk (positive = ModelDoctor better)."""
    if abs(risk_delta) < 0.0005:
        return "TRUST error equals the baseline"
    word = "lower" if risk_delta > 0 else "higher"
    return f"TRUST error is {abs(risk_delta) * 100:.1f} points {word} than the baseline"


def evaluate_system(
    correct: np.ndarray,
    confidence: np.ndarray,
    trustworthiness: np.ndarray,
    decisions: List[str],
    reliability_mode: str = "model2",
    model2_winner: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full test-split evaluation: ModelDoctor (TRUST decisions from thresholds tuned on the
    calibration split) vs the confidence baseline at the same coverage, plus curves.

    Parameters
    ----------
    correct : array of int
        1 if Model 1's test prediction was correct, else 0.
    confidence : array of float
        Model 1 max predicted probability (baseline score).
    trustworthiness : array of float
        ModelDoctor P(correct) estimates.
    decisions : list[str]
        TRUST / REVIEW / REJECT for each test row.
    reliability_mode : str
        "model2" or "confidence_fallback" (Model 2 had insufficient signal).
    model2_winner : str, optional
        Name of the selected Model 2 candidate (used only to word the verdict).
    """
    correct = np.asarray(correct, dtype=int)
    confidence = np.asarray(confidence, dtype=float)
    trust = np.asarray(trustworthiness, dtype=float)
    decisions_arr = np.asarray(decisions)
    n = len(correct)
    is_incorrect = 1 - correct

    md_accepted = decisions_arr == "TRUST"
    n_acc = int(md_accepted.sum())
    baseline_accepted = accept_top_k(confidence, n_acc)

    no_filter = selective_metrics(correct, np.ones(n, dtype=bool))
    md = selective_metrics(correct, md_accepted)
    base = selective_metrics(correct, baseline_accepted)

    comparison = [
        {"method": "No filtering", **no_filter},
        {"method": "Confidence baseline", **base},
        {"method": "ModelDoctor", **md},
    ]

    auroc = {
        "confidence_baseline": _safe_auc(is_incorrect, 1.0 - confidence),
        "modeldoctor": _safe_auc(is_incorrect, 1.0 - trust),
    }
    auprc = {
        "confidence_baseline": _safe_auprc(is_incorrect, 1.0 - confidence),
        "modeldoctor": _safe_auprc(is_incorrect, 1.0 - trust),
        "random": round(float(is_incorrect.mean()), 4) if n else 0.0,
    }
    aurc = {
        "confidence_baseline": area_under_risk_coverage(correct, confidence),
        "modeldoctor": area_under_risk_coverage(correct, trust),
    }

    decision_distribution = []
    for name in ["TRUST", "REVIEW", "REJECT"]:
        mask = decisions_arr == name
        cnt = int(mask.sum())
        decision_distribution.append({
            "decision": name,
            "count": cnt,
            "share": round(cnt / n, 4) if n else 0.0,
            "observed_error_rate": round(float(is_incorrect[mask].mean()), 4) if cnt else None,
            "mean_trustworthiness": round(float(trust[mask].mean()), 4) if cnt else None,
        })

    risk_delta = round(base["selective_risk"] - md["selective_risk"], 4)
    if n_acc == 0:
        verdict = "ModelDoctor accepted no test predictions; no comparison possible at zero coverage."
    elif reliability_mode != "model2":
        verdict = (
            "Model 2 had insufficient error signal, so ModelDoctor fell back to Model 1 confidence: its "
            "score equals the baseline, and decisions differ only where inputs outside the calibration "
            f"range were sent to REVIEW ({_delta_phrase(risk_delta)} at {md['coverage'] * 100:.1f}% coverage)."
        )
    elif model2_winner == "Calibrated confidence":
        verdict = (
            "Model 2 selected the calibrated-confidence candidate (the extra reliability features did not "
            "improve out-of-fold Brier on the calibration split), so its ranking equals the confidence "
            f"baseline; {_delta_phrase(risk_delta)} at {md['coverage'] * 100:.1f}% coverage."
        )
    elif risk_delta > 0.005:
        verdict = (
            f"ModelDoctor lowers error among accepted predictions by {risk_delta * 100:.1f} points "
            f"vs the confidence baseline at the same {md['coverage'] * 100:.1f}% coverage."
        )
    elif risk_delta < -0.005:
        verdict = (
            f"ModelDoctor is worse than the confidence baseline by {-risk_delta * 100:.1f} points "
            f"at the same {md['coverage'] * 100:.1f}% coverage on this dataset."
        )
    else:
        verdict = "ModelDoctor matches the confidence baseline at the same coverage on this dataset."

    n_errors = int(is_incorrect.sum())
    reliability_note = None
    if n_errors < MIN_ERRORS_FOR_STABLE_METRICS:
        reliability_note = (
            f"Only {n_errors} test errors: error-detection metrics (AUROC, AUPRC, precision/recall, risk) "
            "are highly uncertain and should not be used to draw conclusions."
        )

    return {
        "n_test": int(n),
        "n_test_errors": n_errors,
        "metrics_reliability_note": reliability_note,
        "metric_definitions": METRIC_DEFINITIONS,
        "overall_error_rate": no_filter["selective_risk"],
        "comparison": comparison,
        "selective_risk_improvement": risk_delta,
        "verdict": verdict,
        "auroc": auroc,
        "auprc": auprc,
        "aurc": aurc,
        "error_detection": {
            "modeldoctor": error_detection_metrics(correct, ~md_accepted),
            "confidence_baseline": error_detection_metrics(correct, ~baseline_accepted),
        },
        "calibration": {
            "modeldoctor": {
                "ece": compute_ece(correct, trust),
                "brier": compute_brier_score(correct, trust),
            },
            "confidence_baseline": {
                "ece": compute_ece(correct, confidence),
                "brier": compute_brier_score(correct, confidence),
            },
        },
        "error_probability": {
            "modeldoctor": error_probability_metrics(correct, trust),
            "confidence_baseline": error_probability_metrics(correct, confidence),
        },
        "matched_bands": matched_band_comparison(correct, confidence, decisions_arr),
        "decision_distribution": decision_distribution,
        "risk_coverage_curve": {
            "modeldoctor": risk_coverage_curve(correct, trust),
            "baseline": risk_coverage_curve(correct, confidence),
        },
        "calibration_curve": {
            "modeldoctor": calibration_curve(correct, trust),
            "baseline": calibration_curve(correct, confidence),
        },
    }
