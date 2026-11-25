"""Phase 4 tests: regression thresholds, TRUST / REVIEW / REJECT decisions, matched-coverage
evaluation and fallbacks. Synthetic data only."""

import functools
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import ml.pipeline as pl
from ml.inference import load_artifacts, save_artifacts
from ml.pipeline import TrainingConfig, predict_regression_records, train_modeldoctor
from ml.reliability.evaluation_regression import (
    evaluate_regression_mitigation,
    matched_decisions,
    regression_risk_coverage,
)
from ml.reliability.mitigation_regression import (
    RegressionMitigationEngine,
    resolve_error_tolerance,
    tune_regression_thresholds,
    wilson_upper,
)
from ml.reliability.model2_regression import train_regression_model2
from ml.task import REGRESSION

warnings.filterwarnings("ignore")


def hetero_df(n=1500, seed=0, scale=1.0):
    rng = np.random.RandomState(seed)
    x1, x2 = rng.randn(n), rng.uniform(-3, 3, n)
    region = rng.choice(["north", "south", "online"], n)
    y = (2 * x1 + x2 ** 2 + rng.randn(n) * np.where(region == "online", 3.0, 0.4)) * scale
    return pd.DataFrame({"x1": x1, "x2": x2, "region": region, "y": y})


def cfg(**kw):
    return TrainingConfig(task_type=REGRESSION, fast=True, **kw)


@pytest.fixture(scope="module")
def trained():
    return train_modeldoctor(hetero_df(), "y", cfg())


def synthetic_oof(n=400, seed=0):
    """Estimated error = true expected error; observed errors scatter around it."""
    rng = np.random.RandomState(seed)
    est = rng.uniform(0.05, 1.0, n)
    err = est * rng.exponential(1.0, n)
    return est, err


# ---------------------------------------------------------------------------
# Tolerance and threshold derivation
# ---------------------------------------------------------------------------

def test_error_tolerance_from_config_or_train_oof_residuals():
    r = np.r_[np.arange(1.0, 101.0), np.nan]
    t = resolve_error_tolerance(r, target_scale=2.0, quantile=0.9)
    assert t["value"] == pytest.approx(np.quantile(np.arange(1.0, 101.0), 0.9))
    assert t["scaled"] == pytest.approx(t["value"] / 2.0) and t["source"].startswith("train_oof")
    c = resolve_error_tolerance(r, target_scale=2.0, error_tolerance=7.0)
    assert c == {"value": 7.0, "scaled": 3.5, "source": "configured"}
    z = resolve_error_tolerance(np.r_[np.zeros(95), np.ones(5)], 1.0)  # q90 = 0 -> smallest positive
    assert z["value"] == 1.0
    for bad in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError):
            resolve_error_tolerance(r, 1.0, error_tolerance=bad)


def test_trust_threshold_is_maximum_coverage_meeting_target_risk():
    # 10 rows sorted by score; failures at ranks 6, 9, 10 (1-based).
    scores = np.arange(1.0, 11.0) / 10
    err = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 1], dtype=float)
    res = tune_regression_thresholds(scores, err, tol_scaled=0.5, target_risk=0.2, min_coverage=0.3,
                                     reject_min_rows=2, min_rows=5)
    # accept <=0.8: 1/8 = 12.5% <= 20%; <=0.9: 2/9 = 22% > 20%  -> trust = 0.8
    assert res["trust_threshold"] == pytest.approx(0.8) and res["target_risk_met"]
    assert res["calibration_coverage"] == pytest.approx(0.8)
    assert res["calibration_failure_rate"] == pytest.approx(0.125)


def test_reject_band_requires_rows_and_failure_rate():
    est, err = synthetic_oof()
    tol = np.quantile(err, 0.9)
    res = tune_regression_thresholds(est, err, tol, target_risk=0.05, min_coverage=0.3)
    bands = {b["name"]: b for b in res["band_metadata"]}
    if res["reject_threshold"] is not None:
        assert res["reject_threshold"] >= (res["trust_threshold"] or -np.inf)  # equal = empty REVIEW band
        assert bands["REJECT"]["count"] >= 30
        base = res["calibration_base_failure_rate"]
        assert bands["REJECT"]["failure_rate"] >= max(0.30, min(0.5, 2 * base))
    assert sum(b["count"] for b in bands.values()) == len(est)
    assert bands["TRUST"]["failure_rate_upper95"] >= bands["TRUST"]["failure_rate"]


def test_unreachable_target_and_no_separation():
    rng = np.random.RandomState(1)
    est = rng.rand(300)
    err = rng.exponential(size=300)  # unrelated to the estimate
    res = tune_regression_thresholds(est, err, tol_scaled=np.quantile(err, 0.5), target_risk=0.05)
    assert not res["target_risk_met"]
    if res["trust_threshold"] is not None:  # only kept when genuinely better than accepting everything
        assert res["calibration_failure_rate"] < res["calibration_base_failure_rate"]


def test_constant_scores_and_small_calibration_sets():
    err = np.r_[np.zeros(95), np.full(5, 10.0)]
    ok = tune_regression_thresholds(np.full(100, 0.3), err, tol_scaled=1.0, target_risk=0.05)
    assert ok["trust_threshold"] == pytest.approx(0.3) and ok["target_risk_met"]  # 5% <= 5%
    bad = tune_regression_thresholds(np.full(100, 0.3), np.r_[np.zeros(80), np.full(20, 10.0)], 1.0)
    assert bad["trust_threshold"] is None and "no separation" in bad["message"]
    tiny = tune_regression_thresholds(np.arange(20.0), np.arange(20.0), 5.0)
    assert tiny["status"] == "insufficient_calibration"
    assert tiny["trust_threshold"] is None and tiny["reject_threshold"] is None
    nan_rows = tune_regression_thresholds(np.r_[np.full(30, np.nan), np.arange(40.0)], np.arange(70.0), 5.0)
    assert nan_rows["calibration_rows"] == 40


def test_wilson_upper_bound():
    assert wilson_upper(0, 0) is None
    assert 0 < wilson_upper(0, 40) < 0.1
    assert wilson_upper(5, 100) > 0.05


# ---------------------------------------------------------------------------
# Decision boundaries and overrides
# ---------------------------------------------------------------------------

def engine(trust=0.5, reject=1.0, force=False):
    th = {"trust_threshold": trust, "reject_threshold": reject}
    return RegressionMitigationEngine(th, 2.0, {"value": 2.0, "scaled": 1.0, "source": "configured"}, force)


@pytest.mark.parametrize("est, expected", [
    (0.0, "TRUST"), (0.5, "TRUST"), (0.5000001, "REVIEW"), (1.0, "REVIEW"), (1.0000001, "REJECT"),
])
def test_decision_boundaries(est, expected):
    assert engine().decide(est) == expected


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, None])
def test_invalid_estimates_go_to_review_with_a_warning(bad):
    out = engine().evaluate_record(bad, [])
    assert out["decision"] == "REVIEW"
    assert "INVALID_ESTIMATE" in {w["code"] for w in out["warnings"]}


@pytest.mark.parametrize("flag", ["outside_reliability_support", "model2_unavailable",
                                  "reliability_features_unavailable"])
def test_never_trust_overrides(flag):
    assert engine().decide(0.1, [{"type": flag}]) == "REVIEW"


def test_disabled_bands_and_anomaly_override():
    e = engine(trust=None, reject=None)
    assert {e.decide(x) for x in (0.0, 5.0, 100.0)} == {"REVIEW"}
    assert engine(force=True).decide(0.1, [{"type": "unseen_category", "feature": "c"}]) == "REVIEW"
    assert engine(force=False).decide(0.1, [{"type": "unseen_category", "feature": "c"}]) == "TRUST"
    with pytest.raises(ValueError):
        engine(trust=1.0, reject=0.5)


def test_explanation_never_claims_a_probability_or_a_bound():
    e = engine()
    out = e.evaluate_record(0.2, [])
    text = " ".join(e.explain(0.2, out["decision"], out["warnings"])["reasons"])
    assert "not a guaranteed bound" in text and "probability" not in text.lower()
    assert e.explain(0.2, "TRUST", [])["risk_level"] == "LOW"


# ---------------------------------------------------------------------------
# Coverage / risk / matched coverage
# ---------------------------------------------------------------------------

def test_matched_coverage_and_risk_hand_checked():
    y = np.zeros(6)
    pred = np.array([0.1, 0.2, 3.0, 0.3, 4.0, 5.0])  # abs errors
    est = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    dec = ["TRUST", "TRUST", "REVIEW", "REVIEW", "REJECT", "REJECT"]
    base = {"reverse": -est}
    ev = evaluate_regression_mitigation(y, pred, est, dec, error_tolerance=1.0, baselines=base)
    md = {b["decision"]: b for b in ev["modeldoctor_bands"]}
    assert md["TRUST"]["mae"] == pytest.approx(0.15) and md["TRUST"]["failure_rate"] == 0.0
    assert md["REJECT"]["failure_rate"] == 1.0 and ev["overall_coverage"] == pytest.approx(2 / 6, abs=1e-4)
    assert ev["no_filtering"]["failure_rate"] == pytest.approx(0.5)
    bl = {b["decision"]: b for b in ev["matched_coverage_baselines"][0]["bands"]}
    assert [bl[d]["count"] for d in ("TRUST", "REVIEW", "REJECT")] == [2, 2, 2]
    assert bl["TRUST"]["mae"] == pytest.approx(4.5)  # reverse order trusts the two worst rows
    assert ev["matched_coverage_baselines"][0]["trust_failures_reduction"] == 2
    np.testing.assert_array_equal(matched_decisions(est, 2, 2), dec)


def test_risk_coverage_curve_endpoints():
    err = np.array([1.0, 2.0, 3.0, 4.0])
    rc = regression_risk_coverage(err, err > 2.5, score=err)
    assert rc["curve"][0]["selective_mae"] == 1.0 and rc["curve"][-1]["selective_mae"] == 2.5
    assert rc["aurc_mae"] == pytest.approx(np.mean([1.0, 1.5, 2.0, 2.5]))


# ---------------------------------------------------------------------------
# End-to-end: calibration-only thresholds, fallbacks, persistence
# ---------------------------------------------------------------------------

def test_end_to_end_report_and_consistency(trained, tmp_path):
    art, rep = trained
    json.dumps(rep)  # serializable
    ev = rep["mitigation_evaluation"]
    assert sum(b["count"] for b in ev["modeldoctor_bands"]) == ev["n_test"]
    assert {c["baseline"] for c in ev["matched_coverage_baselines"]} == {
        "Model 1 bootstrap spread", "Local training error (kNN out-of-fold residual)",
        "Distance to training data (kNN)"}
    save_artifacts(tmp_path, art)
    loaded = load_artifacts(tmp_path)
    X = hetero_df(seed=5).drop(columns="y").head(30)
    a, b = predict_regression_records(art, X), predict_regression_records(loaded, X)
    assert [r["decision"] for r in a] == [r["decision"] for r in b]
    for r in a:
        assert {"prediction", "estimated_abs_error", "decision", "warnings", "risk_level", "explanation",
                "reasons", "reliability_mode"} <= set(r)
        assert "class_probabilities" not in r and "confidence" not in r


def test_thresholds_do_not_depend_on_test_labels():
    df = hetero_df(n=1200)
    from ml.data.splitter import split_regression_data

    test_idx = split_regression_data(df, "y", random_state=42).X_test_raw.index
    altered = df.copy()
    altered.loc[test_idx, "y"] = altered.loc[test_idx, "y"] * -7 + 1000  # wreck test labels only
    _, r1 = train_modeldoctor(df, "y", cfg())
    _, r2 = train_modeldoctor(altered, "y", cfg())
    keys = ("trust_threshold", "reject_threshold", "error_tolerance_scaled")
    assert {k: r1["thresholds"][k] for k in keys} == {k: r2["thresholds"][k] for k in keys}
    assert r1["mitigation_evaluation"] != r2["mitigation_evaluation"]  # only the evaluation changes


def test_scale_invariance_with_configured_tolerance():
    base = hetero_df(n=900)
    _, r1 = train_modeldoctor(base, "y", cfg(force_model="Ridge Regression", error_tolerance=2.0))
    _, r2 = train_modeldoctor(base.assign(y=base["y"] * 1e6), "y",
                              cfg(force_model="Ridge Regression", error_tolerance=2.0e6))
    d1 = [b["count"] for b in r1["mitigation_evaluation"]["modeldoctor_bands"]]
    d2 = [b["count"] for b in r2["mitigation_evaluation"]["modeldoctor_bands"]]
    assert d1 == d2
    assert r2["thresholds"]["trust_threshold_units"] == pytest.approx(r1["thresholds"]["trust_threshold_units"] * 1e6,
                                                                      rel=1e-5)


def test_fallback_mode_runs_end_to_end_and_never_trusts_unsupported_inputs(monkeypatch):
    monkeypatch.setattr(pl, "train_regression_model2",
                        functools.partial(train_regression_model2, min_rows=10 ** 6))
    df = hetero_df(n=800)
    art, rep = train_modeldoctor(df, "y", cfg())
    assert art["reliability_mode"] == "constant_fallback" and art["model2"] is None
    th = rep["thresholds"]
    # constant scores: TRUST is all-or-nothing depending on the overall calibration failure rate
    assert th["trust_threshold"] is None or th["target_risk_met"]
    far = df.drop(columns="y").head(1).assign(x1=1e6)
    assert predict_regression_records(art, far)[0]["decision"] != "TRUST"


def test_model2_and_extractor_failures_never_trust(trained, monkeypatch):
    art, _ = trained
    X = hetero_df(seed=8).drop(columns="y").head(20)
    clean = predict_regression_records(art, X)
    monkeypatch.setattr(type(art["model2"]), "predict_scaled_error",
                        lambda self, f: (_ for _ in ()).throw(RuntimeError("down")))
    broken = predict_regression_records(art, X)
    assert all(r["decision"] != "TRUST" for r in broken)
    np.testing.assert_allclose([r["prediction"] for r in broken], [r["prediction"] for r in clean], rtol=1e-12)
    monkeypatch.setattr(type(art["extractor"]), "transform_with_flags",
                        lambda self, X, m: (_ for _ in ()).throw(RuntimeError("down")))
    no_feats = predict_regression_records(art, X)
    assert all(r["decision"] != "TRUST" for r in no_feats)
    assert all("ESTIMATE_UNAVAILABLE" in {w["code"] for w in r["warnings"]} for r in no_feats)


def test_small_dataset_runs_end_to_end():
    art, rep = train_modeldoctor(hetero_df(n=500), "y", cfg())
    assert rep["mitigation_evaluation"]["n_test"] == 100
    assert rep["thresholds"]["status"] in ("ok", "insufficient_calibration")
