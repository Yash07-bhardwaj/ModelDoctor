"""Phase 5 tests: regression evaluation metrics, leakage audit, reproducibility and the
experiment workflow. Hand-checkable synthetic examples."""

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.inference import load_artifacts, save_artifacts
from ml.pipeline import TrainingConfig, train_modeldoctor
from ml.reliability.audit_regression import PROBE_ROWS, behaviour_fingerprint, dataset_fingerprint, index_hash
from ml.reliability.evaluation_regression import error_estimation_metrics, model1_metrics
from ml.task import REGRESSION

warnings.filterwarnings("ignore")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_regression_experiments.py"


def hetero_df(n=900, seed=0):
    rng = np.random.RandomState(seed)
    x1, x2 = rng.randn(n), rng.uniform(-3, 3, n)
    region = rng.choice(["north", "south", "online"], n)
    y = 2 * x1 + x2 ** 2 + rng.randn(n) * np.where(region == "online", 3.0, 0.4)
    return pd.DataFrame({"x1": x1, "x2": x2, "region": region, "y": y})


def cfg(**kw):
    return TrainingConfig(task_type=REGRESSION, fast=True, **kw)


@pytest.fixture(scope="module")
def run():
    df = hetero_df()
    art, rep = train_modeldoctor(df, "y", cfg())
    return df, art, rep


# ---------------------------------------------------------------------------
# Model 1 metrics
# ---------------------------------------------------------------------------

def test_model1_metrics_hand_checked():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 3.0, 6.0])
    m = model1_metrics(y, p, y_train_median=2.0)
    assert m["mae"] == pytest.approx(0.5) and m["rmse"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(1 - 4 / 5)
    assert m["median_baseline"]["mae"] == pytest.approx(1.0)  # |2-1|+0+1+2 / 4
    assert m["mae_skill_vs_median"] == pytest.approx(0.5)


def test_zero_error_and_negative_targets():
    y = np.array([-5.0, -1.0, 0.0, 3.0])
    m = model1_metrics(y, y.copy(), y_train_median=-1.0)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["r2"] == 1.0 and m["mae_skill_vs_median"] == 1.0


def test_constant_targets_make_r2_undefined():
    m = model1_metrics(np.full(5, 7.0), np.array([7.0, 7.0, 8.0, 6.0, 7.0]), 7.0)
    assert m["r2"] is None and "constant" in m["r2_note"]
    assert m["mae_skill_vs_median"] is None  # median baseline is already perfect


def test_non_finite_predictions_and_bad_inputs_are_errors():
    with pytest.raises(ValueError, match="non-finite prediction"):
        model1_metrics(np.ones(3), np.array([1.0, np.nan, np.inf]), 1.0)
    with pytest.raises(ValueError, match="different lengths"):
        model1_metrics(np.ones(3), np.ones(4), 1.0)
    with pytest.raises(ValueError, match="targets"):
        model1_metrics(np.array([1.0, np.nan]), np.ones(2), 1.0)


def test_unequal_prediction_scale_is_reported_not_hidden():
    y = np.linspace(1, 10, 20)
    m = model1_metrics(y, y * 1000, float(np.median(y)))
    assert m["mae"] > 1000 and m["r2"] < -1000 and m["mae_skill_vs_median"] < 0


def test_empty_evaluation_set():
    m = model1_metrics(np.array([]), np.array([]), 0.0)
    assert m["n"] == 0 and m["mae"] is None and m["r2"] is None
    e = error_estimation_metrics(np.array([]), np.array([]))
    assert e["n"] == 0 and e["spearman"] is None and e["calibration_bins"] == []


# ---------------------------------------------------------------------------
# Model 2 (error estimation) metrics
# ---------------------------------------------------------------------------

def test_error_estimation_metrics_hand_checked():
    obs = np.arange(1.0, 11.0)
    perfect = error_estimation_metrics(obs, obs.copy(), constant_estimate=5.5, n_bins=5,
                                       ranking_baselines={"reversed": -obs})
    assert perfect["mae"] == 0.0 and perfect["spearman"] == pytest.approx(1.0)
    assert perfect["constant_reference"]["mae"] == pytest.approx(2.5)
    assert perfect["ranking_baselines"]["reversed"] == pytest.approx(-1.0)
    assert [b["mean_observed"] for b in perfect["calibration_bins"]] == [1.5, 3.5, 5.5, 7.5, 9.5]
    shifted = error_estimation_metrics(obs, obs + 2)
    assert shifted["mae"] == pytest.approx(2.0) and shifted["spearman"] == pytest.approx(1.0)


def test_missing_estimates_are_excluded_and_counted():
    obs = np.arange(1.0, 7.0)
    est = np.array([1.0, np.nan, 3.0, np.inf, 5.0, 6.0])
    e = error_estimation_metrics(obs, est)
    assert e["n"] == 4 and e["n_missing_estimates"] == 2 and e["mae"] == 0.0
    allnan = error_estimation_metrics(obs, np.full(6, np.nan))
    assert allnan["n"] == 0 and allnan["mae"] is None


def test_constant_inputs_give_undefined_correlations():
    e = error_estimation_metrics(np.zeros(10), np.linspace(0, 1, 10))  # zero-error predictions
    assert e["spearman"] is None and e["pearson"] is None and e["mae"] == pytest.approx(0.5)
    dust = error_estimation_metrics(np.arange(10.0), np.full(10, 0.1) + np.arange(10) * 1e-18)  # float-constant
    assert dust["pearson"] is None and dust["spearman"] is None
    one = error_estimation_metrics(np.array([1.0]), np.array([2.0]))
    assert one["n"] == 1 and one["spearman"] is None


# ---------------------------------------------------------------------------
# Pipeline report: separation, leakage, reproducibility
# ---------------------------------------------------------------------------

def test_report_separates_the_four_questions(run):
    _, _, rep = run
    json.dumps(rep)
    te = rep["test_evaluation"]
    assert set(te) == {"model1", "model2", "uncertainty"}
    assert te["uncertainty"]["prediction_intervals"] is None  # never claimed
    assert "modeldoctor_bands" in rep["mitigation_evaluation"]
    assert te["model1"]["n"] == rep["mitigation_evaluation"]["n_test"] == te["model2"]["n"]
    assert "roc_auc" not in json.dumps(te) and "accuracy" not in json.dumps(te)


def test_leakage_audit_passes_with_evidence(run):
    df, art, rep = run
    audit = rep["leakage_audit"]
    assert audit["passed"]
    names = {c["check"]: c for c in audit["checks"]}
    n_train = rep["splits"]["train"]["rows"]
    assert names["Model 1 feature preprocessing fitted on train only"]["passed"] is True
    assert f"saw {n_train} rows" in names["Model 1 feature preprocessing fitted on train only"]["evidence"]
    assert names["Model 2 trained on calibration rows only (out-of-fold Model 1 errors)"]["passed"]
    assert names["Test partition used once, for final evaluation"]["passed"]


def test_leakage_audit_detects_a_model_fitted_on_too_many_rows(run):
    from ml.reliability.audit_regression import leakage_audit
    from ml.data.splitter import split_regression_data

    df, art, rep = run
    splits = split_regression_data(df, "y", random_state=42)
    leaky = dict(art)
    m1 = leaky["model1"]
    import copy
    m1 = copy.deepcopy(m1)
    m1.fit(pd.concat([splits.X_train_raw, splits.X_test_raw]), np.r_[splits.y_train, splits.y_test])
    leaky["model1"] = m1
    audit = leakage_audit(splits, leaky, model1_cv_rows=len(splits.X_train_raw), model2_rows=len(splits.X_calib_raw),
                          threshold_rows=10, test_rows_evaluated=len(splits.X_test_raw))
    assert not audit["passed"]


@pytest.mark.parametrize("strategy", ["group_column", "time_column"])
def test_leakage_audit_covers_group_and_time_splits(strategy):
    df = hetero_df(n=900).assign(g=np.arange(900) % 90, t=np.arange(900) // 3)
    _, rep = train_modeldoctor(df, "y", cfg(**{strategy: "g" if strategy == "group_column" else "t"}))
    checks = {c["check"] for c in rep["leakage_audit"]["checks"]}
    assert rep["leakage_audit"]["passed"]
    assert ("No group spans two partitions" in checks) or ("Partitions are strictly ordered in time" in checks)


def test_metrics_are_tied_to_the_saved_artifacts(run, tmp_path):
    df, art, rep = run
    from ml.data.splitter import split_regression_data

    save_artifacts(tmp_path, art)
    loaded = load_artifacts(tmp_path)
    r = rep["reproducibility"]
    assert loaded["run_id"] == r["run_id"]
    probe = split_regression_data(df, "y", random_state=42).X_calib_raw.head(PROBE_ROWS)
    assert behaviour_fingerprint(loaded, probe) == r["behaviour_fingerprint"]["sha256"]
    other, _ = train_modeldoctor(df, "y", cfg(force_model="Ridge Regression"))
    assert behaviour_fingerprint(other, probe) != r["behaviour_fingerprint"]["sha256"]


def test_reproducible_with_fixed_seed_and_sensitive_to_seed():
    df = hetero_df(n=700, seed=3)
    _, a = train_modeldoctor(df, "y", cfg(random_state=5))
    _, b = train_modeldoctor(df, "y", cfg(random_state=5))
    _, c = train_modeldoctor(df, "y", cfg(random_state=6))
    ra, rb, rc = a["reproducibility"], b["reproducibility"], c["reproducibility"]
    assert ra["dataset"] == rb["dataset"] and ra["split"] == rb["split"]
    assert ra["split"]["index_sha256"] != rc["split"]["index_sha256"]
    for key in ("mae", "rmse", "r2"):
        assert a["test_evaluation"]["model1"][key] == pytest.approx(b["test_evaluation"]["model1"][key], rel=1e-9)
    assert a["thresholds"]["trust_threshold"] == b["thresholds"]["trust_threshold"]
    assert [x["count"] for x in a["mitigation_evaluation"]["modeldoctor_bands"]] == \
        [x["count"] for x in b["mitigation_evaluation"]["modeldoctor_bands"]]


def test_fingerprints_and_index_hash():
    df = hetero_df(n=50)
    assert dataset_fingerprint(df) == dataset_fingerprint(df.copy())
    assert dataset_fingerprint(df) != dataset_fingerprint(df.assign(y=df.y + 1e-9))
    assert index_hash(pd.Index([3, 1, 2])) == index_hash(pd.Index([1, 2, 3]))


# ---------------------------------------------------------------------------
# Experiment workflow
# ---------------------------------------------------------------------------

def _script():
    spec = importlib.util.spec_from_file_location("run_regression_experiments", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_experiment_script_writes_a_new_run_and_never_overwrites(tmp_path, monkeypatch):
    mod = _script()
    small = [dict(d, n=600) for d in mod.DATASETS[1:3]]
    run_dir = mod.run(tmp_path, fast=True, datasets=small, on_progress=lambda m: None)
    files = sorted(p.name for p in run_dir.iterdir())
    assert files == ["heteroscedastic_continuous.json", "heteroscedastic_region.json", "summary.md"]
    text = (run_dir / "summary.md").read_text(encoding="utf-8")
    for heading in ("1. Model 1", "2. Model 2", "3. Uncertainty", "4. Mitigation", "5. Leakage", "6. Reproducibility"):
        assert heading in text
    manifest = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest) == 1 and json.loads(manifest[0])["path"] == str(run_dir)

    # Same timestamp + same data -> same run id: must refuse instead of overwriting.
    class Frozen:
        @staticmethod
        def now(tz=None):
            from datetime import datetime as real
            return real.strptime(run_dir.name.split("_")[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=tz)

    monkeypatch.setattr(mod, "datetime", Frozen)
    with pytest.raises(FileExistsError):
        mod.run(tmp_path, fast=True, datasets=small, on_progress=lambda m: None)
