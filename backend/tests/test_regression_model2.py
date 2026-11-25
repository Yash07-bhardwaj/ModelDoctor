"""Phase 3 tests: regression Model 2 (expected absolute error estimator).
Synthetic data with controlled, input-dependent noise."""

import functools
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

import ml.pipeline as pl
import ml.reliability.model2_regression as m2r
from ml.data.splitter import split_regression_data
from ml.inference import ModelLoadError, load_artifacts, save_artifacts, validate_artifacts
from ml.models.regression_selector import make_regression_cv_splits
from ml.pipeline import TrainingConfig, predict_regression_reliability, train_regression_reliability
from ml.reliability.model2_regression import (
    CONSTANT_CANDIDATE,
    TargetScaleError,
    robust_target_scale,
    scaled_abs_error,
    train_regression_model2,
)
from ml.reliability.regression_features import REGRESSION_FEATURES, RegressionReliabilityFeatureExtractor
from ml.task import REGRESSION

warnings.filterwarnings("ignore")


def hetero_df(n=1500, seed=0, scale=1.0, shift=0.0):
    """Noise is 8x larger for region == 'online': a learnable error pattern."""
    rng = np.random.RandomState(seed)
    x1, x2 = rng.randn(n), rng.uniform(-3, 3, n)
    region = rng.choice(["north", "south", "online"], n)
    noise = rng.randn(n) * np.where(region == "online", 3.0, 0.4)
    y = (2 * x1 + x2 ** 2 + noise) * scale + shift
    return pd.DataFrame({"x1": x1, "x2": x2, "region": region, "y": y})


def cfg(**kw):
    return TrainingConfig(task_type=REGRESSION, fast=True, **kw)


@pytest.fixture(scope="module")
def run():
    df = hetero_df()
    art, rep, dev = train_regression_reliability(df, "y", cfg(force_model="Random Forest"))
    return df, art, rep, dev


# ---------------------------------------------------------------------------
# Error target
# ---------------------------------------------------------------------------

def test_scaled_abs_error_handles_zero_and_negative_targets():
    y = np.array([0.0, -5.0, 2.0, -1e-9])
    pred = np.array([1.0, -3.0, 2.0, 1e-9])
    np.testing.assert_allclose(scaled_abs_error(y, pred, 2.0), [0.5, 1.0, 0.0, 1e-9])


def test_robust_target_scale():
    y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # MAD = 1 -> 1.4826, ignores the outlier
    assert robust_target_scale(y) == (pytest.approx(1.4826), "mad")
    zero_inflated = np.r_[np.zeros(90), np.arange(1.0, 11.0)]  # MAD = 0 and IQR = 0
    s, method = robust_target_scale(zero_inflated)
    assert s > 0 and method == "std"
    assert robust_target_scale(-1e6 * y)[0] == pytest.approx(1e6 * 1.4826)
    with pytest.raises(TargetScaleError):
        robust_target_scale(np.full(10, 3.0))


def test_calibration_error_targets_come_from_unseen_rows(run, monkeypatch):
    df, art, rep, dev = run
    s = art["target_scale"]
    splits = split_regression_data(df, "y", random_state=42)
    pred = art["model1"].predict(splits.X_calib_raw)
    np.testing.assert_allclose(dev["scaled_error"], np.abs(splits.y_calib - pred) / s)
    # Model 1 is fit (CV + refit) on TRAIN rows only.
    seen = {}
    real = pl.select_regression_model

    def spy(X_train_raw, *a, **k):
        seen["rows"] = set(X_train_raw.index)
        return real(X_train_raw, *a, **k)

    monkeypatch.setattr(pl, "select_regression_model", spy)
    train_regression_reliability(df, "y", cfg(force_model="Ridge Regression"))
    assert not seen["rows"] & set(splits.X_calib_raw.index)
    assert not seen["rows"] & set(splits.X_test_raw.index)


# ---------------------------------------------------------------------------
# Out-of-fold generation and leakage
# ---------------------------------------------------------------------------

class Recorder(BaseEstimator, RegressorMixin):
    fitted = []

    def fit(self, X, y):
        Recorder.fitted.append(set(X["row_id"]))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_model2_oof_estimates_never_come_from_a_model_fit_on_that_row(monkeypatch):
    rng = np.random.RandomState(0)
    n = 200
    feats = pd.DataFrame({"row_id": np.arange(n, dtype=float), "bootstrap_std": rng.rand(n)})
    err = feats["bootstrap_std"].to_numpy() + 0.1 * rng.rand(n)
    folds = make_regression_cv_splits(n, "random", random_state=0)
    Recorder.fitted = []
    monkeypatch.setattr(m2r, "_candidates", lambda cols, rs: [("Recorder", Recorder(), ["row_id"])])
    train_regression_model2(feats, err, folds, target_scale=1.0)
    for (tr, va), fitted in zip(folds, Recorder.fitted):
        assert not fitted & set(va)  # validation rows were never in that fold's fit
        assert fitted == set(tr)


def test_train_residual_feature_uses_out_of_fold_not_in_sample_residuals(run):
    df, art, _, _ = run
    ex = art["extractor"]
    splits = split_regression_data(df, "y", random_state=42)
    in_sample = np.abs(splits.y_train - art["model1"].predict(splits.X_train_raw))
    # A Random Forest fits its own rows closely: in-sample residuals understate the error.
    assert np.nanmean(ex.train_abs_residual_) > 1.5 * in_sample.mean()

    fresh = RegressionReliabilityFeatureExtractor(n_bootstraps=0)
    with pytest.raises(ValueError, match="out-of-fold"):
        fresh.fit(splits.X_train_raw, None, splits.y_train, np.full(len(splits.y_train), np.nan),
                  1.0, ["x1", "x2"], ["region"])


def test_reliability_training_never_reads_the_test_partition(monkeypatch):
    real = pl.split_regression_data

    class Poison:
        def __getattr__(self, name):
            raise AssertionError(f"test partition accessed ({name})")

        def __len__(self):
            raise AssertionError("test partition accessed (len)")

    def poisoned(*a, **k):
        s = real(*a, **k)
        s.X_test_raw, s.y_test = Poison(), Poison()
        s.get_split_summary = lambda: {"poisoned": True}
        return s

    monkeypatch.setattr(pl, "split_regression_data", poisoned)
    _, rep, _ = train_regression_reliability(hetero_df(n=800), "y", cfg())
    assert rep["test_partition"] == "untouched"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_feature_extraction_is_consistent_between_training_and_inference(run, tmp_path):
    df, art, _, dev = run
    splits = split_regression_data(df, "y", random_state=42)
    save_artifacts(tmp_path, art)
    loaded = load_artifacts(tmp_path)
    again, _ = loaded["extractor"].transform_with_flags(splits.X_calib_raw, loaded["model1"])
    assert list(again.columns) == REGRESSION_FEATURES
    pd.testing.assert_frame_equal(again, dev["features"])
    assert np.isfinite(again.to_numpy()).all()
    assert "y" not in again.columns


def test_features_never_depend_on_the_records_target(run):
    df, art, _, _ = run
    X = df.drop(columns="y").head(50)
    a, _ = art["extractor"].transform_with_flags(X, art["model1"])
    b, _ = art["extractor"].transform_with_flags(X.assign(), art["model1"])
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Model 2 behaviour
# ---------------------------------------------------------------------------

def test_model2_learns_the_controlled_error_pattern(run):
    df, art, rep, dev = run
    m = rep["model2"]
    assert m["status"] == "success" and art["reliability_mode"] == "model2"
    assert m["winner"] != CONSTANT_CANDIDATE
    cands = m["candidates"]
    assert cands[m["winner"]]["mse"] < cands[CONSTANT_CANDIDATE]["mse"]
    assert m["oof_winner_metrics_scaled"]["spearman"] > 0.3
    X = hetero_df(seed=9).drop(columns="y").head(600)
    res = predict_regression_reliability(art, X)
    est = np.array([r["estimated_abs_error"] for r in res])
    online = X["region"].to_numpy() == "online"
    assert est[online].mean() > 2.0 * est[~online].mean()
    assert np.all(est >= 0) and np.all(np.isfinite(est))
    assert all(r["error_estimate_source"] == "model2" for r in res)


def test_save_load_gives_identical_estimates(run, tmp_path):
    _, art, _, _ = run
    save_artifacts(tmp_path, art)
    assert (tmp_path / "model2.joblib").exists() and (tmp_path / "extractor.joblib").exists()
    loaded = load_artifacts(tmp_path)
    X = hetero_df(seed=3).drop(columns="y").head(40)
    a = predict_regression_reliability(art, X)
    b = predict_regression_reliability(loaded, X)
    assert [r["estimated_abs_error"] for r in a] == [r["estimated_abs_error"] for r in b]
    # Multi-threaded forests can differ in the last bit after reloading (summation order).
    np.testing.assert_allclose([r["prediction"] for r in a], [r["prediction"] for r in b], rtol=1e-12)


@pytest.mark.parametrize("scale, shift", [(1.0, -50.0), (1e6, 0.0)])
def test_estimates_are_scale_and_sign_invariant(scale, shift):
    base = hetero_df(n=900)
    a, _, _ = train_regression_reliability(base, "y", cfg(force_model="Ridge Regression"))
    moved = base.assign(y=base["y"] * scale + shift)
    b, _, _ = train_regression_reliability(moved, "y", cfg(force_model="Ridge Regression"))
    X = base.drop(columns="y").head(60)
    ra, rb = predict_regression_reliability(a, X), predict_regression_reliability(b, X)
    ea = np.array([r["estimated_error_scaled"] for r in ra])
    eb = np.array([r["estimated_error_scaled"] for r in rb])
    np.testing.assert_allclose(eb, ea, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose([r["estimated_abs_error"] for r in rb],
                               np.array([r["estimated_abs_error"] for r in ra]) * scale, rtol=1e-5)


def test_zero_inflated_negative_targets_give_finite_non_negative_estimates():
    df = hetero_df(n=900)
    df["y"] = np.where(df["x1"] < 0, 0.0, -np.abs(df["y"]))
    art, rep, _ = train_regression_reliability(df, "y", cfg())
    res = predict_regression_reliability(art, df.drop(columns="y").head(100))
    est = np.array([r["estimated_abs_error"] for r in res])
    assert np.all(np.isfinite(est)) and np.all(est >= 0)
    assert rep["target_scale"]["value"] > 0


# ---------------------------------------------------------------------------
# Fallbacks and robustness
# ---------------------------------------------------------------------------

def _folds(n):
    return make_regression_cv_splits(n, "random", random_state=0)


def test_constant_errors_insufficient_rows_and_no_signal():
    rng = np.random.RandomState(0)
    feats = pd.DataFrame({c: rng.rand(300) for c in REGRESSION_FEATURES})
    res = train_regression_model2(feats, np.full(300, 0.25), _folds(300), target_scale=2.0)
    assert res["status"] == "constant_error" and res["model2"] is None
    assert res["fallback_scaled_error"] == pytest.approx(0.25)

    small = train_regression_model2(feats.head(20), rng.rand(20), _folds(20), target_scale=1.0)
    assert small["status"] == "insufficient_data"

    noise = train_regression_model2(feats, rng.exponential(size=300), _folds(300), target_scale=1.0)
    assert noise["status"] == "no_signal" and noise["model2"] is None
    assert noise["candidates_scores"][CONSTANT_CANDIDATE]["selected"]


def test_invalid_error_targets_are_rejected():
    feats = pd.DataFrame({"bootstrap_std": np.arange(100.0)})
    with pytest.raises(ValueError):
        train_regression_model2(feats, np.r_[np.ones(99), np.nan], _folds(100), 1.0)
    with pytest.raises(ValueError):
        train_regression_model2(feats, -np.ones(100), _folds(100), 1.0)


def test_failed_candidates_do_not_stop_training(monkeypatch):
    class Broken(BaseEstimator, RegressorMixin):
        def fit(self, X, y):
            raise RuntimeError("boom")

    real = m2r._candidates

    def with_broken(cols, rs):
        c = real(cols, rs)
        return c[:2] + [("Broken", Broken(), cols)] + c[2:]

    monkeypatch.setattr(m2r, "_candidates", with_broken)
    rng = np.random.RandomState(1)
    feats = pd.DataFrame({c: rng.rand(300) for c in REGRESSION_FEATURES})
    err = 2 * feats["knn_target_std"].to_numpy() + 0.05 * rng.rand(300)
    res = train_regression_model2(feats, err, _folds(300), target_scale=1.0)
    assert res["candidates_scores"]["Broken"]["status"] == "failed"
    assert res["status"] == "success" and res["winner_name"] != "Broken"


def test_end_to_end_fallback_without_model2_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(pl, "train_regression_model2",
                        functools.partial(train_regression_model2, min_rows=10 ** 6))
    df = hetero_df(n=800)
    art, rep, _ = train_regression_reliability(df, "y", cfg())
    assert art["reliability_mode"] == "constant_fallback" and art["model2"] is None
    assert rep["model2"]["status"] == "insufficient_data"
    save_artifacts(tmp_path, art)
    assert not (tmp_path / "model2.joblib").exists()
    loaded = load_artifacts(tmp_path)
    X = df.drop(columns="y").head(20)
    res = predict_regression_reliability(loaded, X)
    expected = art["fallback_scaled_error"] * art["target_scale"]
    assert all(r["estimated_abs_error"] == pytest.approx(expected) for r in res)
    assert all(r["error_estimate_source"] == "calibration_mean_error" for r in res)
    far = X.head(1).assign(x1=1e6)
    flags = predict_regression_reliability(loaded, far)[0]["flags"]
    assert any(f["type"] == "outside_reliability_support" for f in flags)


def test_non_finite_features_and_model2_failure_are_safe(run, monkeypatch):
    _, art, _, _ = run
    m2 = art["model2"]
    X = hetero_df(seed=4).drop(columns="y").head(3)
    feats, _ = art["extractor"].transform_with_flags(X, art["model1"])
    feats.iloc[0, 0] = np.nan
    feats.iloc[1, 0] = np.inf
    est = m2.predict_scaled_error(feats)
    assert np.all(np.isfinite(est)) and np.all(est >= 0)
    viol = m2.support_violations(feats)
    assert feats.columns[0] in viol[0] and feats.columns[0] in viol[1]

    clean = predict_regression_reliability(art, X)
    monkeypatch.setattr(type(m2), "predict_scaled_error", lambda self, f: (_ for _ in ()).throw(RuntimeError("x")))
    broken = predict_regression_reliability(art, X)
    # Model 1 untouched (to the last bit, modulo multi-threaded forest summation order)
    np.testing.assert_allclose([r["prediction"] for r in broken], [r["prediction"] for r in clean], rtol=1e-12)
    assert all(r["error_estimate_source"] == "calibration_mean_error" for r in broken)
    assert all(any(f["type"] == "model2_unavailable" for f in r["flags"]) for r in broken)


def test_artifact_validation_for_reliability_bundles(run):
    _, art, _, _ = run
    validate_artifacts(art)
    with pytest.raises(ModelLoadError, match="Model 2"):
        validate_artifacts(dict(art, model2=None))
    with pytest.raises(ModelLoadError, match="target scale"):
        validate_artifacts(dict(art, target_scale=0.0))
    with pytest.raises(ModelLoadError, match="reliability mode"):
        validate_artifacts(dict(art, reliability_mode="magic"))
