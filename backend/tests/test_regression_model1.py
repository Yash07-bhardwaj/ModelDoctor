"""Phase 2 tests: regression Model 1 candidates, CV selection, predictions across target
scales, persistence, leakage boundaries and reproducibility. Synthetic data only."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_regressor

from ml.inference import InferenceEngine, ModelLoadError, load_artifacts, save_artifacts, validate_artifacts
from ml.models.regression_candidates import REGRESSION_CANDIDATES, get_regression_candidates
from ml.models.regression_selector import (
    make_regression_cv_splits,
    regression_scores,
    select_regression_model,
)
from ml.pipeline import (
    TrainingConfig,
    coerce_records,
    predict_records,
    predict_regression,
    train_regression_model1,
)
from ml.task import REGRESSION

warnings.filterwarnings("ignore")
CFG = dict(task_type=REGRESSION, fast=True)


def linear_df(n=900, seed=0, scale=1.0, shift=0.0):
    rng = np.random.RandomState(seed)
    x1, x2 = rng.randn(n), rng.randn(n)
    cat = rng.choice(["a", "b", "c"], n)
    y = (3.0 * x1 - 2.0 * x2 + 1.5 * (cat == "b") + 0.2 * rng.randn(n)) * scale + shift
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})


def nonlinear_df(n=900, seed=1):
    rng = np.random.RandomState(seed)
    x1, x2 = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    y = 4.0 * (x1 > 0) * (x2 > 0) + np.sin(3 * x1) + 0.1 * rng.randn(n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture(scope="module")
def linear_run():
    return train_regression_model1(linear_df(), "y", TrainingConfig(**CFG))


# ---------------------------------------------------------------------------
# Candidates and metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", REGRESSION_CANDIDATES)
def test_every_candidate_fits_and_predicts_finite(name):
    df = linear_df(n=400)
    pipe = get_regression_candidates(["x1", "x2"], ["cat"], fast=True)[name]
    assert is_regressor(pipe)
    pipe.fit(df.drop(columns="y"), df["y"])
    pred = pipe.predict(df.drop(columns="y"))
    assert pred.shape == (400,) and np.all(np.isfinite(pred))
    assert regression_scores(df["y"], pred)["r2"] > 0.5


def test_knn_skipped_for_very_large_train_sets():
    assert "KNN" not in get_regression_candidates(["x"], [], n_train_rows=10**6)


def test_regression_scores_hand_checked_and_undefined_r2():
    s = regression_scores(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert s["mae"] == pytest.approx(2 / 3) and s["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert s["r2"] == pytest.approx(1 - 4 / 2)
    assert regression_scores(np.array([2.0, 2.0]), np.array([1.0, 3.0]))["r2"] is None


# ---------------------------------------------------------------------------
# Cross-validation and selection
# ---------------------------------------------------------------------------

def test_cv_folds_partition_the_train_rows():
    folds = make_regression_cv_splits(100, "random", random_state=0)
    val = np.concatenate([v for _, v in folds])
    assert len(folds) == 5 and sorted(val) == list(range(100))
    assert all(not set(t) & set(v) for t, v in folds)


def test_group_cv_never_splits_a_group():
    groups = np.repeat(np.arange(20), 5)
    for tr, va in make_regression_cv_splits(100, "group", groups):
        assert not set(groups[tr]) & set(groups[va])


def test_time_cv_validates_only_on_later_times():
    rng = np.random.RandomState(0)
    times = rng.permutation(np.repeat(np.arange(30), 4))  # unordered input, tied timestamps
    folds = make_regression_cv_splits(120, "time", times)
    assert len(folds) >= 2
    for tr, va in folds:
        assert times[tr].max() < times[va].min()


def test_linear_relationship_selects_linear_model(linear_run):
    _, rep = linear_run
    assert rep["selected_model"] == "Ridge Regression"
    assert rep["leaderboard"][0]["model"] == "Ridge Regression"
    assert rep["calibration_holdout"]["model1"]["r2"] > 0.95
    best = rep["leaderboard"][0]["mae_mean"]
    assert best < rep["median_baseline_cv"]["mae_mean"]


def test_nonlinear_relationship_selects_nonlinear_model():
    _, rep = train_regression_model1(nonlinear_df(), "y", TrainingConfig(**CFG))
    assert rep["selected_model"] != "Ridge Regression"
    lb = {r["model"]: r for r in rep["leaderboard"]}
    assert lb[rep["selected_model"]]["mae_mean"] < lb["Ridge Regression"]["mae_mean"]


@pytest.mark.parametrize("metric, better", [("mae", min), ("rmse", min), ("r2", max)])
def test_selection_metric_is_configurable(metric, better):
    df = nonlinear_df(n=500)
    res = select_regression_model(df[["x1", "x2"]], df["y"].to_numpy(), ["x1", "x2"], [],
                                  selection_metric=metric, fast=True)
    vals = {r["model"]: r[f"{metric}_mean"] for r in res["leaderboard"]}
    assert vals[res["winner_name"]] == better(vals.values())


def test_selection_input_validation():
    X, y = pd.DataFrame({"x": np.arange(100.0)}), np.arange(100.0)
    with pytest.raises(ValueError, match="metric"):
        select_regression_model(X, y, ["x"], [], selection_metric="accuracy")
    with pytest.raises(ValueError, match="Unknown regression candidate"):
        select_regression_model(X, y, ["x"], [], force_model="Logistic Regression")
    with pytest.raises(ValueError, match="constant"):
        select_regression_model(X, np.ones(100), ["x"], [])
    with pytest.raises(ValueError, match="non-finite"):
        select_regression_model(X, np.r_[y[:-1], np.inf], ["x"], [])


def test_force_model_and_failed_candidates(monkeypatch):
    df = linear_df(n=300)
    X, y = df.drop(columns="y"), df["y"].to_numpy()
    res = select_regression_model(X, y, ["x1", "x2"], ["cat"], force_model="KNN", fast=True)
    assert res["winner_name"] == "KNN" and len(res["leaderboard"]) == 1

    import ml.models.regression_selector as sel

    real = sel.get_regression_candidates

    class Broken:
        def fit(self, *a, **k):
            raise RuntimeError("boom")

    def with_broken(*a, **k):
        c = real(*a, **k)
        c["Ridge Regression"].steps[-1] = ("regressor", Broken())
        return c

    monkeypatch.setattr(sel, "get_regression_candidates", with_broken)
    res = select_regression_model(X, y, ["x1", "x2"], ["cat"], fast=True)
    failed = [r for r in res["leaderboard"] if r["status"] == "failed"]
    assert [r["model"] for r in failed] == ["Ridge Regression"] and res["winner_name"] != "Ridge Regression"

    monkeypatch.setattr(sel, "get_regression_candidates",
                        lambda *a, **k: {"Only": clone(with_broken(*a, **k)["Ridge Regression"])})
    with pytest.raises(RuntimeError, match="All regression candidates failed"):
        select_regression_model(X, y, ["x1", "x2"], ["cat"], fast=True)


# ---------------------------------------------------------------------------
# Predictions across target scales / signs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale, shift", [(1.0, -500.0), (1e-9, 0.0), (1e9, 0.0), (1.0, 5e12)])
def test_predictions_follow_target_scale_and_sign(scale, shift):
    df = linear_df(scale=scale, shift=shift)
    art, rep = train_regression_model1(df, "y", TrainingConfig(**CFG, force_model="XGBoost"))
    pred = predict_regression(art, df.drop(columns="y").head(200))
    y = df["y"].head(200).to_numpy()
    assert np.all(np.isfinite(pred))
    # Error relative to the target's spread is the same whatever the scale/offset.
    assert np.mean(np.abs(pred - y)) / np.std(y) < 0.2
    assert rep["calibration_holdout"]["model1"]["r2"] > 0.9


def test_zero_heavy_target_predicts_near_zero():
    df = linear_df()
    df.loc[df["x1"] < 0, "y"] = 0.0
    art, _ = train_regression_model1(df, "y", TrainingConfig(**CFG, force_model="Random Forest"))
    zeros = df[df["x1"] < -1].drop(columns="y")
    assert np.abs(predict_regression(art, zeros)).mean() < 0.5


def test_ridge_is_scale_equivariant():
    base = linear_df()
    a, _ = train_regression_model1(base, "y", TrainingConfig(**CFG, force_model="Ridge Regression"))
    b, _ = train_regression_model1(base.assign(y=base["y"] * 1e9), "y",
                                   TrainingConfig(**CFG, force_model="Ridge Regression"))
    X = base.drop(columns="y").head(50)
    np.testing.assert_allclose(predict_regression(b, X), predict_regression(a, X) * 1e9, rtol=1e-6)


# ---------------------------------------------------------------------------
# Persistence, feature order, task separation
# ---------------------------------------------------------------------------

def test_save_load_predict_without_retraining(linear_run, tmp_path):
    art, _ = linear_run
    save_artifacts(tmp_path, art)
    assert not (tmp_path / "model2.joblib").exists() and not (tmp_path / "engine.joblib").exists()
    loaded = load_artifacts(tmp_path)
    assert loaded["task_type"] == REGRESSION and loaded["artifact_version"] == 1
    assert [f["name"] for f in loaded["feature_schema"]] == ["x1", "x2", "cat"]
    X = linear_df(seed=5).drop(columns="y").head(40)
    np.testing.assert_array_equal(predict_regression(loaded, X), predict_regression(art, X))


def test_feature_order_and_schema_coercion(linear_run):
    art, _ = linear_run
    X = linear_df(seed=6).drop(columns="y").head(30)
    shuffled = X[["cat", "x2", "x1"]]
    coerced = coerce_records(shuffled, art["feature_schema"])
    np.testing.assert_array_equal(predict_regression(art, coerced), predict_regression(art, X))


def test_regression_and_classification_bundles_are_not_interchangeable(linear_run):
    art, _ = linear_run
    with pytest.raises(ModelLoadError, match="classification models"):
        InferenceEngine(art)
    with pytest.raises(ValueError, match="classification bundles only"):
        predict_records(art, linear_df().drop(columns="y").head(3))
    bad = dict(art, model1=get_regression_candidates(["x"], [])["Ridge Regression"].steps[-1][1])
    validate_artifacts(bad)  # a regressor is accepted
    from sklearn.linear_model import LogisticRegression
    with pytest.raises(ModelLoadError, match="not a regressor"):
        validate_artifacts(dict(art, model1=LogisticRegression()))
    with pytest.raises(ModelLoadError, match="incomplete"):
        validate_artifacts({k: v for k, v in art.items() if k != "artifact_version"})


# ---------------------------------------------------------------------------
# Leakage boundaries and reproducibility
# ---------------------------------------------------------------------------

def test_model1_training_never_reads_the_test_partition(monkeypatch):
    import ml.pipeline as pl

    real = pl.split_regression_data

    class Poison:
        def __getattr__(self, name):
            raise AssertionError(f"test partition accessed ({name})")

        def __len__(self):
            raise AssertionError("test partition accessed (len)")

    def poisoned(*a, **k):
        s = real(*a, **k)
        s.X_test_raw = Poison()
        s.y_test = Poison()
        s.get_split_summary = lambda: {"poisoned": True}
        return s

    monkeypatch.setattr(pl, "split_regression_data", poisoned)
    _, rep = pl.train_regression_model1(linear_df(n=600), "y", TrainingConfig(**CFG))
    assert rep["test_partition"] == "untouched"


def test_selector_only_sees_train_rows(monkeypatch):
    import ml.pipeline as pl

    seen = {}
    real = pl.select_regression_model

    def spy(X_train_raw, y_train, *a, **k):
        seen["index"] = set(X_train_raw.index)
        return real(X_train_raw, y_train, *a, **k)

    monkeypatch.setattr(pl, "select_regression_model", spy)
    real_split = pl.split_regression_data
    holder = {}

    def keep(*a, **k):
        holder["s"] = real_split(*a, **k)
        return holder["s"]

    monkeypatch.setattr(pl, "split_regression_data", keep)
    pl.train_regression_model1(linear_df(n=600), "y", TrainingConfig(**CFG))
    s = holder["s"]
    assert seen["index"] == set(s.X_train_raw.index)
    assert not seen["index"] & (set(s.X_calib_raw.index) | set(s.X_test_raw.index))


def test_reproducible_with_fixed_seed():
    df = nonlinear_df(n=600)
    a, ra = train_regression_model1(df, "y", TrainingConfig(**CFG, random_state=7))
    b, rb = train_regression_model1(df, "y", TrainingConfig(**CFG, random_state=7))
    assert [r["model"] for r in ra["leaderboard"]] == [r["model"] for r in rb["leaderboard"]]
    # Scores are unrounded (rounding broke selection for tiny targets), so compare to float noise:
    # multi-threaded forests can differ in the last bit between runs.
    for x, y in zip(ra["leaderboard"], rb["leaderboard"]):
        for k in ("mae_mean", "rmse_mean", "r2_mean"):
            assert x[k] == pytest.approx(y[k], rel=1e-12)
    X = df.drop(columns="y").head(50)
    # Multi-threaded tree ensembles (n_jobs=-1) may sum in a different order: identical to
    # floating-point rounding, not bit-for-bit.
    np.testing.assert_allclose(predict_regression(a, X), predict_regression(b, X), rtol=1e-12, atol=0)


def test_group_strategy_flows_into_group_cv():
    df = linear_df(n=900)
    df["store"] = np.random.RandomState(2).randint(0, 60, len(df))
    _, rep = train_regression_model1(df, "y", TrainingConfig(**CFG, group_column="store"))
    assert rep["cv_strategy"] == "group" and "store" not in rep["features"]["numeric"]
