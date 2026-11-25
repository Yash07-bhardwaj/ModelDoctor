"""Phase 1 tests: regression task type, target validation, preprocessing boundaries and
leakage-safe train / calibration / test splitting. All data is generated."""

import numpy as np
import pandas as pd
import pytest

from ml.data.loader import (
    DatasetValidationError,
    InsufficientRowsError,
    InvalidTargetError,
    MissingTargetError,
    UnsupportedTaskError,
    load_dataset,
)
from ml.data.preprocessing import build_preprocessor
from ml.data.splitter import (
    SplitValidationError,
    split_data,
    split_regression_data,
    validate_split_fractions,
    validate_splits,
)
from ml.pipeline import TrainingConfig, prepare_regression_data, train_modeldoctor
from ml.task import CLASSIFICATION, REGRESSION, InvalidTaskTypeError, resolve_task_type


def make_df(n=600, seed=0, scale=1.0, shift=0.0, integer=False):
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    cat = rng.choice(["a", "b", "c"], n)
    y = (2.0 * x1 + (cat == "b") + 0.3 * rng.randn(n)) * scale + shift
    if integer:
        y = np.round(y).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": rng.uniform(0, 10, n), "cat": cat, "y": y})
    df.loc[rng.rand(n) < 0.05, "x2"] = np.nan
    return df


def load_reg(df, target="y"):
    return load_dataset(df.to_csv(index=False), target, task_type=REGRESSION)


# ---------------------------------------------------------------------------
# Task type
# ---------------------------------------------------------------------------

def test_task_type_resolution():
    assert resolve_task_type(None) == CLASSIFICATION
    assert resolve_task_type(" Regression ") == REGRESSION
    with pytest.raises(InvalidTaskTypeError):
        resolve_task_type("forecasting")
    assert TrainingConfig().task_type == CLASSIFICATION


def test_classification_default_still_rejects_continuous_targets_with_hint():
    df = make_df()
    with pytest.raises(UnsupportedTaskError) as exc:
        load_dataset(df.to_csv(index=False), "y")
    msg = str(exc.value)
    assert "Only classification targets are supported in this version." in msg
    assert "task_type='regression'" in msg


def test_regression_training_is_not_silently_run_as_classification():
    art, rep = train_modeldoctor(make_df(), "y", TrainingConfig(task_type=REGRESSION, fast=True))
    assert rep["task"] == REGRESSION and art["task_type"] == REGRESSION
    assert "classes" not in art and "classes" not in rep


# ---------------------------------------------------------------------------
# Regression target validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {},                               # continuous
    {"integer": True},                # integer-valued
    {"shift": -1000.0},               # all negative
    {"scale": 1e-9},                  # very small
    {"scale": 1e12, "shift": 5e12},   # very large
])
def test_valid_regression_targets_load_unchanged(kwargs):
    df = make_df(**kwargs)
    out = load_reg(df)
    np.testing.assert_array_equal(out["y"].to_numpy(), df["y"].to_numpy())


def test_zero_heavy_target_is_valid():
    df = make_df()
    df.loc[df.index[:300], "y"] = 0.0
    assert (load_reg(df)["y"] == 0).sum() == 300


@pytest.mark.parametrize("mutate, error, fragment", [
    (lambda d: d.assign(y=d["y"].mask(d.index < 3)), InvalidTargetError, "missing values"),
    (lambda d: d.assign(y=d["y"].where(d.index != 5, np.inf)), InvalidTargetError, "infinite"),
    (lambda d: d.assign(y=7.5), InvalidTargetError, "constant"),
    (lambda d: d.assign(y=np.where(d.index % 2, "high", "low")), InvalidTargetError, "must be numeric"),
    (lambda d: d.assign(y=d.index % 2 == 0), InvalidTargetError, "must be numeric"),
    (lambda d: d.assign(leak=d["y"]), DatasetValidationError, "identical to the target"),
    (lambda d: d[["y"]], DatasetValidationError, "no feature columns"),
])
def test_invalid_regression_targets_are_rejected(mutate, error, fragment):
    with pytest.raises(error) as exc:
        load_reg(mutate(make_df()))
    assert fragment in str(exc.value)


def test_insufficient_rows_and_missing_target():
    with pytest.raises(InsufficientRowsError):
        load_reg(make_df(n=200))
    with pytest.raises(MissingTargetError):
        load_reg(make_df(), target="nope")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def test_random_split_is_disjoint_proportional_and_reproducible():
    df = load_reg(make_df(n=1000))
    a = split_regression_data(df, "y", random_state=3)
    b = split_regression_data(df, "y", random_state=3)
    c = split_regression_data(df, "y", random_state=4)
    idx = [a.X_train_raw.index, a.X_calib_raw.index, a.X_test_raw.index]
    assert [len(i) for i in idx] == [600, 200, 200]
    assert set().union(*map(set, idx)) == set(df.index)
    assert not set(idx[0]) & set(idx[1]) and not set(idx[0]) & set(idx[2]) and not set(idx[1]) & set(idx[2])
    assert a.X_train_raw.index.equals(b.X_train_raw.index)
    assert not a.X_train_raw.index.equals(c.X_train_raw.index)
    assert "y" not in a.feature_names and a.y_train.dtype == float
    np.testing.assert_array_equal(a.y_test, df.loc[a.X_test_raw.index, "y"].to_numpy())
    summary = a.get_split_summary()
    assert summary["strategy"] == "random" and summary["task_type"] == "regression"


@pytest.mark.parametrize("fractions", [(0.6, 0.2), (0.6, 0.3, 0.2), (0.8, 0.2, 0.0), (1.2, -0.1, -0.1), ("a", 1, 2)])
def test_invalid_split_fractions(fractions):
    with pytest.raises(SplitValidationError):
        validate_split_fractions(fractions)


def test_custom_valid_fractions_and_too_few_rows():
    df = load_reg(make_df(n=1000))
    s = split_regression_data(df, "y", fractions=(0.5, 0.25, 0.25))
    assert [len(s.X_train_raw), len(s.X_calib_raw), len(s.X_test_raw)] == [500, 250, 250]
    with pytest.raises(SplitValidationError):
        split_regression_data(df.head(80), "y")


def test_validate_splits_detects_leakage_and_bad_partitions():
    ok = [pd.Index(range(0, 40)), pd.Index(range(40, 80)), pd.Index(range(80, 120))]
    validate_splits(*ok)
    with pytest.raises(SplitValidationError, match="share"):
        validate_splits(ok[0], pd.Index(range(30, 70)), ok[2])
    with pytest.raises(SplitValidationError, match="empty"):
        validate_splits(ok[0], pd.Index([]), ok[2])
    with pytest.raises(SplitValidationError, match="at least"):
        validate_splits(ok[0], pd.Index(range(40, 45)), ok[2])
    with pytest.raises(SplitValidationError, match="duplicate"):
        validate_splits(pd.Index(list(range(0, 39)) + [0]), ok[1], ok[2])
    groups = pd.Series([i // 10 for i in range(120)])
    validate_splits(*ok, groups=groups)
    with pytest.raises(SplitValidationError, match="group leakage"):
        validate_splits(pd.Index(range(0, 45)), pd.Index(range(45, 80)), ok[2], groups=groups)
    times = pd.Series(np.arange(120))
    validate_splits(*ok, times=times)
    with pytest.raises(SplitValidationError, match="Temporal"):
        validate_splits(ok[2], ok[1], ok[0], times=times)


def test_group_split_keeps_groups_together_and_drops_group_feature():
    df = make_df(n=900)
    df["customer"] = np.random.RandomState(1).randint(0, 90, len(df))
    cfg = TrainingConfig(task_type=REGRESSION, group_column="customer")
    splits, num, cat, _ = prepare_regression_data(load_reg(df), "y", cfg)
    parts = [splits.X_train_raw.index, splits.X_calib_raw.index, splits.X_test_raw.index]
    sets = [set(df.loc[p, "customer"]) for p in parts]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert "customer" not in splits.feature_names and "customer" not in num + cat
    assert splits.strategy == "group"
    df.loc[df.index[0], "customer"] = np.nan
    with pytest.raises(DatasetValidationError):
        prepare_regression_data(load_reg(df), "y", cfg)


@pytest.mark.parametrize("as_dates", [False, True])
def test_time_split_is_ordered_and_never_splits_a_timestamp(as_dates):
    df = make_df(n=800)
    t = np.repeat(np.arange(160), 5)  # 5 rows share every timestamp
    df["ts"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(t, unit="D") if as_dates else t
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)  # input order is irrelevant
    cfg = TrainingConfig(task_type=REGRESSION, time_column="ts")
    s, _, _, _ = prepare_regression_data(load_reg(df), "y", cfg)
    ts = pd.to_datetime(df["ts"]) if as_dates else df["ts"]
    tr, ca, te = (ts.loc[x.index] for x in (s.X_train_raw, s.X_calib_raw, s.X_test_raw))
    assert tr.max() < ca.min() and ca.max() < te.min()
    assert "ts" not in s.feature_names and s.strategy == "time"


def test_time_split_rejects_unparseable_or_degenerate_time():
    df = make_df(n=600)
    df["ts"] = ["not a date"] * len(df)
    with pytest.raises(SplitValidationError, match="parseable"):
        split_regression_data(df, "y", time_column="ts")
    df["ts"] = 1  # a single time point cannot be ordered into three partitions
    with pytest.raises(SplitValidationError):
        split_regression_data(df, "y", time_column="ts")


def test_split_config_errors():
    df = load_reg(make_df())
    df["g"], df["t"] = 1, 2
    with pytest.raises(SplitValidationError):
        split_regression_data(df, "y", group_column="g", time_column="t")
    with pytest.raises(SplitValidationError):
        split_regression_data(df, "y", group_column="missing")
    with pytest.raises(SplitValidationError, match="duplicate index"):
        split_regression_data(pd.concat([df, df]), "y")


def test_duplicate_records_across_splits_are_reported():
    df = make_df(n=600)
    df = pd.concat([df, df.head(100)], ignore_index=True)
    s = split_regression_data(df, "y", random_state=0)
    assert s.duplicate_rows_across_splits > 0
    assert split_regression_data(make_df(n=600), "y").duplicate_rows_across_splits == 0


# ---------------------------------------------------------------------------
# Preprocessing boundaries (fit on TRAIN only, unknown categories, stable order)
# ---------------------------------------------------------------------------

def test_preprocessing_fits_on_train_only_and_handles_unknown_and_missing():
    df = load_reg(make_df(n=1000))
    splits, num, cat, _ = prepare_regression_data(df, "y")
    assert num == ["x1", "x2"] and cat == ["cat"]
    pre = build_preprocessor(num, cat).fit(splits.X_train_raw)
    names = list(pre.get_feature_names_out())

    calib = splits.X_calib_raw.copy()
    calib.iloc[0, calib.columns.get_loc("cat")] = "never_seen"
    calib.iloc[1, calib.columns.get_loc("x1")] = np.nan
    out = pre.transform(calib)
    assert out.shape[1] == len(names) and np.isfinite(out).all()
    # Unknown category -> all-zero one-hot block (documented strategy: handle_unknown="ignore").
    cat_cols = [i for i, n in enumerate(names) if n.startswith("cat_")]
    assert np.all(out[0, cat_cols] == 0)
    # Statistics come from the train partition only.
    imputer = pre.named_transformers_["numeric"].named_steps["imputer"]
    np.testing.assert_allclose(imputer.statistics_[0], splits.X_train_raw["x1"].median())
    # Column order is stable regardless of the column order of incoming records.
    shuffled = calib[list(reversed(calib.columns))]
    np.testing.assert_allclose(pre.transform(shuffled), out)


# ---------------------------------------------------------------------------
# Classification separation
# ---------------------------------------------------------------------------

def test_classification_split_unchanged_and_regression_loader_rejects_labels():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"x": rng.randn(600), "label": rng.choice(["yes", "no"], 600)})
    clf = load_dataset(df.to_csv(index=False), "label")  # default task
    s = split_data(clf, "label", random_state=42)
    assert s.classes == ["no", "yes"] and len(s.X_train_raw) == 360
    with pytest.raises(InvalidTargetError):
        load_dataset(df.to_csv(index=False), "label", task_type=REGRESSION)
