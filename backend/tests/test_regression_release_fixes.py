"""Phase 7 (release audit) regression tests: one test per confirmed defect."""

import io
import warnings

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from ml.data.splitter import SplitValidationError, split_regression_data
from ml.inference import engine_for
from ml.models.regression_selector import select_regression_model
from ml.pipeline import RecordValidationError, TrainingConfig, train_modeldoctor
from ml.reliability.mitigation_regression import RegressionMitigationEngine, tune_regression_thresholds

warnings.filterwarnings("ignore")


def test_model1_selection_is_scale_free_for_tiny_targets():
    """Defect: rounded CV scores tied every candidate at 0.0 for targets ~1e-9."""
    rng = np.random.RandomState(0)
    a = rng.uniform(-3, 3, 600)
    X, y = pd.DataFrame({"a": a}), np.sin(3 * a) + 0.05 * rng.randn(600)
    big = select_regression_model(X, y, ["a"], [], fast=True)
    tiny = select_regression_model(X, y * 1e-9, ["a"], [], fast=True)
    assert tiny["winner_name"] == big["winner_name"] != "Ridge Regression"
    assert all(r["mae_mean"] > 0 for r in tiny["leaderboard"])


def test_reject_band_survives_perfect_separation():
    """Defect: REJECT was disabled exactly when the high-error band separated perfectly."""
    s = np.r_[np.linspace(0, 1, 100), np.linspace(2, 3, 100)]
    e = np.r_[np.zeros(100), np.full(100, 5.0)]
    res = tune_regression_thresholds(s, e, tol_scaled=1.0, target_risk=0.05, min_coverage=0.3)
    bands = {b["name"]: b for b in res["band_metadata"]}
    # TRUST = largest band meeting 5% (it may include a few high rows); everything above is REJECT.
    assert bands["TRUST"]["failure_rate"] <= 0.05 and bands["TRUST"]["count"] >= 100
    assert res["reject_threshold"] == pytest.approx(res["trust_threshold"])  # previously None
    assert bands["REVIEW"]["count"] == 0
    assert bands["REJECT"]["count"] == 200 - bands["TRUST"]["count"] and bands["REJECT"]["failure_rate"] == 1.0
    eng = RegressionMitigationEngine(res, 1.0, {"value": 1.0, "scaled": 1.0, "source": "configured"})
    t = res["trust_threshold"]
    assert [eng.decide(x) for x in (0.5, t, 2.9)] == ["TRUST", "TRUST", "REJECT"]


@pytest.mark.parametrize("n_keys, kind", [(4, "group"), (7, "time")])
def test_too_few_groups_or_times_fail_early_with_a_clear_message(n_keys, kind):
    """Defect: calibration got 1 group / time point and Model 2 CV crashed after Model 1 trained."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"x": rng.randn(900), "k": np.arange(900) % n_keys})
    df["y"] = 2 * df.x + rng.randn(900)
    kw = {"group_column": "k"} if kind == "group" else {"time_column": "k"}
    with pytest.raises(SplitValidationError) as exc:
        split_regression_data(df, "y", **kw)
    assert "at least" in str(exc.value)


@pytest.mark.parametrize("n_groups", [5, 12])
def test_enough_groups_still_train(n_groups):
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"x": rng.randn(900), "g": np.arange(900) % n_groups})
    df["y"] = 2 * df.x + rng.randn(900)
    _, rep = train_modeldoctor(df, "y", TrainingConfig(task_type="regression", fast=True, group_column="g"))
    assert rep["leakage_audit"]["passed"]


def _forest_bundle():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"x1": rng.randn(700), "x2": rng.randn(700)})
    df["y"] = np.sin(2 * df.x1) + df.x2 ** 2 + 0.1 * rng.randn(700)
    art, _ = train_modeldoctor(df, "y", TrainingConfig(task_type="regression", fast=True, force_model="Random Forest"))
    return df, art


def test_inputs_beyond_float32_range_are_rejected_not_crashing():
    """Defect: 1e39 passed validation and crashed tree models (HTTP 500)."""
    _, art = _forest_bundle()
    eng = engine_for(art)
    with pytest.raises(RecordValidationError, match="too large"):
        eng.predict([{"x1": 1e39, "x2": 0.0}])
    ok = eng.predict([{"x1": 1e20, "x2": 0.0}])[0]  # finite and representable: scored, never trusted
    assert ok["decision"] != "TRUST"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    original = settings.storage_dir, settings.model_session
    settings.storage_dir = tmp_path_factory.mktemp("storage_fixes")
    settings.model_session = None
    with TestClient(app) as c:
        yield c
    settings.storage_dir, settings.model_session = original


def _upload(client, df):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    return client.post("/api/datasets/upload", files={"file": ("d.csv", buf, "text/csv")}).json()["session_id"]


def test_api_huge_input_is_422_and_bad_group_column_is_400(client):
    df, _ = _forest_bundle()
    sid = _upload(client, df)
    r = client.post(f"/api/train/{sid}", json={"target_column": "y", "task_type": "regression", "fast": True,
                                               "force_model": "Random Forest"})
    assert r.status_code == 202
    r = client.post(f"/api/predict/{sid}", json={"features": {"x1": 1e39, "x2": 0.0}})
    assert r.status_code == 422 and "too large" in r.json()["detail"]

    few = df.assign(g=np.arange(len(df)) % 4)
    sid2 = _upload(client, few)
    r = client.post(f"/api/train/{sid2}", json={"target_column": "y", "task_type": "regression", "group_column": "g"})
    assert r.status_code == 400 and "at least" in r.json()["detail"]


def test_batch_reports_rejected_numeric_outcomes(client):
    df, _ = _forest_bundle()
    sid = _upload(client, df)
    client.post(f"/api/train/{sid}", json={"target_column": "y", "task_type": "regression", "fast": True})
    batch = df.head(10).astype({"y": object})
    batch.loc[batch.index[0], "y"] = "oops"
    buf = io.BytesIO(batch.to_csv(index=False).encode())
    body = client.post(f"/api/predict/{sid}/batch", files={"file": ("b.csv", buf, "text/csv")}).json()
    assert body["summary"]["labels_recorded"] == 9 and body["summary"]["labels_rejected"] == 1
