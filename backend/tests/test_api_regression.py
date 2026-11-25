"""Phase 6 API tests: regression training, prediction, artifacts, validation and fallback,
plus checks that classification routes keep their contracts. Synthetic data, fast mode."""

import io
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import sessions
from app.core.config import settings
from app.main import app
from ml.inference import ModelLoadError, RegressionInferenceEngine, engine_from_directory, load_artifacts

warnings.filterwarnings("ignore")
LOANS = Path(__file__).resolve().parents[2] / "sample_data" / "loans.csv"


def reg_df(n=900, seed=0):
    rng = np.random.RandomState(seed)
    x1, x2 = rng.randn(n), rng.uniform(-3, 3, n)
    region = rng.choice(["north", "south", "online"], n)
    y = 2 * x1 + x2 ** 2 + rng.randn(n) * np.where(region == "online", 3.0, 0.4) - 10
    return pd.DataFrame({"x1": x1, "x2": x2, "region": region, "price": y})


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    original = settings.storage_dir, settings.model_session
    settings.storage_dir = tmp_path_factory.mktemp("storage_reg")
    settings.model_session = None
    with TestClient(app) as c:
        yield c
    settings.storage_dir, settings.model_session = original


def _upload(client, df, name="data.csv"):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/api/datasets/upload", files={"file": (name, buf, "text/csv")})
    assert r.status_code == 201, r.text
    return r.json()


def _train(client, sid, **body):
    return client.post(f"/api/train/{sid}", json={"fast": True, **body})


@pytest.fixture(scope="module")
def reg_session(client):
    df = reg_df()
    sid = _upload(client, df)["session_id"]
    r = _train(client, sid, target_column="price", task_type="regression")
    assert r.status_code == 202, r.text
    status = client.get(f"/api/train/{sid}/status").json()
    assert status["status"] == "completed", status
    return sid, df


def _rec(df, i=0):
    rec = df.drop(columns="price").iloc[i].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}


# ---------------------------------------------------------------------------
# Upload / profile / training validation
# ---------------------------------------------------------------------------

def test_upload_flags_regression_candidates_without_choosing_the_task(client):
    body = _upload(client, reg_df())
    cols = {c["name"]: c for c in body["columns"]}
    assert cols["price"]["is_regression_candidate"] and not cols["region"]["is_regression_candidate"]


def test_profile_by_task_type(client):
    sid = _upload(client, reg_df())["session_id"]
    clf = client.get(f"/api/datasets/{sid}/profile", params={"target_column": "price"}).json()
    assert not clf["valid"] and clf["suggested_task"] == "regression"
    reg = client.get(f"/api/datasets/{sid}/profile",
                     params={"target_column": "price", "task_type": "regression"}).json()
    assert reg["valid"] and reg["detected_task"] == "regression"
    assert reg["class_distribution"] == {} and reg["target_summary"]["negative_share"] > 0.5
    bad = client.get(f"/api/datasets/{sid}/profile", params={"target_column": "region", "task_type": "regression"})
    assert bad.json()["valid"] is False and "numeric" in bad.json()["validation_error"]
    r = client.get(f"/api/datasets/{sid}/profile", params={"target_column": "price", "task_type": "forecast"})
    assert r.status_code == 400


@pytest.mark.parametrize("body, fragment", [
    ({"target_column": "price"}, "Only classification targets"),                      # task never inferred
    ({"target_column": "region", "task_type": "regression"}, "must be numeric"),
    ({"target_column": "price", "task_type": "regression", "force_model": "Logistic Regression"}, "not available"),
    ({"target_column": "price", "task_type": "regression", "group_column": "nope"}, "not found"),
    ({"target_column": "price", "task_type": "regression", "group_column": "x1", "time_column": "x2"}, "either"),
])
def test_invalid_regression_training_requests(client, body, fragment):
    sid = _upload(client, reg_df())["session_id"]
    r = _train(client, sid, **body)
    assert r.status_code == 400 and fragment in r.json()["detail"]


def test_regression_only_options_rejected_for_classification(client):
    sid = _upload(client, pd.read_csv(LOANS).head(600))["session_id"]
    r = _train(client, sid, target_column="default", error_tolerance=1.0)
    assert r.status_code == 400 and "regression" in r.json()["detail"]
    r = _train(client, sid, target_column="default", force_model="Ridge Regression")
    assert r.status_code == 400
    r = _train(client, sid, target_column="default", error_tolerance=-1, task_type="regression")
    assert r.status_code == 422


def test_invalid_regression_target_values(client):
    df = reg_df()
    df.loc[3, "price"] = np.nan
    sid = _upload(client, df)["session_id"]
    r = _train(client, sid, target_column="price", task_type="regression")
    assert r.status_code == 400 and "missing" in r.json()["detail"] and "Traceback" not in r.text


# ---------------------------------------------------------------------------
# Report / schema / predict
# ---------------------------------------------------------------------------

def test_regression_report_and_schema(client, reg_session):
    sid, _ = reg_session
    rep = client.get(f"/api/report/{sid}").json()
    assert rep["task"] == "regression" and rep["leakage_audit"]["passed"]
    assert {"model1", "model2", "uncertainty"} <= set(rep["test_evaluation"])
    assert "classes" not in rep and "auroc" not in rep
    schema = client.get(f"/api/schema/{sid}").json()
    assert schema["task_type"] == "regression" and "classes" not in schema
    assert [f["name"] for f in schema["features"]] == ["x1", "x2", "region"]


def test_regression_single_prediction_response(client, reg_session):
    sid, df = reg_session
    r = client.post(f"/api/predict/{sid}", json={"features": _rec(df)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task_type"] == "regression"
    assert isinstance(body["prediction"], float) and body["estimated_abs_error"] >= 0
    assert body["decision"] in ("TRUST", "REVIEW", "REJECT") and body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    for forbidden in ("class_probabilities", "confidence", "trustworthiness", "classes"):
        assert forbidden not in body
    assert "not a guaranteed bound" in " ".join(body["reasons"])


@pytest.mark.parametrize("mutate, status", [
    (lambda r: {**r, "x1": "abc"}, 422),
    (lambda r: {k: v for k, v in r.items() if k != "x2"}, 422),
    (lambda r: {**r, "extra": 1}, 422),
])
def test_invalid_numeric_inputs(client, reg_session, mutate, status):
    sid, df = reg_session
    r = client.post(f"/api/predict/{sid}", json={"features": mutate(_rec(df))})
    assert r.status_code == status and "Traceback" not in r.text


def test_out_of_range_input_is_not_trusted(client, reg_session):
    sid, df = reg_session
    body = client.post(f"/api/predict/{sid}", json={"features": {**_rec(df), "x1": 1e6}}).json()
    assert body["decision"] != "TRUST"
    assert {"OUT_OF_RANGE", "UNSUPPORTED_ESTIMATE"} & {w["code"] for w in body["warnings"]}


def test_batch_prediction_records_numeric_outcomes_and_monitoring(client, reg_session):
    sid, df = reg_session
    batch = df.head(30).copy()
    batch.loc[batch.index[0], "price"] = np.nan
    buf = io.BytesIO(batch.to_csv(index=False).encode())
    r = client.post(f"/api/predict/{sid}/batch", files={"file": ("b.csv", buf, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 30 and body["summary"]["labels_recorded"] == 29
    assert isinstance(body["results"][1]["actual"], float) and body["results"][0]["actual"] is None

    mon = client.get(f"/api/monitoring/{sid}").json()
    assert mon["task_type"] == "regression" and mon["labeled_predictions"] == 29
    ove = mon["observed_vs_estimated"]
    assert ove["observed_mae"] is not None and ove["mean_estimated_abs_error"] is not None
    assert isinstance(mon["recent_predictions"][0]["prediction"], float)


def test_regression_feedback_validation(client, reg_session):
    sid, df = reg_session
    pid = client.post(f"/api/predict/{sid}", json={"features": _rec(df, 2)}).json()["prediction_id"]
    r = client.post(f"/api/feedback/{sid}", json={"items": [
        {"prediction_id": pid, "actual": 12.5},
        {"prediction_id": pid, "actual": "high"},
        {"prediction_id": "nope", "actual": 1.0},
    ]}).json()
    assert r["accepted"] == 1 and len(r["rejected"]) == 2
    assert any("finite number" in x["reason"] for x in r["rejected"])


# ---------------------------------------------------------------------------
# Deployed /api/v1 with a regression model
# ---------------------------------------------------------------------------

def test_v1_serves_a_regression_model(client, reg_session):
    sid, df = reg_session
    assert client.put("/api/v1/model/active", json={"session_id": sid}).status_code == 200
    info = client.get("/api/v1/model/info").json()
    assert info["task"] == "regression" and "classes" not in info and info["error_tolerance"]["value"] > 0
    one = client.post("/api/v1/predict", json={"features": _rec(df, 4)}).json()
    assert one["task_type"] == "regression" and "class_probabilities" not in one and "model_confidence" not in one
    batch = client.post("/api/v1/predict/batch", json={"records": [_rec(df, i) for i in range(5)]}).json()
    assert batch["count"] == 5 and batch["task_type"] == "regression"
    assert all(isinstance(x["prediction"], float) for x in batch["results"])


# ---------------------------------------------------------------------------
# Artifacts: independence, missing Model 2, fallback
# ---------------------------------------------------------------------------

def test_artifacts_load_into_the_matching_engine_only(reg_session, tmp_path):
    sid, _ = reg_session
    path = sessions.session_dir(sid)
    assert isinstance(engine_from_directory(path), RegressionInferenceEngine)
    art = load_artifacts(path)
    assert art["task_type"] == "regression" and "classes" not in art
    from ml.inference import InferenceEngine
    with pytest.raises(ModelLoadError):
        InferenceEngine(art)  # never into classification inference


def test_missing_model2_artifact_is_a_controlled_error(client, reg_session, tmp_path):
    sid, df = reg_session
    copy_id = "f" * 32
    dst = sessions.storage_root() / copy_id
    shutil.copytree(sessions.session_dir(sid), dst)
    (dst / "model2.joblib").unlink()
    r = client.post(f"/api/predict/{copy_id}", json={"features": _rec(df)})
    assert r.status_code in (409, 503) and "Model 2" in r.json()["detail"] and "Traceback" not in r.text
    shutil.rmtree(dst)


def test_fallback_bundle_serves_without_model2(client, monkeypatch):
    import functools

    import ml.pipeline as pl
    from ml.reliability.model2_regression import train_regression_model2

    monkeypatch.setattr(pl, "train_regression_model2", functools.partial(train_regression_model2, min_rows=10 ** 6))
    df = reg_df(seed=4)
    sid = _upload(client, df)["session_id"]
    assert _train(client, sid, target_column="price", task_type="regression").status_code == 202
    assert client.get(f"/api/train/{sid}/status").json()["status"] == "completed"
    assert not (sessions.session_dir(sid) / "model2.joblib").exists()
    body = client.post(f"/api/predict/{sid}", json={"features": _rec(df)}).json()
    assert body["reliability_mode"] == "constant_fallback"
    assert body["error_estimate_source"] == "calibration_mean_error"
    assert client.get(f"/api/report/{sid}").json()["model2"]["status"] == "insufficient_data"


def test_model2_failure_keeps_model1_prediction(client, reg_session, monkeypatch):
    sid, df = reg_session
    engine = sessions.load_engine(sid)
    clean = client.post(f"/api/predict/{sid}", json={"features": _rec(df, 7)}).json()
    m2 = engine.artifacts["model2"]
    monkeypatch.setattr(type(m2), "predict_scaled_error", lambda self, f: (_ for _ in ()).throw(RuntimeError("x")))
    broken = client.post(f"/api/predict/{sid}", json={"features": _rec(df, 7)}).json()
    assert broken["prediction"] == pytest.approx(clean["prediction"], rel=1e-12)
    assert broken["decision"] != "TRUST" and broken["error_estimate_source"] == "calibration_mean_error"


# ---------------------------------------------------------------------------
# Classification contracts unchanged
# ---------------------------------------------------------------------------

def test_classification_routes_keep_their_contract(client):
    df = pd.read_csv(LOANS).head(800)
    sid = _upload(client, df)["session_id"]
    assert _train(client, sid, target_column="default").status_code == 202
    assert client.get(f"/api/train/{sid}/status").json()["status"] == "completed"
    rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in df.drop(columns="default").iloc[0].items()}
    body = client.post(f"/api/predict/{sid}", json={"features": rec}).json()
    assert {"prediction", "confidence", "trustworthiness", "class_probabilities"} <= set(body)
    assert "task_type" not in body and "estimated_abs_error" not in body
    assert "classes" in client.get(f"/api/schema/{sid}").json()
    assert "auroc" in client.get(f"/api/report/{sid}").json()
    mon = client.get(f"/api/monitoring/{sid}").json()
    assert "task_type" not in mon and "mean_confidence" in mon
