"""Prediction endpoints (Phase B: true outcome unknown)."""

import uuid
import warnings
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, File, UploadFile

from app.api.deps import normalize_label, parse_csv_bytes, read_upload_limited, require_artifacts
from app.core import sessions
from app.core.config import settings
from app.core.errors import bad_request, unprocessable
from app.schemas.models import BatchPredictResponse, PredictRequest, PredictResponse
from ml.pipeline import RecordValidationError, coerce_records, predict_records

router = APIRouter(prefix="/api/predict", tags=["predict"])


def _score(artifacts: Dict[str, Any], records: pd.DataFrame, allow_extra: bool) -> List[Dict[str, Any]]:
    try:
        X = coerce_records(records, artifacts["feature_schema"], allow_extra_columns=allow_extra)
    except RecordValidationError as exc:
        raise unprocessable(str(exc))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return predict_records(artifacts, X)


def _log_rows(results: List[Dict[str, Any]], records: pd.DataFrame, source: str) -> List[Dict[str, Any]]:
    now = sessions.now_iso()
    raw = records.astype(object).where(records.notna(), None).to_dict(orient="records")
    rows = []
    for res, rec in zip(results, raw):
        res["prediction_id"] = uuid.uuid4().hex
        rows.append({
            "prediction_id": res["prediction_id"],
            "timestamp": now,
            "source": source,
            "prediction": res["prediction"],
            "confidence": res["confidence"],
            "trustworthiness": res["trustworthiness"],
            "decision": res["decision"],
            "warning_codes": [w["code"] for w in res["warnings"]],
            "features": {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()},
        })
    return rows


@router.post("/{session_id}", response_model=PredictResponse)
def predict_single(session_id: str, body: PredictRequest):
    """Score one record: prediction, confidence, trustworthiness, TRUST/REVIEW/REJECT and warnings."""
    artifacts = require_artifacts(session_id)
    if not body.features:
        raise unprocessable("'features' must contain at least one field.")
    records = pd.DataFrame([body.features])
    results = _score(artifacts, records, allow_extra=False)
    sessions.append_jsonl(session_id, "predictions.jsonl", _log_rows(results, records, "single"))
    return results[0]


@router.post("/{session_id}/batch", response_model=BatchPredictResponse)
async def predict_batch(session_id: str, file: UploadFile = File(...)):
    """
    Score a CSV of records. Extra columns are ignored. If the CSV also contains the
    target column, those values are recorded as true outcomes (feedback) for monitoring;
    they are never used to make the predictions.
    """
    artifacts = require_artifacts(session_id)
    df = parse_csv_bytes(await read_upload_limited(file))
    if len(df) == 0:
        raise bad_request("Batch CSV has no rows.")
    if len(df) > settings.max_batch_rows:
        raise bad_request(f"Batch CSV has {len(df)} rows; the limit is {settings.max_batch_rows}.")

    target = artifacts["target_column"]
    actuals = df[target].tolist() if target in df.columns else None
    expected = {f["name"] for f in artifacts["feature_schema"]}
    ignored = [c for c in df.columns if c not in expected]
    features_df = df[[c for c in df.columns if c in expected]]
    missing = [f["name"] for f in artifacts["feature_schema"] if f["name"] not in features_df.columns]
    if missing:
        raise unprocessable(f"Batch CSV is missing required column(s): {missing}")

    results = _score(artifacts, features_df, allow_extra=False)
    sessions.append_jsonl(session_id, "predictions.jsonl", _log_rows(results, features_df, "batch"))

    labels_recorded = 0
    if actuals is not None:
        now = sessions.now_iso()
        fb = [
            {"prediction_id": r["prediction_id"], "actual": normalize_label(a), "timestamp": now}
            for r, a in zip(results, actuals) if a is not None and not pd.isna(a)
        ]
        sessions.append_jsonl(session_id, "feedback.jsonl", fb)
        labels_recorded = len(fb)
        for r, a in zip(results, actuals):
            r["actual"] = None if a is None or pd.isna(a) else normalize_label(a)

    summary = {d: sum(1 for r in results if r["decision"] == d) for d in ["TRUST", "REVIEW", "REJECT"]}
    summary["labels_recorded"] = labels_recorded
    for r in results:
        r.pop("reliability_features", None)
    return {
        "session_id": session_id,
        "count": len(results),
        "ignored_columns": ignored,
        "summary": summary,
        "results": results,
    }
