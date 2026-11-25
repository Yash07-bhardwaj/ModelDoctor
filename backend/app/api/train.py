"""Training endpoints: start background training and poll status."""

import warnings

from fastapi import APIRouter, BackgroundTasks

from app.core import sessions, training
from app.core.config import settings
from app.core.errors import bad_request, conflict
from app.schemas.models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    TrainRequest,
    TrainStartResponse,
    TrainStatusResponse,
)
from ml.data.loader import DatasetValidationError, load_dataset
from ml.pipeline import STAGES, TrainingConfig, prepare_regression_data

router = APIRouter(prefix="/api/train", tags=["training"])


@router.post("/{session_id}", response_model=TrainStartResponse, status_code=202)
def start_training(session_id: str, body: TrainRequest, background_tasks: BackgroundTasks):
    """Validate the target column, then run the full ModelDoctor setup as a background task."""
    sessions.session_dir(session_id)
    if training.is_running(session_id):
        raise conflict("Training is already running for this session.")

    _validate_task_options(body)
    # Fail fast on invalid targets so the client gets a 400 instead of a failed job
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_dataset(str(sessions.raw_csv_path(session_id)), body.target_column, task_type=body.task_type)
    except DatasetValidationError as exc:
        raise bad_request(str(exc))
    for col in (body.group_column, body.time_column):
        if col is not None and col not in df.columns:
            raise bad_request(f"Column '{col}' not found in the dataset.")
    if body.task_type == "regression" and (body.group_column or body.time_column):
        try:  # cheap: preparation + split only, so an unusable group / time column is a 400 now
            prepare_regression_data(df, body.target_column, TrainingConfig(
                task_type="regression", group_column=body.group_column, time_column=body.time_column))
        except DatasetValidationError as exc:
            raise bad_request(str(exc))

    if not training.try_claim(session_id):
        raise conflict("Training is already running for this session.")

    extra = {}
    if body.task_type == "regression":
        extra = {
            "task_type": "regression",
            "error_tolerance": body.error_tolerance,
            "group_column": body.group_column,
            "time_column": body.time_column,
        }
        if body.regression_metric is not None:
            extra["regression_metric"] = body.regression_metric
    config = TrainingConfig(
        target_risk=body.target_risk,
        min_coverage=body.min_coverage,
        force_model=body.force_model,
        force_review_on_anomalies=body.force_review_on_anomalies,
        fast=settings.fast_training if body.fast is None else body.fast,
        **extra,
    )
    sessions.clear_training_outputs(session_id)
    sessions.update_meta(session_id, target_column=body.target_column, task_type=body.task_type,
                         config=config.to_dict())
    sessions.set_status(session_id, "pending", message="Training queued.", progress=0.0)
    background_tasks.add_task(training.run_training, session_id, body.target_column, config)
    return {"session_id": session_id, "status": "pending", "message": "Training started."}


def _validate_task_options(body: TrainRequest) -> None:
    """Reject options that do not belong to the requested task (never silently ignore them)."""
    models = REGRESSION_MODELS if body.task_type == "regression" else CLASSIFICATION_MODELS
    if body.force_model is not None and body.force_model not in models:
        raise bad_request(f"Model '{body.force_model}' is not available for {body.task_type}. "
                          f"Choose one of: {', '.join(models)}.")
    if body.task_type == "classification":
        used = [k for k in ("regression_metric", "error_tolerance", "group_column", "time_column")
                if getattr(body, k) is not None]
        if used:
            raise bad_request(f"Option(s) {used} apply to task_type='regression' only.")
    elif body.group_column is not None and body.time_column is not None:
        raise bad_request("Use either group_column or time_column, not both.")


@router.get("/{session_id}/status", response_model=TrainStatusResponse)
def training_status(session_id: str):
    status = sessions.get_status(session_id)
    return {**status, "session_id": session_id, "stages": STAGES}
