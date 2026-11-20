"""Pydantic request / response schemas for the ModelDoctor REST API.

The same shapes are mirrored in frontend/src/api/types.js (JSDoc) for the React app.
Deeply nested chart / diagnostic payloads are typed as dicts to keep the contract
flexible; every key that the dashboard relies on is typed explicitly.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["TRUST", "REVIEW", "REJECT"]
FeatureValue = Union[float, int, str, bool, None]


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    code: str


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ColumnSummary(BaseModel):
    name: str
    dtype: Literal["numeric", "categorical"]
    missing_count: int
    unique_count: int
    is_target_candidate: bool


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: List[ColumnSummary]
    preview: List[Dict[str, Any]]
    suggested_target: Optional[str]


class ProfileResponse(BaseModel):
    """Output of ml.data.profiler.profile_dataset plus validation result."""
    model_config = ConfigDict(extra="allow")
    session_id: str
    valid: bool
    validation_error: Optional[str] = None
    row_count: int
    column_count: int
    target_column: str
    detected_task: Optional[str] = None
    class_distribution: Dict[str, int] = {}
    warnings: List[str] = []


class SessionSummary(BaseModel):
    session_id: str
    filename: Optional[str]
    created_at: Optional[str]
    target_column: Optional[str]
    rows: Optional[int]
    status: str
    stage: Optional[str]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    target_column: str = Field(..., min_length=1)
    target_risk: float = Field(0.05, gt=0.0, lt=1.0, description="Max error rate among TRUST predictions")
    min_coverage: float = Field(0.30, gt=0.0, le=1.0, description="Minimum share of predictions to TRUST")
    force_model: Optional[Literal["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "KNN"]] = None
    force_review_on_anomalies: bool = False
    fast: Optional[bool] = Field(None, description="Smaller ensembles for quick demos (server default if omitted)")


class TrainStartResponse(BaseModel):
    session_id: str
    status: str
    message: str


class TrainStatusResponse(BaseModel):
    session_id: str
    status: Literal["uploaded", "pending", "running", "completed", "failed"]
    stage: Optional[str] = None
    stages: List[str]
    progress: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class LeaderboardRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    score: Optional[float]
    train_seconds: Optional[float]
    status: str


class ComparisonRow(BaseModel):
    method: str
    coverage: float
    selective_risk: float
    accepted: int
    total: int


class ReportResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    session_id: str
    task: str
    target_column: str
    classes: List[str]
    selected_model: str
    leaderboard: List[LeaderboardRow]
    reliability_mode: Literal["model2", "confidence_fallback"]
    overall_error_rate: float
    comparison: List[ComparisonRow]
    verdict: str
    auroc: Dict[str, Optional[float]]
    auprc: Dict[str, Optional[float]]
    risk_coverage_curve: Dict[str, List[Dict[str, float]]]
    calibration_curve: Dict[str, List[Dict[str, Any]]]
    decision_distribution: List[Dict[str, Any]]
    thresholds: Dict[str, Any]
    profile: Dict[str, Any]


# ---------------------------------------------------------------------------
# Schema / Predict
# ---------------------------------------------------------------------------

class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    type: Literal["numeric", "categorical"]
    missing_rate: float
    integer: Optional[bool] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    categories: Optional[List[str]] = None
    most_frequent: Optional[str] = None


class SchemaResponse(BaseModel):
    session_id: str
    target_column: str
    classes: List[str]
    features: List[FeatureSpec]


class PredictRequest(BaseModel):
    features: Dict[str, FeatureValue] = Field(
        ...,
        description="Raw feature values keyed by column name. Use null for an unknown value.",
        examples=[{"annual_income": 42000, "age": 35, "loan_amount": 15000, "employment_type": "self_employed"}],
    )


class Warning_(BaseModel):
    code: str
    message: str


class PredictResponse(BaseModel):
    prediction_id: str
    prediction: str
    confidence: float
    trustworthiness: float
    decision: Decision
    warnings: List[Warning_]
    class_probabilities: Dict[str, float]
    reliability_features: Dict[str, Any]


class BatchPredictResponse(BaseModel):
    session_id: str
    count: int
    ignored_columns: List[str]
    summary: Dict[str, int]
    results: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Feedback / Monitoring
# ---------------------------------------------------------------------------

class FeedbackItem(BaseModel):
    prediction_id: str = Field(..., min_length=1)
    actual: Union[str, int, float, bool]


class FeedbackRequest(BaseModel):
    items: List[FeedbackItem] = Field(..., min_length=1)


class FeedbackResponse(BaseModel):
    accepted: int
    rejected: List[Dict[str, str]]
    labeled_total: int


class PredictionLogRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    prediction_id: str
    timestamp: str
    source: str
    prediction: str
    confidence: float
    trustworthiness: float
    decision: Decision
    actual: Optional[str] = None


class MonitoringResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    session_id: str
    total_predictions: int
    labeled_predictions: int
    coverage: Optional[float]
    decision_distribution: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]
    observed_vs_estimated: Dict[str, Any]
    alerts: List[Dict[str, str]]
    retraining_recommended: bool
    recent_predictions: List[PredictionLogRow]
