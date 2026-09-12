"""Pydantic request/response models for the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """One window per retained sensor, each exactly ``window_len`` raw
    samples -- the same shape the model bundle was trained on
    (``ModelBundle.sensor_names`` / ``window_len``, both echoed by
    ``GET /health``)."""

    sensors: dict[str, list[float]] = Field(
        ..., description="sensor name -> raw samples for this window"
    )

    @field_validator("sensors")
    @classmethod
    def not_empty(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        if not v:
            raise ValueError("at least one sensor window is required")
        return v


class PredictBatchRequest(BaseModel):
    windows: list[PredictRequest]


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    conformal_set: list[str]
    is_silent: bool
    warnings: list[str] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    predicted_class: str
    top_features: list[dict]
    method: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    classes: list[str]
    sensor_names: list[str]
    window_len: int
    sample_rate_hz: int
    bundle_created_at: datetime


class AuditSummaryResponse(BaseModel):
    n_windows: int
    class_balance_verdict: str
    recommended_sensors: list[str]
    excluded_sensors: dict[str, str]


class DriftSummaryResponse(BaseModel):
    n_alarms: int
    n_reference: int
    n_current: int
    top_features: list[dict]
