from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _validate_numeric_features(features: dict[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in features.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature {key!r} must be numeric") from exc
        if numeric != numeric or numeric in {float("inf"), float("-inf")}:
            raise ValueError(f"Feature {key!r} must be finite")
        cleaned[str(key)] = numeric
    return cleaned


class SinglePredictionRequest(BaseModel):
    features: dict[str, float]

    @field_validator("features", mode="before")
    @classmethod
    def validate_features(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError("features must be an object mapping feature names to numeric values")
        return _validate_numeric_features(value)


class BatchPredictionRequest(BaseModel):
    records: list[dict[str, float]] = Field(min_length=1)

    @field_validator("records", mode="before")
    @classmethod
    def validate_records(cls, value: Any) -> list[dict[str, float]]:
        if not isinstance(value, list):
            raise ValueError("records must be a list of feature objects")
        return [_validate_numeric_features(record) for record in value]


class PredictionItem(BaseModel):
    treatment_probability: float
    control_probability: float
    uplift: float
    recommend_treatment: bool


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    prediction: PredictionItem


class BatchPredictionResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    selection_metric: str
    dataset_variant: str
    metrics: dict[str, float]
    feature_count: int
    required_columns: list[str]


class VersionResponse(BaseModel):
    service: str
    version: str
