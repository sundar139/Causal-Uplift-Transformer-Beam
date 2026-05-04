from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from causal_uplift.bundle import InferenceBundle, load_inference_bundle, new_request_id
from causal_uplift.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    SinglePredictionRequest,
    VersionResponse,
)

SERVICE_NAME = "causal-uplift-transformer-beam"
SERVICE_VERSION = "0.2.0"

logger = logging.getLogger("causal_uplift.serve")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

BUNDLE: InferenceBundle | None = None
MODEL_LOADED = False
STARTUP_ERROR: str | None = None


def _bundle_dir() -> Path:
    return Path(os.getenv("PRODUCTION_BUNDLE_DIR", "models/production"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global BUNDLE, MODEL_LOADED, STARTUP_ERROR
    try:
        BUNDLE = load_inference_bundle(_bundle_dir())
        MODEL_LOADED = True
        STARTUP_ERROR = None
        logger.info(
            "production_bundle_loaded model=%s version=%s",
            BUNDLE.model_name,
            BUNDLE.created_at_utc,
        )
    except Exception as exc:
        BUNDLE = None
        MODEL_LOADED = False
        STARTUP_ERROR = str(exc)
        logger.exception("production_bundle_load_failed")
    yield


app = FastAPI(
    title="Causal Uplift Production API",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "status": "ok"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED,
        model_name=BUNDLE.model_name if BUNDLE else None,
        model_version=BUNDLE.created_at_utc if BUNDLE else None,
    )


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(service=SERVICE_NAME, version=SERVICE_VERSION)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    bundle = _require_bundle()
    return ModelInfoResponse(
        model_name=bundle.model_name,
        model_version=bundle.created_at_utc,
        selection_metric=str(bundle.metadata.get("selection_metric", "qini_auc")),
        dataset_variant=str(bundle.metadata.get("dataset_variant", "")),
        metrics={
            "qini_auc": float(bundle.metadata.get("qini_auc", 0.0)),
            "uplift_auc": float(bundle.metadata.get("uplift_auc", 0.0)),
            "policy_gain_top10": float(bundle.metadata.get("policy_gain_top10", 0.0)),
            "policy_gain_top20": float(bundle.metadata.get("policy_gain_top20", 0.0)),
            "policy_gain_top30": float(bundle.metadata.get("policy_gain_top30", 0.0)),
            "treatment_response_auc": float(bundle.metadata.get("treatment_response_auc", 0.0)),
        },
        feature_count=int(bundle.feature_schema["feature_count"]),
        required_columns=bundle.required_columns,
    )


def _require_bundle() -> InferenceBundle:
    if BUNDLE is None or not MODEL_LOADED:
        detail = "Production model bundle is not loaded."
        if STARTUP_ERROR:
            detail = f"{detail} {STARTUP_ERROR}"
        raise HTTPException(status_code=503, detail=detail)
    return BUNDLE


def _prediction_response(
    bundle: InferenceBundle,
    records: list[dict[str, float]],
    request_id: str,
) -> list[dict[str, object]]:
    try:
        return bundle.predict_records(records)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict_uplift", response_model=PredictionResponse)
def predict_uplift(payload: SinglePredictionRequest, request: Request) -> PredictionResponse:
    bundle = _require_bundle()
    request_id = new_request_id()
    logger.info("predict_uplift request_id=%s client=%s", request_id, request.client)
    prediction = _prediction_response(bundle, [payload.features], request_id)[0]
    return PredictionResponse(
        request_id=request_id,
        model_name=bundle.model_name,
        model_version=bundle.created_at_utc,
        prediction=prediction,
    )


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    bundle = _require_bundle()
    request_id = new_request_id()
    logger.info(
        "predict_batch request_id=%s record_count=%s client=%s",
        request_id,
        len(payload.records),
        request.client,
    )
    predictions = _prediction_response(bundle, payload.records, request_id)
    return BatchPredictionResponse(
        request_id=request_id,
        model_name=bundle.model_name,
        model_version=bundle.created_at_utc,
        predictions=predictions,
    )
