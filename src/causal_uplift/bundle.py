from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from causal_uplift.baselines import UpliftPrediction
from causal_uplift.config import AppConfig, TrainingConfig, load_training_config
from causal_uplift.data import resolve_materialization_paths
from causal_uplift.preprocessing import NumericFeaturePreprocessor
from causal_uplift.transformer import FTTransformerUpliftModel, TorchUpliftTrainer

PROJECT_NAME = "causal-uplift-transformer-beam"
DEFAULT_BUNDLE_DIR = Path("models/production")
SUPPORTED_CHAMPIONS = {"s_learner_logistic", "ft_transformer"}


@dataclass(slots=True)
class InferenceBundle:
    bundle_dir: Path
    model: Any
    preprocessor: NumericFeaturePreprocessor
    metadata: dict[str, Any]
    feature_schema: dict[str, Any]

    @property
    def model_name(self) -> str:
        return str(self.metadata["champion_model"])

    @property
    def created_at_utc(self) -> str:
        return str(self.metadata["created_at_utc"])

    @property
    def required_columns(self) -> list[str]:
        return [str(column) for column in self.feature_schema["required_columns"]]

    def validate_features(self, records: list[dict[str, float]]) -> None:
        missing_by_row: dict[int, list[str]] = {}
        for idx, record in enumerate(records):
            missing = [column for column in self.required_columns if column not in record]
            if missing:
                missing_by_row[idx] = missing
        if missing_by_row:
            raise ValueError(f"Missing required feature columns: {missing_by_row}")

    def predict_records(self, records: list[dict[str, float]]) -> list[dict[str, Any]]:
        self.validate_features(records)
        frame = pd.DataFrame(records)
        transformed = self.preprocessor.transform(frame)
        transformed_frame = pd.DataFrame(transformed, columns=self.preprocessor.feature_columns)

        if self.model_name == "ft_transformer":
            prediction = self._predict_transformer(transformed)
        else:
            prediction = self.model.predict_uplift(transformed_frame)

        return [
            {
                "treatment_probability": float(treatment_probability),
                "control_probability": float(control_probability),
                "uplift": float(uplift),
                "recommend_treatment": bool(uplift > 0.0),
            }
            for treatment_probability, control_probability, uplift in zip(
                prediction.treatment_proba,
                prediction.control_proba,
                prediction.uplift,
                strict=True,
            )
        ]

    def _predict_transformer(self, transformed: np.ndarray) -> UpliftPrediction:
        return self.model.predict_uplift(transformed)


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_report_path(training_config: TrainingConfig, artifact_root: Path) -> Path:
    variant = training_config.data.dataset_variant
    return artifact_root / "reports" / variant / "best_model_summary.json"


def resolve_source_model_path(
    champion_model: str,
    training_config: TrainingConfig,
    model_root: Path,
) -> Path:
    if champion_model == "ft_transformer":
        return model_root / training_config.artifacts.best_model_name
    return model_root / f"{champion_model}.joblib"


def _feature_columns_from_train_split(training_config: TrainingConfig) -> tuple[list[str], Path]:
    paths = resolve_materialization_paths(training_config)["processed_paths"]
    train_path: Path = paths["train"]
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing materialized training split at {train_path}. "
            "Run `uv run python -m causal_uplift.data materialize --config "
            "configs/training_full.yaml` before building the inference bundle."
        )

    frame = pd.read_parquet(train_path)
    ignored = {
        training_config.data.target_col,
        training_config.data.treatment_col,
        "split",
    }
    return [column for column in frame.columns if column not in ignored], train_path


def fit_preprocessor_for_bundle(training_config: TrainingConfig) -> NumericFeaturePreprocessor:
    paths = resolve_materialization_paths(training_config)["processed_paths"]
    train_path: Path = paths["train"]
    feature_columns, _ = _feature_columns_from_train_split(training_config)
    frame = pd.read_parquet(train_path, columns=feature_columns)
    return NumericFeaturePreprocessor().fit(frame)


def build_feature_schema(
    training_config: TrainingConfig,
    preprocessor: NumericFeaturePreprocessor,
) -> dict[str, Any]:
    feature_names = list(preprocessor.feature_columns)
    return {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "required_columns": feature_names,
        "target_column": training_config.data.target_col,
        "treatment_column": training_config.data.treatment_col,
        "dtype_policy": {
            "accepted": "numeric values coercible to float",
            "missing": "rejected for required feature columns",
            "invalid": "rejected when a value cannot be parsed as a finite number",
        },
    }


def build_example_request(feature_names: list[str]) -> dict[str, Any]:
    return {
        "features": {
            feature_name: round((idx + 1) / 10.0, 4)
            for idx, feature_name in enumerate(feature_names)
        }
    }


def build_prediction_contract() -> dict[str, Any]:
    return {
        "endpoint": "/predict_uplift",
        "input_format": {"features": {"feature_name": "numeric_value"}},
        "output_format": {
            "request_id": "string",
            "model_name": "string",
            "model_version": "created_at_utc timestamp",
            "prediction": {
                "treatment_probability": "float",
                "control_probability": "float",
                "uplift": "float",
                "recommend_treatment": "boolean",
            },
        },
        "example_request_path": "models/production/example_request.json",
    }


def _load_training_config_payload(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_model_metadata(
    report_payload: dict[str, Any],
    training_config_payload: dict[str, Any],
    config_path: Path,
    report_path: Path,
    source_model_path: Path,
) -> dict[str, Any]:
    source_predictions_path = Path(str(report_payload.get("source_predictions_path", "")))
    source_metrics_path = Path(str(report_payload.get("source_metrics_path", "")))
    return {
        "project_name": PROJECT_NAME,
        "champion_model": report_payload["best_model"],
        "selection_metric": report_payload.get("selection_metric", "qini_auc"),
        "qini_auc": float(report_payload.get("qini_auc", 0.0)),
        "uplift_auc": float(report_payload.get("uplift_auc", 0.0)),
        "policy_gain_top10": float(report_payload.get("policy_gain_top10", 0.0)),
        "policy_gain_top20": float(report_payload.get("policy_gain_top20", 0.0)),
        "policy_gain_top30": float(report_payload.get("policy_gain_top30", 0.0)),
        "treatment_response_auc": float(report_payload.get("treatment_response_auc", 0.0)),
        "dataset_variant": "full",
        "training_config": {
            "path": relative_path(config_path),
            "payload": training_config_payload,
        },
        "created_at_utc": utc_now(),
        "source_report_path": relative_path(report_path),
        "source_model_path": relative_path(source_model_path),
        "source_predictions_path": relative_path(source_predictions_path),
        "source_metrics_path": relative_path(source_metrics_path),
    }


def _resolve_transformer_metadata(
    training_config: TrainingConfig, artifact_root: Path
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "embedding_dim": training_config.transformer.embedding_dim,
        "num_layers": training_config.transformer.num_layers,
        "num_heads": training_config.transformer.num_heads,
        "dropout": training_config.transformer.dropout,
        "hidden_dim": training_config.transformer.hidden_dim,
        "learning_rate": training_config.training.learning_rate,
        "weight_decay": training_config.training.weight_decay,
        "batch_size": training_config.training.batch_size,
        "max_epochs": training_config.training.max_epochs,
        "early_stopping_patience": training_config.training.early_stopping_patience,
    }
    tuning_path = (
        artifact_root / "tuning" / training_config.data.dataset_variant / "best_params.json"
    )
    if tuning_path.exists():
        tuning_payload = read_json(tuning_path)
        tuned_params = (
            tuning_payload.get("models", {}).get("ft_transformer", {}).get("best_params", {})
        )
        if isinstance(tuned_params, dict):
            params.update(tuned_params)
    return params


def build_inference_bundle(
    config_path: str | Path,
    output_dir: str | Path = DEFAULT_BUNDLE_DIR,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_path = Path(output_dir)
    app_config = AppConfig.from_env()
    training_config = load_training_config(config_path)
    report_path = resolve_report_path(training_config, app_config.artifact_dir)
    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing champion report at {report_path}. Run full training and reporting first."
        )

    report_payload = read_json(report_path)
    champion_model = str(report_payload.get("best_model", ""))
    if champion_model not in SUPPORTED_CHAMPIONS:
        raise ValueError(
            f"Unsupported production champion {champion_model!r}. "
            f"Supported champions: {sorted(SUPPORTED_CHAMPIONS)}"
        )

    source_model_path = resolve_source_model_path(
        champion_model,
        training_config,
        app_config.model_dir,
    )
    if not source_model_path.exists():
        raise FileNotFoundError(
            f"Selected champion artifact is missing at {source_model_path}. "
            f"Re-run full training for {champion_model} before building the inference bundle."
        )

    preprocessor = fit_preprocessor_for_bundle(training_config)
    feature_schema = build_feature_schema(training_config, preprocessor)

    output_path.mkdir(parents=True, exist_ok=True)
    model_filename = (
        "champion_model.pt" if champion_model == "ft_transformer" else "champion_model.joblib"
    )
    target_model_path = output_path / model_filename
    shutil.copy2(source_model_path, target_model_path)
    preprocessor_path = output_path / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    metadata = build_model_metadata(
        report_payload=report_payload,
        training_config_payload=_load_training_config_payload(config_path),
        config_path=config_path,
        report_path=report_path,
        source_model_path=source_model_path,
    )
    if champion_model == "ft_transformer":
        metadata["transformer"] = _resolve_transformer_metadata(
            training_config,
            app_config.artifact_dir,
        )

    paths = {
        "model_metadata": output_path / "model_metadata.json",
        "feature_schema": output_path / "feature_schema.json",
        "example_request": output_path / "example_request.json",
        "prediction_contract": output_path / "prediction_contract.json",
    }
    write_json(paths["model_metadata"], metadata)
    write_json(paths["feature_schema"], feature_schema)
    write_json(paths["example_request"], build_example_request(feature_schema["feature_names"]))
    write_json(paths["prediction_contract"], build_prediction_contract())

    return {
        "bundle_dir": str(output_path),
        "champion_model": champion_model,
        "model_path": str(target_model_path),
        "preprocessor_path": str(preprocessor_path),
        "metadata_path": str(paths["model_metadata"]),
        "feature_schema_path": str(paths["feature_schema"]),
        "example_request_path": str(paths["example_request"]),
        "prediction_contract_path": str(paths["prediction_contract"]),
    }


def _load_transformer_bundle(
    bundle_dir: Path, metadata: dict[str, Any], feature_count: int
) -> TorchUpliftTrainer:
    transformer_config = metadata.get("transformer", {})
    model = FTTransformerUpliftModel(
        num_features=feature_count + 1,
        d_token=int(transformer_config.get("embedding_dim", 32)),
        num_layers=int(transformer_config.get("num_layers", 2)),
        num_heads=int(transformer_config.get("num_heads", 4)),
        dropout=float(transformer_config.get("dropout", 0.1)),
        hidden_dim=int(transformer_config.get("hidden_dim", 128)),
    )
    trainer = TorchUpliftTrainer(
        model=model,
        learning_rate=float(transformer_config.get("learning_rate", 1e-3)),
        weight_decay=float(transformer_config.get("weight_decay", 1e-4)),
        batch_size=int(transformer_config.get("batch_size", 1024)),
        epochs=int(transformer_config.get("max_epochs", 1)),
        patience=int(transformer_config.get("early_stopping_patience", 1)),
    )
    checkpoint = torch.load(bundle_dir / "champion_model.pt", map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint)
    trainer.model.to(trainer.device)
    return trainer


def load_inference_bundle(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> InferenceBundle:
    bundle_path = Path(bundle_dir)
    metadata_path = bundle_path / "model_metadata.json"
    schema_path = bundle_path / "feature_schema.json"
    preprocessor_path = bundle_path / "preprocessor.joblib"
    if not metadata_path.exists() or not schema_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Production bundle is incomplete under {bundle_path}. "
            "Run `uv run python scripts/build_inference_bundle.py --config "
            "configs/training_full.yaml`."
        )

    metadata = read_json(metadata_path)
    feature_schema = read_json(schema_path)
    preprocessor = joblib.load(preprocessor_path)
    champion_model = str(metadata["champion_model"])
    if champion_model == "ft_transformer":
        model = _load_transformer_bundle(
            bundle_path,
            metadata,
            int(feature_schema["feature_count"]),
        )
    else:
        model_path = bundle_path / "champion_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing bundled champion model at {model_path}")
        model = joblib.load(model_path)

    return InferenceBundle(
        bundle_dir=bundle_path,
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        feature_schema=feature_schema,
    )


def new_request_id() -> str:
    return str(uuid.uuid4())
