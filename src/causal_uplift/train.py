from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from causal_uplift.baselines import (
    SLearnerBaseline,
    TLearnerBaseline,
    TwoModelUpliftBaseline,
    UpliftPrediction,
)
from causal_uplift.config import AppConfig, TrainingConfig, load_training_config
from causal_uplift.data import (
    create_train_validation_test_split,
    load_criteo_dataset,
    load_criteo_sample,
)
from causal_uplift.evaluate import (
    build_prediction_frame,
    compute_uplift_metrics,
    export_evaluation_artifacts,
)
from causal_uplift.mlflow_utils import initialize_mlflow, log_run_payload, start_model_run
from causal_uplift.preprocessing import NumericFeaturePreprocessor
from causal_uplift.transformer import FTTransformerUpliftModel, TorchUpliftTrainer


def run_smoke_training(sample_size: int = 10_000) -> dict[str, str | dict[str, float]]:
    config = AppConfig.from_env()
    initialize_mlflow(config)

    dataset = load_criteo_sample(sample_size=sample_size, random_state=config.random_state)

    model = TwoModelUpliftBaseline(random_state=config.random_state)
    model.fit(dataset.X_train, dataset.y_train, dataset.treatment_train)

    predictions = model.predict_uplift(dataset.X_test)
    metrics = compute_uplift_metrics(
        y_true=dataset.y_test.to_numpy(),
        treatment=dataset.treatment_test.to_numpy(),
        uplift=predictions.uplift,
        treatment_proba=predictions.treatment_proba,
    )

    model_path = Path(config.model_dir) / "smoke_uplift_baseline.joblib"
    metrics_path = Path(config.artifact_dir) / "smoke_metrics.json"
    predictions_path = Path(config.artifact_dir) / "smoke_predictions.csv"
    prediction_frame = build_prediction_frame(
        y_true=dataset.y_test.to_numpy(),
        treatment=dataset.treatment_test.to_numpy(),
        uplift=predictions.uplift,
        treatment_proba=predictions.treatment_proba,
        control_proba=predictions.control_proba,
        model_name="two_model_logistic_smoke",
    )

    with start_model_run(
        model_name="two_model_logistic_smoke",
        tags={"pipeline": "smoke", "component": "baseline"},
    ):
        params = {
            "sample_size": sample_size,
            "random_state": config.random_state,
            "model": "two_model_logistic_regression",
        }

        joblib.dump(model, model_path)
        export_evaluation_artifacts(
            metrics=metrics,
            predictions=prediction_frame,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
        )

        log_run_payload(
            params=params,
            metrics=metrics,
            artifact_paths=[model_path, metrics_path, predictions_path],
        )

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def _train_baseline_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    t_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[object, UpliftPrediction]:
    model_registry: dict[str, object] = {
        "two_model_logistic": TwoModelUpliftBaseline(),
        "s_learner_logistic": SLearnerBaseline(),
        "t_learner_logistic": TLearnerBaseline(),
    }
    model = model_registry[model_name]
    model.fit(X_train, y_train, t_train)
    predictions: UpliftPrediction = model.predict_uplift(X_test)
    return model, predictions


def _rank_models(results: list[dict[str, float | str]]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    ranking = frame.sort_values(
        by=["qini_auc", "policy_gain_top20"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return ranking


def run_full_training(config_path: str | Path) -> dict[str, object]:
    app_config = AppConfig.from_env()
    training_config: TrainingConfig = load_training_config(config_path)
    initialize_mlflow(app_config)

    dataset = load_criteo_dataset(sample_size=training_config.sample_size)
    split = create_train_validation_test_split(
        dataset,
        validation_size=training_config.split.validation_size,
        test_size=training_config.split.test_size,
        random_state=app_config.random_state,
    )

    preprocessor = NumericFeaturePreprocessor()
    X_train_np = preprocessor.fit_transform(split.X_train)
    X_validation_np = preprocessor.transform(split.X_validation)
    X_test_np = preprocessor.transform(split.X_test)

    feature_columns = preprocessor.feature_columns
    X_train_df = pd.DataFrame(X_train_np, columns=feature_columns)
    X_test_df = pd.DataFrame(X_test_np, columns=feature_columns)

    y_train = split.y_train.reset_index(drop=True)
    y_validation = split.y_validation.reset_index(drop=True)
    y_test = split.y_test.reset_index(drop=True)
    t_train = split.treatment_train.reset_index(drop=True)
    t_validation = split.treatment_validation.reset_index(drop=True)
    t_test = split.treatment_test.reset_index(drop=True)

    evaluation_dir = Path(app_config.artifact_dir) / "evaluation"
    results: list[dict[str, float | str]] = []
    model_artifacts: dict[str, tuple[Path, Path]] = {}

    for model_name in training_config.models:
        if model_name == "ft_transformer":
            transformer_model = FTTransformerUpliftModel(
                num_features=X_train_np.shape[1] + 1,
                d_token=training_config.transformer.d_token,
                num_layers=training_config.transformer.num_layers,
                num_heads=training_config.transformer.num_heads,
                dropout=training_config.transformer.dropout,
            )
            trainer = TorchUpliftTrainer(
                model=transformer_model,
                learning_rate=training_config.transformer.learning_rate,
                weight_decay=training_config.transformer.weight_decay,
                batch_size=training_config.transformer.batch_size,
                epochs=training_config.transformer.epochs,
                patience=training_config.transformer.patience,
                random_state=app_config.random_state,
            )

            train_inputs = np.concatenate(
                [X_train_np, t_train.to_numpy(dtype=np.float32).reshape(-1, 1)],
                axis=1,
            )
            validation_inputs = np.concatenate(
                [X_validation_np, t_validation.to_numpy(dtype=np.float32).reshape(-1, 1)],
                axis=1,
            )

            model_path = Path(app_config.model_dir) / "best_transformer_uplift.pt"
            trainer.fit(
                X_train=train_inputs,
                y_train=y_train.to_numpy(dtype=np.float32),
                X_validation=validation_inputs,
                y_validation=y_validation.to_numpy(dtype=np.float32),
                checkpoint_path=model_path,
            )
            prediction = trainer.predict_uplift(X_test_np)
            model_kind = "transformer"
            run_params: dict[str, int | float | str] = {
                "sample_size": training_config.sample_size,
                "random_state": app_config.random_state,
                "d_token": training_config.transformer.d_token,
                "num_layers": training_config.transformer.num_layers,
                "num_heads": training_config.transformer.num_heads,
                "dropout": training_config.transformer.dropout,
                "learning_rate": training_config.transformer.learning_rate,
                "weight_decay": training_config.transformer.weight_decay,
                "batch_size": training_config.transformer.batch_size,
                "epochs": training_config.transformer.epochs,
                "patience": training_config.transformer.patience,
            }
        else:
            trained_model, prediction = _train_baseline_model(
                model_name=model_name,
                X_train=X_train_df,
                y_train=y_train,
                t_train=t_train,
                X_test=X_test_df,
            )
            model_path = Path(app_config.model_dir) / f"{model_name}.joblib"
            joblib.dump(trained_model, model_path)
            model_kind = "baseline"
            run_params = {
                "sample_size": training_config.sample_size,
                "random_state": app_config.random_state,
                "model": model_name,
            }

        metrics = compute_uplift_metrics(
            y_true=y_test.to_numpy(),
            treatment=t_test.to_numpy(),
            uplift=prediction.uplift,
            treatment_proba=prediction.treatment_proba,
        )

        metrics_path = evaluation_dir / f"{model_name}_metrics.json"
        predictions_path = evaluation_dir / f"{model_name}_predictions.csv"
        prediction_frame = build_prediction_frame(
            y_true=y_test.to_numpy(),
            treatment=t_test.to_numpy(),
            uplift=prediction.uplift,
            treatment_proba=prediction.treatment_proba,
            control_proba=prediction.control_proba,
            model_name=model_name,
        )
        export_evaluation_artifacts(
            metrics=metrics,
            predictions=prediction_frame,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
        )

        with start_model_run(
            model_name=model_name,
            tags={"pipeline": "full", "model_kind": model_kind},
        ):
            log_run_payload(
                params=run_params,
                metrics=metrics,
                artifact_paths=[model_path, metrics_path, predictions_path],
            )

        results.append({"model_name": model_name, **metrics})
        model_artifacts[model_name] = (metrics_path, predictions_path)

    ranking = _rank_models(results)
    best_model_name = str(ranking.iloc[0]["model_name"])
    best_metrics_path, best_predictions_path = model_artifacts[best_model_name]

    final_metrics_path = evaluation_dir / "full_training_metrics.json"
    final_predictions_path = evaluation_dir / "test_predictions.csv"
    shutil.copyfile(best_metrics_path, final_metrics_path)
    shutil.copyfile(best_predictions_path, final_predictions_path)

    table_columns = [
        "model_name",
        "qini_auc",
        "policy_gain_top20",
        "uplift_auc",
        "treatment_response_auc",
    ]
    print("\nRanked model metrics:")
    print(ranking[table_columns].to_string(index=False))

    summary = {
        "best_model": best_model_name,
        "ranking": ranking.to_dict(orient="records"),
        "full_metrics_path": str(final_metrics_path),
        "predictions_path": str(final_predictions_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal uplift training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Run smoke baseline training")
    smoke_parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="Number of rows for smoke run",
    )

    full_parser = subparsers.add_parser("full", help="Run full uplift training suite")
    full_parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to full training configuration",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke_training(sample_size=args.sample_size)
        print(json.dumps(result, indent=2))
        return

    if args.command == "full":
        run_full_training(config_path=args.config)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
