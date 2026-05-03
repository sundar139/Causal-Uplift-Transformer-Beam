from __future__ import annotations

import argparse
import json
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
    dataset_variant_from_config,
    load_criteo_dataset,
    load_criteo_sample,
)
from causal_uplift.evaluate import (
    build_prediction_frame,
    compute_uplift_metrics,
    export_evaluation_artifacts,
)
from causal_uplift.mlflow_utils import (
    initialize_mlflow,
    log_artifacts,
    log_run_payload,
    start_model_run,
    start_named_run,
)
from causal_uplift.plots import generate_curve_plots
from causal_uplift.preprocessing import NumericFeaturePreprocessor
from causal_uplift.reporting import generate_reporting_artifacts
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


def _build_variant_paths(
    app_config: AppConfig,
    training_config: TrainingConfig,
) -> dict[str, Path]:
    paths = resolve_training_output_paths(training_config, Path(app_config.artifact_dir))
    evaluation_dir = paths["evaluation_dir"]
    reports_dir = paths["reports_dir"]
    plots_dir = paths["plots_dir"]
    return {
        "evaluation_dir": evaluation_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
    }


def resolve_training_output_paths(
    training_config: TrainingConfig,
    artifact_root: Path,
) -> dict[str, Path]:
    variant = dataset_variant_from_config(training_config)
    return {
        "evaluation_dir": artifact_root / "evaluation" / variant,
        "reports_dir": artifact_root / "reports" / variant,
        "plots_dir": artifact_root / "plots" / variant,
    }


def _dataset_mlflow_tags(
    training_config: TrainingConfig,
    row_count_train: int,
    row_count_validation: int,
    row_count_test: int,
) -> dict[str, str]:
    return {
        "dataset_variant": dataset_variant_from_config(training_config),
        "data_percent10": str(training_config.data.percent10).lower(),
        "sample_size": str(training_config.data.sample_size),
        "row_count_train": str(row_count_train),
        "row_count_validation": str(row_count_validation),
        "row_count_test": str(row_count_test),
    }


def run_full_training(config_path: str | Path) -> dict[str, object]:
    app_config = AppConfig.from_env()
    training_config: TrainingConfig = load_training_config(config_path)
    initialize_mlflow(app_config)

    dataset = load_criteo_dataset(
        sample_size=training_config.data.sample_size,
        percent10=training_config.data.percent10,
        target_col=training_config.data.target_col,
        treatment_col=training_config.data.treatment_col,
    )
    split = create_train_validation_test_split(
        dataset,
        validation_size=training_config.data.validation_size,
        test_size=training_config.data.test_size,
        random_state=training_config.random_state,
    )
    dataset_tags = _dataset_mlflow_tags(
        training_config,
        row_count_train=len(split.X_train),
        row_count_validation=len(split.X_validation),
        row_count_test=len(split.X_test),
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

    variant_paths = _build_variant_paths(app_config, training_config)
    evaluation_dir = variant_paths["evaluation_dir"]
    results: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name in training_config.models:
        if model_name == "ft_transformer":
            transformer_model = FTTransformerUpliftModel(
                num_features=X_train_np.shape[1] + 1,
                d_token=training_config.transformer.embedding_dim,
                num_layers=training_config.transformer.num_layers,
                num_heads=training_config.transformer.num_heads,
                dropout=training_config.transformer.dropout,
                hidden_dim=training_config.transformer.hidden_dim,
            )
            trainer = TorchUpliftTrainer(
                model=transformer_model,
                learning_rate=training_config.training.learning_rate,
                weight_decay=training_config.training.weight_decay,
                batch_size=training_config.training.batch_size,
                epochs=training_config.training.max_epochs,
                patience=training_config.training.early_stopping_patience,
                random_state=training_config.random_state,
            )

            train_inputs = np.concatenate(
                [X_train_np, t_train.to_numpy(dtype=np.float32).reshape(-1, 1)],
                axis=1,
            )
            validation_inputs = np.concatenate(
                [X_validation_np, t_validation.to_numpy(dtype=np.float32).reshape(-1, 1)],
                axis=1,
            )

            model_path = Path(app_config.model_dir) / training_config.artifacts.best_model_name
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
                "sample_size": training_config.data.sample_size,
                "random_state": training_config.random_state,
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
                "sample_size": training_config.data.sample_size,
                "random_state": training_config.random_state,
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
        prediction_frames.append(prediction_frame)

        with start_model_run(
            model_name=model_name,
            tags={
                "pipeline": "full",
                "model_kind": model_kind,
                **dataset_tags,
            },
        ):
            log_run_payload(
                params=run_params,
                metrics=metrics,
                artifact_paths=[model_path, metrics_path, predictions_path],
            )

        results.append({"model_name": model_name, **metrics})

    ranking = _rank_models(results)
    best_model_name = str(ranking.iloc[0]["model_name"])

    final_metrics_path = evaluation_dir / training_config.artifacts.metrics_name
    final_predictions_path = evaluation_dir / training_config.artifacts.predictions_name
    consolidated_metrics = {
        "best_model": best_model_name,
        "dataset_variant": dataset_variant_from_config(training_config),
        "data_percent10": training_config.data.percent10,
        "selection_metric": "qini_auc",
        "tie_breaker": "policy_gain_top20",
        "ranking": ranking.to_dict(orient="records"),
    }
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with final_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(consolidated_metrics, handle, indent=2)
    pd.concat(prediction_frames, ignore_index=True).to_csv(final_predictions_path, index=False)

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


def run_reporting(config_path: str | Path) -> dict[str, object]:
    app_config = AppConfig.from_env()
    training_config = load_training_config(config_path)
    initialize_mlflow(app_config)

    variant_paths = _build_variant_paths(app_config, training_config)
    evaluation_dir = variant_paths["evaluation_dir"]
    reports_dir = variant_paths["reports_dir"]
    plots_dir = variant_paths["plots_dir"]

    metrics_path = evaluation_dir / training_config.artifacts.metrics_name
    predictions_path = evaluation_dir / training_config.artifacts.predictions_name
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing required metrics artifact at {metrics_path}. Run full training first."
        )
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing required prediction artifact at {predictions_path}. Run full training first."
        )

    plot_outputs = generate_curve_plots(predictions_path=predictions_path, output_dir=plots_dir)
    reporting_result = generate_reporting_artifacts(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        output_dir=reports_dir,
        project_name="causal-uplift-transformer-beam",
        python_package="causal_uplift",
        mlflow_tracking_uri=app_config.mlflow_tracking_uri,
        experiment_name=app_config.mlflow_experiment_name,
        plot_artifacts=list(plot_outputs.values()),
    )

    report_paths = reporting_result["report_paths"]
    artifact_paths = [
        report_paths["model_ranking_csv"],
        report_paths["model_ranking_json"],
        report_paths["best_model_summary_json"],
        report_paths["experiment_manifest_json"],
        plot_outputs["qini_curve"],
        plot_outputs["uplift_curve"],
        plot_outputs["policy_gain_curve"],
    ]

    row_counts = {"train": 0, "validation": 0, "test": 0}
    manifest_path = (
        Path(app_config.artifact_dir)
        / "data"
        / dataset_variant_from_config(training_config)
        / "data_manifest.json"
    )
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest_payload = json.load(handle)
        row_counts = {
            "train": int(manifest_payload.get("row_counts", {}).get("train", 0)),
            "validation": int(manifest_payload.get("row_counts", {}).get("validation", 0)),
            "test": int(manifest_payload.get("row_counts", {}).get("test", 0)),
        }

    with start_named_run(
        run_name="reporting-artifact-consolidation",
        tags={
            "project": "causal-uplift-transformer-beam",
            "run_type": "reporting",
            "artifact_scope": "readme_and_dashboard",
            "dataset_variant": dataset_variant_from_config(training_config),
            "data_percent10": str(training_config.data.percent10).lower(),
            "sample_size": str(training_config.data.sample_size),
            "row_count_train": str(row_counts["train"]),
            "row_count_validation": str(row_counts["validation"]),
            "row_count_test": str(row_counts["test"]),
        },
    ):
        log_run_payload(
            params={
                "config_path": str(config_path),
                "source_metrics_path": str(metrics_path),
                "source_predictions_path": str(predictions_path),
            },
            metrics={},
            artifact_paths=[],
        )
        log_artifacts(artifact_paths, artifact_path="reporting")

    summary = {
        "source_metrics": str(metrics_path),
        "source_predictions": str(predictions_path),
        "report_artifacts": [str(path) for path in artifact_paths[:4]],
        "plot_artifacts": [str(path) for path in artifact_paths[4:]],
    }
    print("\nReporting artifact summary:")
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

    report_parser = subparsers.add_parser(
        "report",
        help="Generate consolidated reporting artifacts and plots",
    )
    report_parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration",
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

    if args.command == "report":
        run_reporting(config_path=args.config)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
