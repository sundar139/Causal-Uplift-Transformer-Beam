from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class ReportingArtifacts:
    model_ranking_csv: Path
    model_ranking_json: Path
    best_model_summary_json: Path
    experiment_manifest_json: Path


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def load_metrics_payload(metrics_path: str | Path) -> list[dict[str, float | str]]:
    path = Path(metrics_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if isinstance(payload.get("ranking"), list):
            return payload["ranking"]
        if isinstance(payload.get("models"), list):
            return payload["models"]
        if "qini_auc" in payload:
            single = dict(payload)
            single.setdefault("model_name", str(payload.get("best_model", "best_model")))
            return [single]

    raise ValueError(
        "Unsupported full_training_metrics.json format. "
        "Expected list, {'ranking': [...]}, {'models': [...]}, or single metric dictionary."
    )


def load_predictions_frame(predictions_path: str | Path) -> pd.DataFrame:
    path = Path(predictions_path)
    frame = pd.read_csv(path)
    if "model_name" not in frame.columns:
        frame["model_name"] = "best_model"
    return frame


def build_ranked_model_comparison(metrics_rows: list[dict[str, float | str]]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics_rows)
    required = {"model_name", "qini_auc", "policy_gain_top20"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required metric columns: {sorted(missing)}")

    return frame.sort_values(
        by=["qini_auc", "policy_gain_top20"],
        ascending=[False, False],
    ).reset_index(drop=True)


def select_best_model(ranking: pd.DataFrame) -> pd.Series:
    if ranking.empty:
        raise ValueError("Model ranking cannot be empty.")
    return ranking.iloc[0]


def _best_model_summary(
    best_row: pd.Series,
    predictions: pd.DataFrame,
    source_metrics_path: Path,
    source_predictions_path: Path,
) -> dict[str, float | str]:
    model_name = str(best_row["model_name"])
    model_predictions = predictions[predictions["model_name"] == model_name]

    if model_predictions.empty:
        mean_predicted_uplift = 0.0
        positive_uplift_rate = 0.0
    else:
        uplift = model_predictions["uplift_score"].astype(float)
        mean_predicted_uplift = float(uplift.mean())
        positive_uplift_rate = float((uplift > 0.0).mean())

    return {
        "best_model": model_name,
        "selection_metric": "qini_auc",
        "qini_auc": float(best_row.get("qini_auc", 0.0)),
        "uplift_auc": float(best_row.get("uplift_auc", 0.0)),
        "policy_gain_top10": float(best_row.get("policy_gain_top10", 0.0)),
        "policy_gain_top20": float(best_row.get("policy_gain_top20", 0.0)),
        "policy_gain_top30": float(best_row.get("policy_gain_top30", 0.0)),
        "treatment_response_auc": float(best_row.get("treatment_response_auc", 0.0)),
        "mean_predicted_uplift": mean_predicted_uplift,
        "positive_uplift_rate": positive_uplift_rate,
        "generated_at_utc": _utc_timestamp(),
        "source_metrics_path": str(source_metrics_path),
        "source_predictions_path": str(source_predictions_path),
    }


def _experiment_manifest(
    ranking: pd.DataFrame,
    report_artifacts: ReportingArtifacts,
    plot_artifacts: list[Path],
    project_name: str,
    python_package: str,
    mlflow_tracking_uri: str,
    experiment_name: str,
) -> dict[str, object]:
    return {
        "project_name": project_name,
        "python_package": python_package,
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "experiment_name": experiment_name,
        "generated_at_utc": _utc_timestamp(),
        "available_models": [str(name) for name in ranking["model_name"].tolist()],
        "report_artifacts": [
            str(report_artifacts.model_ranking_csv),
            str(report_artifacts.model_ranking_json),
            str(report_artifacts.best_model_summary_json),
            str(report_artifacts.experiment_manifest_json),
        ],
        "plot_artifacts": [str(path) for path in plot_artifacts],
        "notes": "Best model selected by qini_auc with policy_gain_top20 tie-breaker.",
    }


def generate_reporting_artifacts(
    metrics_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    project_name: str,
    python_package: str,
    mlflow_tracking_uri: str,
    experiment_name: str,
    plot_artifacts: list[Path] | None = None,
) -> dict[str, object]:
    source_metrics_path = Path(metrics_path)
    source_predictions_path = Path(predictions_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    ranking = build_ranked_model_comparison(load_metrics_payload(source_metrics_path))
    predictions = load_predictions_frame(source_predictions_path)
    best = select_best_model(ranking)

    artifacts = ReportingArtifacts(
        model_ranking_csv=target_dir / "model_ranking.csv",
        model_ranking_json=target_dir / "model_ranking.json",
        best_model_summary_json=target_dir / "best_model_summary.json",
        experiment_manifest_json=target_dir / "experiment_manifest.json",
    )

    ranking.to_csv(artifacts.model_ranking_csv, index=False)
    ranking.to_json(artifacts.model_ranking_json, orient="records", indent=2)

    summary = _best_model_summary(best, predictions, source_metrics_path, source_predictions_path)
    with artifacts.best_model_summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    manifest = _experiment_manifest(
        ranking=ranking,
        report_artifacts=artifacts,
        plot_artifacts=plot_artifacts or [],
        project_name=project_name,
        python_package=python_package,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
    )
    with artifacts.experiment_manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "ranking": ranking,
        "best_model_summary": summary,
        "experiment_manifest": manifest,
        "report_paths": {
            "model_ranking_csv": artifacts.model_ranking_csv,
            "model_ranking_json": artifacts.model_ranking_json,
            "best_model_summary_json": artifacts.best_model_summary_json,
            "experiment_manifest_json": artifacts.experiment_manifest_json,
        },
    }
