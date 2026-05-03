from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from causal_uplift.reporting import generate_reporting_artifacts


def test_reporting_artifact_contract_and_tie_breaker(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "artifacts" / "evaluation"
    reports_dir = tmp_path / "artifacts" / "reports"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = evaluation_dir / "full_training_metrics.json"
    predictions_path = evaluation_dir / "test_predictions.csv"

    metrics_payload = {
        "ranking": [
            {
                "model_name": "model_b",
                "qini_auc": 0.21,
                "uplift_auc": 0.18,
                "policy_gain_top10": 0.010,
                "policy_gain_top20": 0.020,
                "policy_gain_top30": 0.021,
                "treatment_response_auc": 0.61,
            },
            {
                "model_name": "model_a",
                "qini_auc": 0.21,
                "uplift_auc": 0.17,
                "policy_gain_top10": 0.012,
                "policy_gain_top20": 0.040,
                "policy_gain_top30": 0.025,
                "treatment_response_auc": 0.59,
            },
        ]
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    pd.DataFrame(
        {
            "y_true": [1, 0, 1, 0],
            "treatment": [1, 0, 1, 0],
            "uplift_score": [0.5, 0.2, 0.4, -0.1],
            "treatment_probability": [0.8, 0.3, 0.7, 0.2],
            "control_probability": [0.3, 0.2, 0.4, 0.2],
            "model_name": ["model_a", "model_a", "model_b", "model_b"],
        }
    ).to_csv(predictions_path, index=False)

    result = generate_reporting_artifacts(
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        output_dir=reports_dir,
        project_name="causal-uplift-transformer-beam",
        python_package="causal_uplift",
        mlflow_tracking_uri="sqlite:///mlflow.db",
        experiment_name="causal-uplift-training",
        plot_artifacts=[],
    )

    report_paths = result["report_paths"]
    assert Path(report_paths["model_ranking_csv"]).exists()
    assert Path(report_paths["model_ranking_json"]).exists()
    assert Path(report_paths["best_model_summary_json"]).exists()
    assert Path(report_paths["experiment_manifest_json"]).exists()

    with Path(report_paths["best_model_summary_json"]).open("r", encoding="utf-8") as handle:
        best_summary = json.load(handle)

    assert best_summary["best_model"] == "model_a"
    assert best_summary["selection_metric"] == "qini_auc"
