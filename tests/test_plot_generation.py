from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_uplift.plots import generate_curve_plots


def test_plot_generation_outputs_png_files(tmp_path: Path) -> None:
    predictions_path = tmp_path / "test_predictions.csv"
    plots_dir = tmp_path / "plots"

    predictions = pd.DataFrame(
        {
            "y_true": [1, 0, 1, 0, 1, 0, 1, 0],
            "treatment": [1, 0, 1, 0, 1, 0, 1, 0],
            "uplift_score": [0.8, 0.1, 0.7, -0.1, 0.6, 0.0, 0.5, -0.2],
            "treatment_probability": [0.9, 0.4, 0.85, 0.3, 0.8, 0.5, 0.75, 0.2],
            "control_probability": [0.3, 0.3, 0.35, 0.4, 0.4, 0.45, 0.45, 0.35],
            "model_name": [
                "model_a",
                "model_a",
                "model_a",
                "model_a",
                "model_b",
                "model_b",
                "model_b",
                "model_b",
            ],
        }
    )
    predictions.to_csv(predictions_path, index=False)

    output = generate_curve_plots(predictions_path=predictions_path, output_dir=plots_dir)

    for path in output.values():
        assert path.exists()
        assert path.stat().st_size > 0
