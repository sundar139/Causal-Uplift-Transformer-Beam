from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _curve_points_for_model(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    y = ordered["y_true"].to_numpy(dtype=float)
    t = ordered["treatment"].to_numpy(dtype=int)
    uplift_score = ordered["uplift_score"].to_numpy(dtype=float)

    n = len(ordered)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty

    treated = (t == 1).astype(float)
    control = (t == 0).astype(float)

    cum_treated = np.cumsum(treated)
    cum_control = np.cumsum(control)
    cum_treated_resp = np.cumsum(y * treated)
    cum_control_resp = np.cumsum(y * control)

    treated_rate = np.divide(
        cum_treated_resp,
        cum_treated,
        out=np.zeros_like(cum_treated_resp),
        where=cum_treated > 0,
    )
    control_rate = np.divide(
        cum_control_resp,
        cum_control,
        out=np.zeros_like(cum_control_resp),
        where=cum_control > 0,
    )

    qini_curve = cum_treated_resp - (
        cum_control_resp
        * np.divide(
            cum_treated,
            cum_control,
            out=np.zeros_like(cum_treated),
            where=cum_control > 0,
        )
    )
    uplift_curve = treated_rate - control_rate

    fractions = (np.arange(1, n + 1, dtype=float)) / float(n)
    cumulative_predicted_uplift = np.cumsum(uplift_score) / np.arange(1, n + 1, dtype=float)

    return fractions, qini_curve, uplift_curve, cumulative_predicted_uplift


def _save_plot(
    predictions: pd.DataFrame,
    y_selector: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))

    has_any = False
    for model_name, group in predictions.groupby("model_name"):
        x, qini_curve, uplift_curve, policy_curve = _curve_points_for_model(group)
        if len(x) == 0:
            continue
        has_any = True
        if y_selector == "qini":
            y_values = qini_curve
        elif y_selector == "uplift":
            y_values = uplift_curve
        else:
            y_values = policy_curve
        plt.plot(x, y_values, label=str(model_name))

    if has_any:
        plt.legend()

    plt.title(title)
    plt.xlabel("Population fraction targeted")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=150)
    plt.close()


def generate_curve_plots(
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    predictions = pd.read_csv(Path(predictions_path))
    required_columns = {"model_name", "y_true", "treatment", "uplift_score"}
    missing = required_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions file is missing required columns: {sorted(missing)}")

    target_dir = Path(output_dir)
    qini_path = target_dir / "qini_curve.png"
    uplift_path = target_dir / "uplift_curve.png"
    policy_path = target_dir / "policy_gain_curve.png"

    _save_plot(
        predictions=predictions,
        y_selector="qini",
        title="Qini Curve by Model",
        y_label="Incremental response",
        output_path=qini_path,
    )
    _save_plot(
        predictions=predictions,
        y_selector="uplift",
        title="Observed Uplift Curve by Model",
        y_label="Observed uplift",
        output_path=uplift_path,
    )
    _save_plot(
        predictions=predictions,
        y_selector="policy",
        title="Policy Gain Curve by Model",
        y_label="Average predicted uplift",
        output_path=policy_path,
    )

    return {
        "qini_curve": qini_path,
        "uplift_curve": uplift_path,
        "policy_gain_curve": policy_path,
    }
