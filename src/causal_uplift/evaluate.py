from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklift.metrics import qini_auc_score, uplift_auc_score

MetricFn = Callable[..., float]


def _safe_metric(metric_fn: MetricFn, *args: object) -> float:
    try:
        value = float(metric_fn(*args))
    except Exception:
        return 0.0
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def _policy_gain(
    y_true: np.ndarray, treatment: np.ndarray, uplift: np.ndarray, top_fraction: float
) -> float:
    if len(y_true) == 0:
        return 0.0

    count = max(1, int(len(y_true) * top_fraction))
    top_idx = np.argsort(uplift)[::-1][:count]

    selected_y = y_true[top_idx]
    selected_t = treatment[top_idx]

    treated = selected_y[selected_t == 1]
    control = selected_y[selected_t == 0]

    if len(treated) == 0 or len(control) == 0:
        return 0.0

    return float(np.mean(treated) - np.mean(control))


def compute_uplift_metrics(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift: np.ndarray,
    treatment_proba: np.ndarray,
) -> dict[str, float]:
    y_arr = np.asarray(y_true, dtype=int)
    t_arr = np.asarray(treatment, dtype=int)
    uplift_arr = np.asarray(uplift, dtype=float)
    treatment_proba_arr = np.asarray(treatment_proba, dtype=float)

    treatment_mask = t_arr == 1
    treatment_response_auc = _safe_metric(
        roc_auc_score,
        y_arr[treatment_mask],
        treatment_proba_arr[treatment_mask],
    )

    qini_auc = _safe_metric(qini_auc_score, y_arr, uplift_arr, t_arr)
    uplift_auc = _safe_metric(uplift_auc_score, y_arr, uplift_arr, t_arr)

    policy_gain_top10 = _policy_gain(y_arr, t_arr, uplift_arr, top_fraction=0.10)
    policy_gain_top20 = _policy_gain(y_arr, t_arr, uplift_arr, top_fraction=0.20)
    policy_gain_top30 = _policy_gain(y_arr, t_arr, uplift_arr, top_fraction=0.30)
    mean_predicted_uplift = float(np.mean(uplift_arr)) if len(uplift_arr) > 0 else 0.0
    positive_uplift_rate = float(np.mean(uplift_arr > 0.0)) if len(uplift_arr) > 0 else 0.0

    return {
        "treatment_response_auc": treatment_response_auc,
        "qini_auc": qini_auc,
        "uplift_auc": uplift_auc,
        "policy_gain_top10": policy_gain_top10,
        "policy_gain_top20": policy_gain_top20,
        "policy_gain_top30": policy_gain_top30,
        "mean_predicted_uplift": mean_predicted_uplift,
        "positive_uplift_rate": positive_uplift_rate,
    }


def build_prediction_frame(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift: np.ndarray,
    treatment_proba: np.ndarray,
    control_proba: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=int),
            "treatment": np.asarray(treatment, dtype=int),
            "uplift_score": np.asarray(uplift, dtype=float),
            "treatment_probability": np.asarray(treatment_proba, dtype=float),
            "control_probability": np.asarray(control_proba, dtype=float),
            "model_name": model_name,
        }
    )


def export_evaluation_artifacts(
    metrics: dict[str, float],
    predictions: pd.DataFrame,
    metrics_path: Path,
    predictions_path: Path,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    predictions.to_csv(predictions_path, index=False)
