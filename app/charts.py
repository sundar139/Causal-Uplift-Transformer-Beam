from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

REPORTS_DIR = Path("artifacts") / "reports" / "full"
MODEL_RANKING_PATH = REPORTS_DIR / "model_ranking.csv"
CHAMPION_CHALLENGER_PATH = REPORTS_DIR / "champion_challenger_summary.json"


def load_model_ranking() -> pd.DataFrame | None:
    if not MODEL_RANKING_PATH.exists():
        return None
    try:
        return pd.read_csv(MODEL_RANKING_PATH)
    except Exception:
        return None


def load_champion_challenger_summary() -> dict | None:
    if not CHAMPION_CHALLENGER_PATH.exists():
        return None
    try:
        return json.loads(CHAMPION_CHALLENGER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def model_ranking_chart(ranking_df: pd.DataFrame | None):
    if ranking_df is None or ranking_df.empty:
        return None
    if "model_name" not in ranking_df.columns or "qini_auc" not in ranking_df.columns:
        return None

    chart_df = ranking_df.sort_values("qini_auc", ascending=False)
    return px.bar(
        chart_df,
        x="model_name",
        y="qini_auc",
        title="Model Ranking by Qini AUC",
        labels={"model_name": "Model", "qini_auc": "Qini AUC"},
        color="model_name",
    )


def policy_gain_chart(ranking_df: pd.DataFrame | None):
    if ranking_df is None or ranking_df.empty:
        return None
    if "model_name" not in ranking_df.columns:
        return None

    gain_columns = [
        column
        for column in ["policy_gain_top10", "policy_gain_top20", "policy_gain_top30"]
        if column in ranking_df.columns
    ]
    if not gain_columns:
        return None

    melted = ranking_df[["model_name", *gain_columns]].melt(
        id_vars="model_name",
        value_vars=gain_columns,
        var_name="Policy Window",
        value_name="Policy Gain",
    )
    return px.line(
        melted,
        x="Policy Window",
        y="Policy Gain",
        color="model_name",
        markers=True,
        title="Policy Gain by Model",
    )


def prediction_probability_chart(prediction: dict | None):
    if not prediction:
        return None
    required = {"treatment_probability", "control_probability"}
    if not required.issubset(prediction.keys()):
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Control", "Treatment"],
                y=[prediction["control_probability"], prediction["treatment_probability"]],
                marker_color=["#6B7280", "#F2C94C"],
            )
        ]
    )
    fig.update_layout(
        title="Predicted Conversion Probability",
        yaxis_title="Probability",
        xaxis_title="Scenario",
        yaxis=dict(range=[0, 1]),
    )
    return fig
