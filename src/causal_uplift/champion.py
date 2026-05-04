from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd


def rank_models_by_policy(records: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(records).copy()
    if frame.empty:
        raise ValueError("Cannot rank an empty model list.")
    required = {"model_name", "qini_auc", "policy_gain_top20"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required ranking columns: {sorted(missing)}")
    return frame.sort_values(
        by=["qini_auc", "policy_gain_top20"],
        ascending=[False, False],
    ).reset_index(drop=True)


def select_champion(records: list[dict[str, Any]] | pd.DataFrame) -> dict[str, Any]:
    ranking = rank_models_by_policy(records)
    return dict(ranking.iloc[0].to_dict())


def compare_against_current_champion(
    records: list[dict[str, Any]] | pd.DataFrame,
) -> dict[str, Any]:
    ranking = rank_models_by_policy(records)
    champion = dict(ranking.iloc[0].to_dict())
    transformer_rows = ranking[
        ranking["model_name"].astype(str).str.contains("ft_transformer", regex=False)
    ]
    if transformer_rows.empty:
        best_transformer = {
            "model_name": "",
            "qini_auc": 0.0,
            "policy_gain_top20": 0.0,
        }
    else:
        best_transformer = dict(transformer_rows.iloc[0].to_dict())

    champion_qini = float(champion.get("qini_auc", 0.0))
    transformer_qini = float(best_transformer.get("qini_auc", 0.0))
    transformer_won = str(champion["model_name"]) == str(best_transformer.get("model_name", ""))
    recommendation = (
        "Promote the causal FT transformer after validation."
        if transformer_won
        else "Keep the current non-transformer champion; treat causal FT as a challenger."
    )
    return {
        "champion_model": str(champion["model_name"]),
        "best_transformer_model": str(best_transformer.get("model_name", "")),
        "champion_qini_auc": champion_qini,
        "best_transformer_qini_auc": transformer_qini,
        "qini_gap": champion_qini - transformer_qini,
        "transformer_won": transformer_won,
        "recommendation": recommendation,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
