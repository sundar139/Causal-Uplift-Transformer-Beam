from __future__ import annotations

from causal_uplift.champion import (
    compare_against_current_champion,
    rank_models_by_policy,
    select_champion,
)


def test_champion_selection_uses_qini_first() -> None:
    champion = select_champion(
        [
            {"model_name": "s_learner_logistic", "qini_auc": 0.2, "policy_gain_top20": 0.9},
            {"model_name": "ft_transformer_causal", "qini_auc": 0.3, "policy_gain_top20": 0.1},
        ]
    )

    assert champion["model_name"] == "ft_transformer_causal"


def test_policy_gain_breaks_qini_ties() -> None:
    ranking = rank_models_by_policy(
        [
            {"model_name": "model_a", "qini_auc": 0.2, "policy_gain_top20": 0.1},
            {"model_name": "model_b", "qini_auc": 0.2, "policy_gain_top20": 0.3},
        ]
    )

    assert ranking.iloc[0]["model_name"] == "model_b"


def test_causal_ft_selected_only_when_it_wins() -> None:
    summary = compare_against_current_champion(
        [
            {"model_name": "s_learner_logistic", "qini_auc": 0.3, "policy_gain_top20": 0.1},
            {"model_name": "ft_transformer_causal", "qini_auc": 0.2, "policy_gain_top20": 0.9},
        ]
    )

    assert summary["champion_model"] == "s_learner_logistic"
    assert summary["transformer_won"] is False


def test_challenger_summary_qini_gap() -> None:
    summary = compare_against_current_champion(
        [
            {"model_name": "s_learner_logistic", "qini_auc": 0.2, "policy_gain_top20": 0.1},
            {"model_name": "ft_transformer_causal", "qini_auc": 0.25, "policy_gain_top20": 0.2},
        ]
    )

    assert summary["champion_model"] == "ft_transformer_causal"
    assert summary["transformer_won"] is True
    assert summary["qini_gap"] == 0.0
