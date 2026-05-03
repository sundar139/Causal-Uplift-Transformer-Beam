from __future__ import annotations

import optuna

from causal_uplift.search_space import suggest_s_learner_params, suggest_transformer_params


def test_transformer_search_space_returns_compatible_attention_shape() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "embedding_dim": 64,
            "num_heads": 8,
            "num_layers": 3,
            "dropout": 0.12,
            "hidden_dim": 128,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 2048,
        }
    )

    params = suggest_transformer_params(trial)

    assert params["embedding_dim"] % params["num_heads"] == 0
    assert params["embedding_dim"] in {32, 64, 96}
    assert params["num_heads"] in {2, 4, 8}


def test_s_learner_search_space_contains_required_keys() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "C": 1.5,
            "penalty": "l2",
            "class_weight": "balanced",
        }
    )

    params = suggest_s_learner_params(trial)

    assert {"C", "penalty", "class_weight", "max_iter"}.issubset(params)
    assert params["penalty"] == "l2"
    assert params["max_iter"] == 500


def test_transformer_search_space_never_returns_invalid_combinations() -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))
    for _ in range(20):
        trial = study.ask()
        params = suggest_transformer_params(trial)
        assert params["embedding_dim"] % params["num_heads"] == 0
        study.tell(trial, 0.0)
