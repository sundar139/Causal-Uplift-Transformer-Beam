from __future__ import annotations

from typing import Any

import optuna

TRANSFORMER_EMBEDDING_DIMS = [32, 64, 96]
TRANSFORMER_NUM_HEADS = [2, 4, 8]


def suggest_transformer_params(trial: optuna.trial.Trial) -> dict[str, Any]:
    embedding_dim = trial.suggest_categorical("embedding_dim", TRANSFORMER_EMBEDDING_DIMS)
    suggested_heads = trial.suggest_categorical("num_heads", TRANSFORMER_NUM_HEADS)
    if embedding_dim % suggested_heads == 0:
        num_heads = suggested_heads
    else:
        valid_heads = [heads for heads in TRANSFORMER_NUM_HEADS if embedding_dim % heads == 0]
        num_heads = max(valid_heads)
        trial.set_user_attr("adjusted_num_heads", num_heads)

    return {
        "embedding_dim": int(embedding_dim),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "num_heads": int(num_heads),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
    }


def suggest_s_learner_params(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "max_iter": 500,
    }
