from __future__ import annotations

import pandas as pd


def get_default_feature_values() -> dict[str, float]:
    return {
        "f0": 0.1,
        "f1": 0.2,
        "f2": 0.3,
        "f3": 0.4,
        "f4": 0.5,
        "f5": 0.6,
        "f6": 0.7,
        "f7": 0.8,
        "f8": 0.9,
        "f9": 1.0,
        "f10": 1.1,
        "f11": 1.2,
    }


def get_feature_descriptions() -> dict[str, str]:
    return {
        "f0": "Numeric feature from Criteo uplift dataset.",
        "f1": "Numeric feature from Criteo uplift dataset.",
        "f2": "Numeric feature from Criteo uplift dataset.",
        "f3": "Numeric feature from Criteo uplift dataset.",
        "f4": "Numeric feature from Criteo uplift dataset.",
        "f5": "Numeric feature from Criteo uplift dataset.",
        "f6": "Numeric feature from Criteo uplift dataset.",
        "f7": "Numeric feature from Criteo uplift dataset.",
        "f8": "Numeric feature from Criteo uplift dataset.",
        "f9": "Numeric feature from Criteo uplift dataset.",
        "f10": "Numeric feature from Criteo uplift dataset.",
        "f11": "Numeric feature from Criteo uplift dataset.",
    }


def build_example_dataframe() -> pd.DataFrame:
    return pd.DataFrame([get_default_feature_values()])
