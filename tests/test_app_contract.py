from __future__ import annotations

import importlib

import pandas as pd
from app.charts import model_ranking_chart, policy_gain_chart, prediction_probability_chart
from app.sample_inputs import build_example_dataframe, get_default_feature_values


def test_default_feature_values_include_f0_to_f11() -> None:
    values = get_default_feature_values()
    for idx in range(12):
        assert f"f{idx}" in values


def test_example_dataframe_has_required_columns() -> None:
    frame = build_example_dataframe()
    expected = [f"f{i}" for i in range(12)]

    assert list(frame.columns) == expected
    assert len(frame) == 1


def test_chart_functions_degrade_gracefully_when_missing_data() -> None:
    assert model_ranking_chart(None) is None
    assert policy_gain_chart(None) is None
    assert prediction_probability_chart(None) is None


def test_chart_functions_return_figures_with_valid_data() -> None:
    ranking = pd.DataFrame(
        {
            "model_name": ["s_learner_logistic", "ft_transformer"],
            "qini_auc": [0.19, 0.17],
            "policy_gain_top10": [0.01, 0.008],
            "policy_gain_top20": [0.006, 0.004],
            "policy_gain_top30": [0.004, 0.003],
        }
    )
    prediction = {"treatment_probability": 0.7, "control_probability": 0.5}

    assert model_ranking_chart(ranking) is not None
    assert policy_gain_chart(ranking) is not None
    assert prediction_probability_chart(prediction) is not None


def test_app_modules_import_successfully() -> None:
    modules = [
        "app.api_client",
        "app.sample_inputs",
        "app.charts",
        "app.components",
        "app.streamlit_app",
    ]
    for module_name in modules:
        assert importlib.import_module(module_name)
