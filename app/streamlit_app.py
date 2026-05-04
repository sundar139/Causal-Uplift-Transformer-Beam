from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from app.api_client import CausalUpliftApiClient, get_api_base_url
from app.charts import (
    load_champion_challenger_summary,
    load_model_ranking,
    model_ranking_chart,
    policy_gain_chart,
    prediction_probability_chart,
)
from app.components import (
    render_footer,
    render_header,
    render_metric_cards,
    render_model_summary,
    render_prediction_result,
)
from app.sample_inputs import (
    build_example_dataframe,
    get_default_feature_values,
    get_feature_descriptions,
)

REQUIRED_FEATURES = [f"f{i}" for i in range(12)]

st.set_page_config(page_title="Causal Uplift Dashboard", layout="wide")


def _numeric_records(df: pd.DataFrame) -> list[dict[str, float]]:
    converted = df.copy()
    for feature in REQUIRED_FEATURES:
        converted[feature] = pd.to_numeric(converted[feature], errors="raise")
    return converted[REQUIRED_FEATURES].astype(float).to_dict(orient="records")


def _overview_section(client: CausalUpliftApiClient) -> None:
    render_header()
    try:
        info = client.model_info()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    render_model_summary(info)
    render_metric_cards(info)


def _model_performance_section() -> None:
    st.header("Model Performance")
    st.write(
        "Multiple model families were trained and evaluated; production keeps "
        "the model that wins the full-dataset ranking policy. In the current "
        "run, s_learner_logistic leads the deployed stack."
    )

    ranking_df = load_model_ranking()
    if ranking_df is None:
        st.warning("Ranking artifact not found at artifacts/reports/full/model_ranking.csv")
    else:
        st.subheader("Ranking Table")
        st.dataframe(ranking_df, use_container_width=True)

    ranking_fig = model_ranking_chart(ranking_df)
    if ranking_fig is not None:
        st.plotly_chart(ranking_fig, use_container_width=True)

    gain_fig = policy_gain_chart(ranking_df)
    if gain_fig is not None:
        st.plotly_chart(gain_fig, use_container_width=True)

    summary = load_champion_challenger_summary()
    st.subheader("Champion / Challenger Summary")
    if summary is None:
        st.info(
            "Champion/challenger summary not found at "
            "artifacts/reports/full/champion_challenger_summary.json"
        )
    else:
        st.json(summary)

    st.caption(
        "Even though transformer and causal transformer families were built, "
        "deployment follows measured out-of-sample ranking. The current "
        "production champion is s_learner_logistic."
    )


def _single_prediction_section(client: CausalUpliftApiClient) -> None:
    st.header("Single Prediction")
    defaults = get_default_feature_values()
    descriptions = get_feature_descriptions()

    values: dict[str, float] = {}
    cols = st.columns(4)
    for idx, feature in enumerate(REQUIRED_FEATURES):
        with cols[idx % 4]:
            values[feature] = st.number_input(
                feature,
                value=float(defaults[feature]),
                help=descriptions.get(feature, ""),
                format="%.6f",
            )

    if st.button("Predict Uplift", type="primary"):
        try:
            response = client.predict_single(values)
        except RuntimeError as exc:
            st.error(str(exc))
            return

        render_prediction_result(response)
        prediction = response.get("prediction", {})
        uplift = float(prediction.get("uplift", 0.0))
        if uplift > 0:
            st.success("Recommendation: target this user for treatment.")
        else:
            st.warning("Recommendation: do not target this user for treatment.")

        fig = prediction_probability_chart(prediction)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)


def _batch_prediction_section(client: CausalUpliftApiClient) -> None:
    st.header("Batch Prediction")
    st.caption("Upload a CSV with columns f0 through f11.")

    example_df = build_example_dataframe()
    st.download_button(
        "Download Example Input CSV",
        data=example_df.to_csv(index=False).encode("utf-8"),
        file_name="example_uplift_input.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Input CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        input_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        return

    missing = [feature for feature in REQUIRED_FEATURES if feature not in input_df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    st.dataframe(input_df.head(20), use_container_width=True)

    if st.button("Predict Batch", type="primary"):
        try:
            records = _numeric_records(input_df)
            response = client.predict_batch(records)
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")
            return

        predictions = response.get("predictions", [])
        predictions_df = pd.DataFrame(predictions)
        st.subheader("Predictions")
        st.dataframe(predictions_df, use_container_width=True)

        merged = pd.concat(
            [input_df.reset_index(drop=True), predictions_df.reset_index(drop=True)],
            axis=1,
        )
        csv_buffer = io.StringIO()
        merged.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download Predictions CSV",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="uplift_predictions.csv",
            mime="text/csv",
        )


def _about_section() -> None:
    st.header("About")
    st.markdown("""
- Dataset: Criteo Uplift Prediction full dataset
- Champion selection policy: primary metric `qini_auc`, tie-breaker `policy_gain_top20`
- Model families:
  - two_model_logistic
  - s_learner_logistic
  - t_learner_logistic
  - ft_transformer
  - ft_transformer_causal
  - ft_transformer_causal_ensemble
- Deployment:
  - FastAPI on Cloud Run
  - Streamlit app as dashboard/client
        """)


def main() -> None:
    st.sidebar.title("Causal Uplift")
    base_url = st.sidebar.text_input("API Base URL", value=get_api_base_url()).strip().rstrip("/")
    client = CausalUpliftApiClient(base_url=base_url)

    if st.sidebar.button("Check API Health"):
        try:
            health = client.health()
            st.sidebar.success(
                "API healthy: "
                f"status={health.get('status')} "
                f"model_loaded={health.get('model_loaded')}"
            )
        except RuntimeError as exc:
            st.sidebar.error(str(exc))

    section = st.sidebar.radio(
        "Navigation",
        ["Overview", "Model Performance", "Single Prediction", "Batch Prediction", "About"],
    )

    if section == "Overview":
        _overview_section(client)
    elif section == "Model Performance":
        _model_performance_section()
    elif section == "Single Prediction":
        _single_prediction_section(client)
    elif section == "Batch Prediction":
        _batch_prediction_section(client)
    else:
        _about_section()

    render_footer()


if __name__ == "__main__":
    main()
