from __future__ import annotations

import streamlit as st


def render_header() -> None:
    st.title("Causal Uplift Dashboard")
    st.caption("Production inference dashboard for uplift targeting decisions.")
    st.info(
        "Positive uplift means treatment is expected to increase conversion probability. "
        "Use predictions as decision support, not guaranteed outcomes."
    )


def render_metric_cards(model_info: dict) -> None:
    metrics = model_info.get("metrics", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Qini AUC", f"{float(metrics.get('qini_auc', 0.0)):.4f}")
    c2.metric("Uplift AUC", f"{float(metrics.get('uplift_auc', 0.0)):.4f}")
    c3.metric("Policy Gain Top10", f"{float(metrics.get('policy_gain_top10', 0.0)):.4f}")
    c4.metric("Policy Gain Top20", f"{float(metrics.get('policy_gain_top20', 0.0)):.4f}")
    c5.metric(
        "Treatment Response AUC",
        f"{float(metrics.get('treatment_response_auc', 0.0)):.4f}",
    )


def render_model_summary(model_info: dict) -> None:
    st.subheader("Champion Model Summary")
    col1, col2, col3 = st.columns(3)
    col1.write(f"**Model**: {model_info.get('model_name', 'unknown')}")
    col2.write(f"**Version**: {model_info.get('model_version', 'unknown')}")
    col3.write(f"**Dataset Variant**: {model_info.get('dataset_variant', 'unknown')}")
    st.write(
        "The deployed champion is selected by the project ranking policy: primary metric qini_auc "
        "with policy_gain_top20 as tie-breaker."
    )


def render_prediction_result(response: dict) -> None:
    prediction = response.get("prediction", {})
    st.subheader("Prediction Result")
    a, b, c, d = st.columns(4)
    a.metric("Treatment Probability", f"{float(prediction.get('treatment_probability', 0.0)):.4f}")
    b.metric("Control Probability", f"{float(prediction.get('control_probability', 0.0)):.4f}")
    c.metric("Uplift", f"{float(prediction.get('uplift', 0.0)):.4f}")
    d.metric("Recommend Treatment", str(bool(prediction.get("recommend_treatment", False))))
    st.caption(f"Request ID: {response.get('request_id', 'n/a')}")


def render_footer() -> None:
    st.divider()
    st.caption(
        "Built for causal-uplift-transformer-beam. "
        "FastAPI serves production predictions; Streamlit provides dashboard and client UX."
    )
