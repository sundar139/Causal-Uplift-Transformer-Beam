from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from causal_uplift import serve


class FakeBundle:
    model_name = "s_learner_logistic"
    created_at_utc = "2026-05-03T00:00:00+00:00"
    metadata = {
        "selection_metric": "qini_auc",
        "dataset_variant": "full",
        "qini_auc": 0.2,
        "uplift_auc": 0.1,
        "policy_gain_top10": 0.03,
        "policy_gain_top20": 0.02,
        "policy_gain_top30": 0.01,
        "treatment_response_auc": 0.8,
    }
    feature_schema = {"feature_count": 2}
    required_columns = ["f0", "f1"]

    def predict_records(self, records: list[dict[str, float]]) -> list[dict[str, Any]]:
        for record in records:
            missing = [column for column in self.required_columns if column not in record]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")
        return [
            {
                "treatment_probability": 0.7,
                "control_probability": 0.4,
                "uplift": 0.3,
                "recommend_treatment": True,
            }
            for _ in records
        ]


def _client_with_fake_bundle(monkeypatch) -> TestClient:
    monkeypatch.setattr(serve, "BUNDLE", FakeBundle())
    monkeypatch.setattr(serve, "MODEL_LOADED", True)
    monkeypatch.setattr(serve, "STARTUP_ERROR", None)
    return TestClient(serve.app)


def test_health_contract(monkeypatch) -> None:
    client = _client_with_fake_bundle(monkeypatch)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_name"] == "s_learner_logistic"


def test_model_info_contract(monkeypatch) -> None:
    client = _client_with_fake_bundle(monkeypatch)
    response = client.get("/model-info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "s_learner_logistic"
    assert payload["dataset_variant"] == "full"
    assert payload["feature_count"] == 2
    assert payload["required_columns"] == ["f0", "f1"]


def test_predict_uplift_response_fields(monkeypatch) -> None:
    client = _client_with_fake_bundle(monkeypatch)
    response = client.post("/predict_uplift", json={"features": {"f0": 0.1, "f1": "0.2"}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"]
    assert payload["model_name"] == "s_learner_logistic"
    assert payload["model_version"] == "2026-05-03T00:00:00+00:00"
    assert payload["prediction"]["recommend_treatment"] is True


def test_predict_batch_response_fields(monkeypatch) -> None:
    client = _client_with_fake_bundle(monkeypatch)
    response = client.post(
        "/predict_batch",
        json={"records": [{"f0": 0.1, "f1": 0.2}, {"f0": 0.3, "f1": 0.4}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"]
    assert len(payload["predictions"]) == 2
    assert payload["predictions"][0]["uplift"] == 0.3


def test_missing_required_feature_returns_clean_error(monkeypatch) -> None:
    client = _client_with_fake_bundle(monkeypatch)
    response = client.post("/predict_uplift", json={"features": {"f0": 0.1}})

    assert response.status_code == 400
    assert "Missing required feature columns" in response.json()["detail"]
    assert response.request is not None
