from __future__ import annotations

from fastapi.testclient import TestClient

from causal_uplift import serve


def test_health_contract() -> None:
    client = TestClient(serve.app)
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload


def test_predict_uplift_contract_placeholder() -> None:
    serve.MODEL = None
    serve.MODEL_LOADED = False

    client = TestClient(serve.app)
    response = client.post(
        "/predict_uplift",
        json={"rows": [{"features": [0.1, 0.2, 0.3]}]},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["model_loaded"] is False
    assert len(payload["predictions"]) == 1

    item = payload["predictions"][0]
    assert set(item.keys()) == {
        "treatment_probability",
        "control_probability",
        "uplift",
        "recommend_treatment",
    }
