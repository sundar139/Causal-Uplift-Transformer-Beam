from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from causal_uplift import serve


def test_health_contract() -> None:
    with TestClient(serve.app) as client:
        response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_loaded" in payload


def test_predict_uplift_contract_placeholder(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    serve.MODEL = None
    serve.MODEL_LOADED = False

    with TestClient(serve.app) as client:
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
