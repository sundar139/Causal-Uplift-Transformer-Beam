from __future__ import annotations

from fastapi.testclient import TestClient

from causal_uplift import serve


def test_health_contract() -> None:
    with TestClient(serve.app) as client:
        response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "model_loaded" in payload


def test_predict_uplift_requires_loaded_bundle(monkeypatch) -> None:
    monkeypatch.setenv("PRODUCTION_BUNDLE_DIR", "missing-test-bundle")
    monkeypatch.setattr(serve, "BUNDLE", None)
    monkeypatch.setattr(serve, "MODEL_LOADED", False)
    monkeypatch.setattr(serve, "STARTUP_ERROR", "bundle missing")

    with TestClient(serve.app) as client:
        response = client.post(
            "/predict_uplift",
            json={"features": {"f0": 0.1, "f1": 0.2}},
        )
    assert response.status_code == 503

    payload = response.json()
    assert "Production model bundle is not loaded" in payload["detail"]
