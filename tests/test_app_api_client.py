from __future__ import annotations

import pytest
from app import api_client
from app.api_client import CausalUpliftApiClient, get_api_base_url


class DummyResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_get_api_base_url_uses_default(monkeypatch) -> None:
    monkeypatch.delenv("CAUSAL_UPLIFT_API_URL", raising=False)
    assert get_api_base_url() == "http://127.0.0.1:8080"


def test_get_api_base_url_strips_trailing_slash(monkeypatch) -> None:
    monkeypatch.setenv("CAUSAL_UPLIFT_API_URL", "https://example.com/")
    assert get_api_base_url() == "https://example.com"


def test_health_calls_expected_endpoint(monkeypatch) -> None:
    called = {}

    def fake_get(url: str, timeout: int):
        called["url"] = url
        called["timeout"] = timeout
        return DummyResponse(200, {"status": "ok"})

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    client = CausalUpliftApiClient(base_url="https://api.test/")

    payload = client.health()
    assert called["url"] == "https://api.test/health"
    assert called["timeout"] == client.timeout
    assert payload["status"] == "ok"


def test_model_info_calls_expected_endpoint(monkeypatch) -> None:
    def fake_get(url: str, timeout: int):
        assert url.endswith("/model-info")
        return DummyResponse(200, {"model_name": "s_learner_logistic"})

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    client = CausalUpliftApiClient(base_url="https://api.test")

    payload = client.model_info()
    assert payload["model_name"] == "s_learner_logistic"


def test_predict_single_posts_expected_payload(monkeypatch) -> None:
    called = {}

    def fake_post(url: str, json: dict, timeout: int):
        called["url"] = url
        called["json"] = json
        called["timeout"] = timeout
        return DummyResponse(200, {"request_id": "abc"})

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    client = CausalUpliftApiClient(base_url="https://api.test")

    response = client.predict_single({"f0": 0.1})
    assert called["url"] == "https://api.test/predict_uplift"
    assert called["json"] == {"features": {"f0": 0.1}}
    assert response["request_id"] == "abc"


def test_predict_batch_posts_expected_payload(monkeypatch) -> None:
    def fake_post(url: str, json: dict, timeout: int):
        assert url.endswith("/predict_batch")
        assert json == {"records": [{"f0": 0.1}]}
        return DummyResponse(200, {"predictions": []})

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    client = CausalUpliftApiClient(base_url="https://api.test")

    response = client.predict_batch([{"f0": 0.1}])
    assert response["predictions"] == []


def test_non_200_raises_clear_error(monkeypatch) -> None:
    def fake_get(url: str, timeout: int):
        return DummyResponse(503, text="service unavailable")

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    client = CausalUpliftApiClient(base_url="https://api.test")

    with pytest.raises(RuntimeError, match="status=503"):
        client.health()
