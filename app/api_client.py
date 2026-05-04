from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_API_BASE_URL = "http://127.0.0.1:8080"
REQUEST_TIMEOUT_SECONDS = 15


def get_api_base_url() -> str:
    url = os.getenv("CAUSAL_UPLIFT_API_URL", DEFAULT_API_BASE_URL).strip()
    if not url:
        return DEFAULT_API_BASE_URL
    return url.rstrip("/")


class CausalUpliftApiClient:
    def __init__(self, base_url: str | None = None, timeout: int = REQUEST_TIMEOUT_SECONDS) -> None:
        chosen_base_url = (base_url or get_api_base_url()).strip()
        if not chosen_base_url:
            chosen_base_url = DEFAULT_API_BASE_URL
        self.base_url = chosen_base_url.rstrip("/")
        self.timeout = timeout

    def _handle_response(self, response: requests.Response, endpoint: str) -> dict[str, Any]:
        if response.status_code != 200:
            detail = response.text.strip()[:500]
            raise RuntimeError(
                f"API request failed for {endpoint}: status={response.status_code}, detail={detail}"
            )
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise RuntimeError(f"API request failed for {endpoint}: invalid JSON response") from exc
        return payload

    def _get(self, endpoint: str) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"API request failed for {endpoint}: {exc}") from exc
        return self._handle_response(response, endpoint)

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"API request failed for {endpoint}: {exc}") from exc
        return self._handle_response(response, endpoint)

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def model_info(self) -> dict[str, Any]:
        return self._get("/model-info")

    def predict_single(self, features: dict[str, float]) -> dict[str, Any]:
        return self._post("/predict_uplift", {"features": features})

    def predict_batch(self, records: list[dict[str, float]]) -> dict[str, Any]:
        return self._post("/predict_batch", {"records": records})
