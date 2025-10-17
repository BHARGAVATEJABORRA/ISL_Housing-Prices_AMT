"""FastAPI smoke tests."""
from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - handled by skip
    TestClient = None  # type: ignore[assignment]

if TestClient:
    from api.main import app


@pytest.mark.usefixtures("fastapi_available")
def test_predict_without_model():
    client = TestClient(app)
    payload = {"rows": [{"Overall_Qual": 7, "Gr_Liv_Area": 1500}]}
    response = client.post("/predict", json=payload)
    assert response.status_code in {200, 503}


@pytest.mark.usefixtures("fastapi_available")
def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
