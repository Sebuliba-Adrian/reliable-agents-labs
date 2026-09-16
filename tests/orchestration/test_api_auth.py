"""Tier 2: chapter 30's own API-key guardrail on `/ask`, against a
scripted model, never a real one. tests/orchestration/conftest.py
bypasses this guardrail by default for every other test in this tier;
these tests remove that bypass to exercise the real dependency.
"""

from fastapi.testclient import TestClient

import reliable_agents_labs.api as api_module
from reliable_agents_labs.api import app, get_model_client, require_api_key
from reliable_agents_labs.models import ModelResult
from tests.fakes import ScriptedModelClient


def _result(text: str) -> ModelResult:
    return ModelResult(
        text=text, input_tokens=5, output_tokens=5, model_id="scripted", provider="scripted"
    )


def test_ask_rejects_a_request_with_no_api_key(monkeypatch):
    app.dependency_overrides.pop(require_api_key, None)
    monkeypatch.setattr(api_module, "API_KEY", "test-key")
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Hello?"})

    assert response.status_code == 401


def test_ask_rejects_the_wrong_api_key(monkeypatch):
    app.dependency_overrides.pop(require_api_key, None)
    monkeypatch.setattr(api_module, "API_KEY", "test-key")
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Hello?"}, headers={"X-API-Key": "wrong-key"})

    assert response.status_code == 401


def test_ask_accepts_the_real_api_key(monkeypatch):
    app.dependency_overrides.pop(require_api_key, None)
    monkeypatch.setattr(api_module, "API_KEY", "test-key")
    fake = ScriptedModelClient([_result("ok")])
    app.dependency_overrides[get_model_client] = lambda: fake
    try:
        client = TestClient(app)
        response = client.post(
            "/ask", json={"question": "Hello?"}, headers={"X-API-Key": "test-key"}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200


def test_ask_fails_closed_when_no_key_is_configured(monkeypatch):
    app.dependency_overrides.pop(require_api_key, None)
    monkeypatch.setattr(api_module, "API_KEY", None)
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Hello?"}, headers={"X-API-Key": "anything"})

    assert response.status_code == 401
