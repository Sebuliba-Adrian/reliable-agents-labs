"""Tier 2: chapter 30's own input and timeout guardrails on the HTTP
layer, against a scripted model, never a real one.
"""

import asyncio

from fastapi.testclient import TestClient

import reliable_agents_labs.api as api_module
from reliable_agents_labs.api import MAX_QUESTION_LENGTH, app, get_model_client
from reliable_agents_labs.models import ModelResult
from tests.fakes import ScriptedModelClient


def _result(text: str) -> ModelResult:
    return ModelResult(
        text=text, input_tokens=5, output_tokens=5, model_id="scripted", provider="scripted"
    )


def test_ask_rejects_a_question_over_the_real_length_limit():
    client = TestClient(app)
    oversized = "x" * (MAX_QUESTION_LENGTH + 1)

    response = client.post("/ask", json={"question": oversized})

    assert response.status_code == 422


def test_ask_rejects_an_empty_question():
    client = TestClient(app)

    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_accepts_a_question_at_exactly_the_limit():
    fake = ScriptedModelClient([_result("ok")])
    app.dependency_overrides[get_model_client] = lambda: fake
    try:
        client = TestClient(app)
        at_limit = "x" * MAX_QUESTION_LENGTH
        response = client.post("/ask", json={"question": at_limit})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200


async def _sleep_forever_generate(**kwargs):
    await asyncio.sleep(3)
    return _result("too late")


class _HangingClient:
    generate = staticmethod(_sleep_forever_generate)


def test_ask_returns_504_when_the_model_call_hangs_past_the_timeout(monkeypatch):
    monkeypatch.setattr(api_module, "REQUEST_TIMEOUT_SECONDS", 0.05)
    app.dependency_overrides[get_model_client] = lambda: _HangingClient()
    try:
        client = TestClient(app)
        response = client.post("/ask", json={"question": "Hello?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 504
