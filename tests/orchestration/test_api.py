"""Tier 2: the HTTP layer's own orchestration tests, against a scripted
model, never a real one. See chapter 7.
"""

from fastapi.testclient import TestClient

from reliable_agents_labs.api import app, get_model_client
from reliable_agents_labs.models import ModelResult, ToolCall
from tests.fakes import ScriptedModelClient


def _result(text: str = "", tool_calls: list[ToolCall] | None = None) -> ModelResult:
    return ModelResult(
        text=text,
        input_tokens=20,
        output_tokens=12,
        model_id="scripted",
        provider="scripted",
        tool_calls=tool_calls or [],
    )


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_a_grounded_answer():
    fake = ScriptedModelClient(
        [
            _result(
                tool_calls=[
                    ToolCall(id="call_1", name="check_inventory", arguments={"sku": "SKU-1029"})
                ]
            ),
            _result(text="SKU-1029 has 4 units in stock, below its reorder point of 20."),
        ]
    )
    app.dependency_overrides[get_model_client] = lambda: fake
    try:
        client = TestClient(app)
        response = client.post("/ask", json={"question": "How many units of SKU-1029?"})
    finally:
        app.dependency_overrides.clear()

    expected = "SKU-1029 has 4 units in stock, below its reorder point of 20."
    assert response.status_code == 200
    assert response.json() == {"answer": expected}


def test_ask_rejects_a_missing_question_field():
    client = TestClient(app)
    response = client.post("/ask", json={})
    assert response.status_code == 422
