"""Tier 2: the reorder agent's structured-output path, against a scripted
model, never a real one. See chapter 5.
"""

import json

import pytest
from pydantic import ValidationError

from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.reorder_agent import ReorderAnswer, ask_reorder_agent_structured
from tests.fakes import ScriptedModelClient


def _scripted(text: str) -> ScriptedModelClient:
    return ScriptedModelClient(
        [
            ModelResult(
                text=text,
                input_tokens=20,
                output_tokens=12,
                model_id="scripted",
                provider="scripted",
            )
        ]
    )


async def test_structured_parses_json():
    fake = _scripted('{"can_answer": false, "answer": "I do not have real inventory data."}')
    result = await ask_reorder_agent_structured(
        "How many units of SKU-1029 are in stock?", client=fake
    )
    assert isinstance(result, ReorderAnswer)
    assert result.can_answer is False
    assert "do not have real inventory data" in result.answer


async def test_structured_strips_fences():
    fenced = '```json\n{"can_answer": true, "answer": "Reorder at 500 units."}\n```'
    fake = _scripted(fenced)
    result = await ask_reorder_agent_structured("What's a reasonable reorder point?", client=fake)
    assert result.can_answer is True
    assert result.answer == "Reorder at 500 units."


async def test_invalid_json_raises():
    """The exercise/failure case: not everything the model returns is
    valid JSON, even when told to always return it. Calling code must be
    ready for this to raise, not assume it never will.
    """
    fake = _scripted("Sure, here's the stock level: about 12 units, give or take.")
    with pytest.raises(json.JSONDecodeError):
        await ask_reorder_agent_structured("How many units are in stock?", client=fake)


async def test_schema_mismatch_raises():
    """Valid JSON, wrong shape, e.g. the model omits a required field.
    This is a distinct failure mode from invalid JSON, and needs its own
    handling, not the same catch-all.
    """
    fake = _scripted('{"answer": "Twelve units, roughly."}')
    with pytest.raises(ValidationError):
        await ask_reorder_agent_structured("How many units are in stock?", client=fake)
