"""Tier 2: the reorder agent's own orchestration tests, against a
scripted model, never a real one.
"""

from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.reorder_agent import ask_reorder_agent
from tests.fakes import ScriptedModelClient


async def test_ask_reorder_agent_returns_the_model_text():
    fake = ScriptedModelClient(
        [
            ModelResult(
                text="I do not have real inventory data to check that.",
                input_tokens=20,
                output_tokens=12,
                model_id="scripted",
                provider="scripted",
            )
        ]
    )
    answer = await ask_reorder_agent("How many units of SKU-1029 are in stock?", client=fake)
    assert "do not have real inventory data" in answer
