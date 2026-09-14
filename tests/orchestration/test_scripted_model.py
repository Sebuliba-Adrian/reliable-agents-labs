"""Tier 2: agent orchestration tests, against a scripted/fake model, never
a real one. See tests/fakes.py for the shared ScriptedModelClient.
"""

from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.orchestration import run_once
from tests.fakes import ScriptedModelClient


async def test_run_once_returns_the_scripted_text():
    fake = ScriptedModelClient(
        [
            ModelResult(
                text="in stock: 12 units",
                input_tokens=10,
                output_tokens=5,
                model_id="scripted",
                provider="scripted",
            )
        ]
    )
    text = await run_once(fake, system="you are a reorder agent", user="check stock")
    assert text == "in stock: 12 units"
