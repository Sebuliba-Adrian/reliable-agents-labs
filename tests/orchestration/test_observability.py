"""Tier 2: the traced wrapper's own orchestration test, against a
scripted model, never a real one. The model call stays fully scripted
either way; whether a trace also reaches Langfuse depends on whether
LANGFUSE_PUBLIC_KEY is set (from .env, once chapter 9's stack is
running). Either way this test's assertion holds: without a key, the
Langfuse client disables itself and @observe's calls are harmless
no-ops; with one, they are a real, local, fire-and-forget OTLP export
that cannot fail this test, Langfuse's client is built not to block or
raise on the application's behalf even if its backend is unreachable.
See chapter 9.
"""

from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.observability import ask_reorder_agent_traced
from tests.fakes import ScriptedModelClient


async def test_traced_wrapper_returns_the_same_answer():
    fake = ScriptedModelClient(
        [
            ModelResult(
                text="A safety stock buffer covers demand spikes.",
                input_tokens=10,
                output_tokens=8,
                model_id="scripted",
                provider="scripted",
            )
        ]
    )
    answer = await ask_reorder_agent_traced("What is safety stock, in general?", client=fake)
    assert answer == "A safety stock buffer covers demand spikes."
