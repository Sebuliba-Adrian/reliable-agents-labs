"""Tier 2: agent orchestration tests, against a scripted/fake model, never
a real one. The fake exists to make control flow deterministic, not to
simulate intelligence, keep it deliberately boring.
"""

from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.orchestration import run_once


class ScriptedModelClient:
    """A deterministic stand-in for any real ModelClient adapter. Feed it a
    list of canned ModelResults; each call to generate() returns the next
    one. Reused across every orchestration test in this book, real chapters
    will feed it richer scripted sequences (e.g. "ask for a tool, then
    answer") as the reorder agent gains tool-calling and stateful workflow
    behavior.
    """

    def __init__(self, scripted_results: list[ModelResult]) -> None:
        self._results = iter(scripted_results)

    async def generate(self, *, system: str, user: str) -> ModelResult:
        return next(self._results)


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
