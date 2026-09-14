"""Shared test fakes. Not a test file itself, imported by tier-2
(orchestration) tests across every chapter.
"""

from reliable_agents_labs.models import ModelResult


class ScriptedModelClient:
    """A deterministic stand-in for any real ModelClient adapter. Feed it a
    list of canned ModelResults; each call to generate() returns the next
    one. Its purpose is to make control flow deterministic, not to
    simulate intelligence, keep scripts boring.
    """

    def __init__(self, scripted_results: list[ModelResult]) -> None:
        self._results = iter(scripted_results)

    async def generate(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> ModelResult:
        return next(self._results)
