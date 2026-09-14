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


class ScriptedEmbeddingClient:
    """A deterministic stand-in for any real EmbeddingClient adapter.
    Chapter 15's orchestration tests care about what `ask_rag_agent` does
    with a given question and a given retrieval result, not about
    embedding text for real, so the vector itself can be anything
    fixed-length and consistent.
    """

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector or [0.0] * 8

    async def embed(self, text: str) -> list[float]:
        return self._vector


class FakeScoredPoint:
    """Just enough shape to stand in for qdrant_client's real ScoredPoint:
    `search_packages` only ever reads `.payload["name"]`,
    `.payload["summary"]`, and `.score`.
    """

    def __init__(self, name: str, score: float, summary: str = "") -> None:
        self.payload = {"name": name, "summary": summary or f"{name} summary"}
        self.score = score


class ScriptedQdrantClient:
    """A deterministic stand-in for AsyncQdrantClient, implementing only
    `query_points`, the one method `search_packages` calls. Chapter 15's
    orchestration tests exercise `ask_rag_agent`'s own logic, prompt
    construction and answer parsing, real retrieval is chapter 12 and
    13's job, already verified against a real Qdrant there.
    """

    def __init__(self, points: list[FakeScoredPoint]) -> None:
        self._points = points

    async def query_points(self, *, collection_name: str, query: list[float], limit: int):
        class _Response:
            pass

        response = _Response()
        response.points = self._points
        return response
