"""Tier 2: hybrid_search's own orchestration tests, against scripted
fakes for Qdrant, the embedder, and the graph driver, never real ones.
See chapter 20.
"""

from reliable_agents_labs.hybrid_retrieval import build_hybrid_context, hybrid_search
from tests.fakes import (
    FakeScoredPoint,
    ScriptedEmbeddingClient,
    ScriptedGraphDriver,
    ScriptedQdrantClient,
)


async def test_hybrid_search_enriches_each_result_with_its_dependents():
    qdrant = ScriptedQdrantClient([FakeScoredPoint("pydantic", 0.9)])
    driver = ScriptedGraphDriver({"pydantic": ["fastapi", "anthropic"]})
    embedder = ScriptedEmbeddingClient()

    results = await hybrid_search("breaking change question", qdrant, driver, embedder)

    assert results[0]["name"] == "pydantic"
    assert results[0]["dependents"] == ["fastapi", "anthropic"]


async def test_hybrid_search_leaves_dependents_empty_when_graph_has_none():
    qdrant = ScriptedQdrantClient([FakeScoredPoint("httpx", 0.7)])
    driver = ScriptedGraphDriver()
    embedder = ScriptedEmbeddingClient()

    results = await hybrid_search("what handles HTTP requests?", qdrant, driver, embedder)

    assert results[0]["dependents"] == []


def test_build_hybrid_context_appends_dependents_clause_only_when_present():
    results = [
        {"name": "pydantic", "summary": "Data validation.", "dependents": ["fastapi"]},
        {"name": "httpx", "summary": "An HTTP client.", "dependents": []},
    ]

    context = build_hybrid_context(results)

    assert "pydantic: Data validation. (depended on by: fastapi)" in context
    assert "httpx: An HTTP client." in context
    assert "httpx: An HTTP client. (depended on by:" not in context
