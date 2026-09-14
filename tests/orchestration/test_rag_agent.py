"""Tier 2: the RAG agent's own orchestration tests, against scripted
fakes for all three of its dependencies (Qdrant, embedder, model),
never real ones. See chapter 15.
"""

from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.rag_agent import ask_rag_agent
from tests.fakes import (
    FakeScoredPoint,
    ScriptedEmbeddingClient,
    ScriptedModelClient,
    ScriptedQdrantClient,
)


def _model_result(text: str) -> ModelResult:
    return ModelResult(
        text=text, input_tokens=20, output_tokens=12, model_id="scripted", provider="scripted"
    )


async def test_answer_cites_the_retrieved_package():
    qdrant = ScriptedQdrantClient([FakeScoredPoint("httpx", 0.8)])
    embedder = ScriptedEmbeddingClient()
    model = ScriptedModelClient(
        [_model_result('{"answer": "httpx is an HTTP client.", "cited_packages": ["httpx"]}')]
    )

    result = await ask_rag_agent(
        "what handles HTTP requests?", qdrant=qdrant, embedder=embedder, model_client=model
    )

    assert result.answer == "httpx is an HTTP client."
    assert result.cited_packages == ["httpx"]


async def test_insufficient_context_leaves_citations_empty():
    qdrant = ScriptedQdrantClient([FakeScoredPoint("numpy", 0.1)])
    embedder = ScriptedEmbeddingClient()
    model = ScriptedModelClient(
        [
            _model_result(
                '{"answer": "The provided context does not cover that.", "cited_packages": []}'
            )
        ]
    )

    result = await ask_rag_agent(
        "how do I bake bread?", qdrant=qdrant, embedder=embedder, model_client=model
    )

    assert "does not cover" in result.answer
    assert result.cited_packages == []
