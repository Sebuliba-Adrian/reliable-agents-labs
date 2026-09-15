"""Tier 2: the graph-RAG agent's own orchestration tests, against
scripted fakes for all four of its dependencies (Qdrant, the graph
driver, the embedder, the model), never real ones. See chapter 21.
"""

from reliable_agents_labs.graph_rag_agent import ask_graph_rag_agent
from reliable_agents_labs.models import ModelResult
from tests.fakes import (
    FakeScoredPoint,
    ScriptedEmbeddingClient,
    ScriptedGraphDriver,
    ScriptedModelClient,
    ScriptedQdrantClient,
)


def _model_result(text: str) -> ModelResult:
    return ModelResult(
        text=text, input_tokens=20, output_tokens=12, model_id="scripted", provider="scripted"
    )


async def test_answer_cites_a_stored_dependent_not_just_the_vector_hit():
    qdrant = ScriptedQdrantClient([FakeScoredPoint("pydantic", 0.9)])
    driver = ScriptedGraphDriver({"pydantic": ["fastapi"]})
    embedder = ScriptedEmbeddingClient()
    model = ScriptedModelClient(
        [
            _model_result(
                '{"answer": "fastapi depends on pydantic.", '
                '"cited_packages": ["pydantic", "fastapi"]}'
            )
        ]
    )

    result = await ask_graph_rag_agent(
        "what would break if pydantic changed?",
        qdrant=qdrant,
        driver=driver,
        embedder=embedder,
        model_client=model,
    )

    assert result.cited_packages == ["pydantic", "fastapi"]


async def test_no_dependents_still_produces_a_grounded_answer():
    qdrant = ScriptedQdrantClient([FakeScoredPoint("httpx", 0.8)])
    driver = ScriptedGraphDriver()
    embedder = ScriptedEmbeddingClient()
    model = ScriptedModelClient(
        [_model_result('{"answer": "httpx is an HTTP client.", "cited_packages": ["httpx"]}')]
    )

    result = await ask_graph_rag_agent(
        "what handles HTTP requests?",
        qdrant=qdrant,
        driver=driver,
        embedder=embedder,
        model_client=model,
    )

    assert result.answer == "httpx is an HTTP client."


async def test_on_retrieval_sees_hybrid_searchs_raw_results():
    qdrant = ScriptedQdrantClient([FakeScoredPoint("pydantic", 0.9)])
    driver = ScriptedGraphDriver({"pydantic": ["fastapi"]})
    embedder = ScriptedEmbeddingClient()
    model = ScriptedModelClient(
        [
            _model_result(
                '{"answer": "fastapi depends on pydantic.", '
                '"cited_packages": ["pydantic", "fastapi"]}'
            )
        ]
    )
    seen = []

    await ask_graph_rag_agent(
        "what would break if pydantic changed?",
        qdrant=qdrant,
        driver=driver,
        embedder=embedder,
        model_client=model,
        on_retrieval=seen.append,
    )

    assert len(seen) == 1
    assert seen[0][0]["name"] == "pydantic"
    assert seen[0][0]["dependents"] == ["fastapi"]
