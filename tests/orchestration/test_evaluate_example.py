"""Tier 2: evaluate_example's own retrieval/generation decomposition,
against scripted fakes for all of ask_rag_agent's dependencies, never a
real retriever or model. See chapter 16's original golden-set scoring
and its own extension: `retrieval_hit` answers "did retrieval find it,"
independent of what the model cited.
"""

from reliable_agents_labs.evaluation import GoldenExample, evaluate_example
from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.rag_agent import RagAnswer
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


async def test_retrieval_hit_true_and_passed_true_when_cited_correctly():
    example = GoldenExample("what handles HTTP requests?", "httpx")
    qdrant = ScriptedQdrantClient([FakeScoredPoint("httpx", 0.8)])
    model = ScriptedModelClient(
        [_model_result('{"answer": "httpx is an HTTP client.", "cited_packages": ["httpx"]}')]
    )

    result = await evaluate_example(
        example, qdrant=qdrant, embedder=ScriptedEmbeddingClient(), model_client=model
    )

    assert result.retrieved == ["httpx"]
    assert result.retrieval_hit is True
    assert result.passed is True


async def test_retrieval_hit_true_but_passed_false_is_a_generation_miss():
    """Retrieval did its job, httpx came back, the model just never
    cited it. This is exactly the case pass_rate alone cannot
    distinguish from a retrieval miss.
    """
    example = GoldenExample("what handles HTTP requests?", "httpx")
    qdrant = ScriptedQdrantClient([FakeScoredPoint("httpx", 0.8)])
    model = ScriptedModelClient(
        [_model_result('{"answer": "The context does not say.", "cited_packages": []}')]
    )

    result = await evaluate_example(
        example, qdrant=qdrant, embedder=ScriptedEmbeddingClient(), model_client=model
    )

    assert result.retrieval_hit is True
    assert result.passed is False


async def test_retrieval_hit_false_is_a_retrieval_miss():
    """httpx never came back from retrieval at all, so the model could
    not have cited it no matter how good the generation step was.
    """
    example = GoldenExample("what handles HTTP requests?", "httpx")
    qdrant = ScriptedQdrantClient([FakeScoredPoint("numpy", 0.1)])
    model = ScriptedModelClient(
        [_model_result('{"answer": "The context does not say.", "cited_packages": []}')]
    )

    result = await evaluate_example(
        example, qdrant=qdrant, embedder=ScriptedEmbeddingClient(), model_client=model
    )

    assert result.retrieval_hit is False
    assert result.passed is False


async def test_retrieval_hit_is_none_when_no_citation_is_expected():
    example = GoldenExample("what handles Kubernetes deployments?", None)
    qdrant = ScriptedQdrantClient([FakeScoredPoint("numpy", 0.1)])
    model = ScriptedModelClient(
        [_model_result('{"answer": "The context does not say.", "cited_packages": []}')]
    )

    result = await evaluate_example(
        example, qdrant=qdrant, embedder=ScriptedEmbeddingClient(), model_client=model
    )

    assert result.retrieval_hit is None
    assert result.passed is True


async def test_retrieval_hit_true_when_the_expected_package_is_only_a_dependent():
    """A real bug this evaluation found the first time it ran live
    against ask_graph_rag_agent: hybrid_search (chapter 20) can surface
    a package as another result's `dependents`, not just as a
    top-level result of its own, and build_hybrid_context puts that
    dependent in the model's context either way. A retrieval check
    that only reads `name` would call this a miss even though the
    model genuinely had the package in front of it.
    """
    example = GoldenExample("if pydantic had a breaking change, what would break?", "fastapi")

    async def ask_fn(question: str, on_retrieval=None, **kwargs) -> RagAnswer:
        if on_retrieval is not None:
            on_retrieval([{"name": "pydantic", "dependents": ["fastapi", "langfuse"]}])
        return RagAnswer(answer="scripted", cited_packages=["fastapi"])

    result = await evaluate_example(example, ask_fn=ask_fn)

    assert result.retrieval_hit is True
    assert result.passed is True
