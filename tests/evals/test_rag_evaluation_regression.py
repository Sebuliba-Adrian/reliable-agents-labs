"""Tier 5: chapter 16's own real evaluation, run again, this time
compared against the last real pass rate actually recorded on disk,
not just chapter 16's fixed 0.75 floor. See chapter 29.
"""

from reliable_agents_labs.evaluation import run_evaluation
from reliable_agents_labs.ingest import load_dependency_names, sync_packages
from reliable_agents_labs.models import build_embedding_client, build_model_client
from reliable_agents_labs.regression import check_for_regression, save_baseline
from reliable_agents_labs.vector_store import build_qdrant_client


def _pass_rate(results) -> float:
    return sum(r.passed for r in results) / len(results)


async def test_rag_evaluation_has_not_regressed_since_the_last_recorded_run():
    client = build_qdrant_client()
    embedder = build_embedding_client()
    model = build_model_client("answer_model")

    await sync_packages(load_dependency_names(), client, embedder)
    results = await run_evaluation(qdrant=client, embedder=embedder, model_client=model)
    rate = _pass_rate(results)

    regression = check_for_regression("rag_evaluation", rate)
    assert regression is None, regression

    save_baseline("rag_evaluation", rate)
