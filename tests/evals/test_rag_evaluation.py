"""Tier 5: evaluation, against a real golden dataset, measuring
probabilistic quality, not per-question pass/fail correctness. The first
one this book's taxonomy actually has, chapter 16 (see chapter 8 and
chapter 10's placeholder notes naming this exact gap).

Unlike tiers 1-4, a single question failing is not automatically a bug,
model output has some real variance. This test asserts a threshold, not
a perfect score, and syncs the real dependency list first so the golden
dataset's expected citations stay meaningful against whatever this
project's real pyproject.toml says today.
"""

from reliable_agents_labs.evaluation import run_evaluation
from reliable_agents_labs.ingest import load_dependency_names, sync_packages
from reliable_agents_labs.models import build_embedding_client, build_model_client
from reliable_agents_labs.vector_store import build_qdrant_client

PASS_RATE_THRESHOLD = 0.75


async def test_rag_evaluation_meets_the_pass_rate_threshold():
    client = build_qdrant_client()
    embedder = build_embedding_client()
    model = build_model_client("answer_model")

    await sync_packages(load_dependency_names(), client, embedder)

    results = await run_evaluation(qdrant=client, embedder=embedder, model_client=model)
    passed = sum(r.passed for r in results)

    assert passed / len(results) >= PASS_RATE_THRESHOLD, [
        (r.question, r.expected, r.actual) for r in results if not r.passed
    ]
