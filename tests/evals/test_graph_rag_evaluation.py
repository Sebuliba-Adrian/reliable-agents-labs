"""Tier 5: evaluation for the graph-RAG agent, chapter 17's own
structural question as a golden example, scored against
`ask_graph_rag_agent` instead of the plain vector-only agent chapter
16's evaluation dataset was written for. See chapter 21.
"""

from reliable_agents_labs.evaluation import (
    STRUCTURAL_GOLDEN_DATASET,
    retrieval_recall,
    run_evaluation,
)
from reliable_agents_labs.graph_etl import load_dependency_graph
from reliable_agents_labs.graph_rag_agent import ask_graph_rag_agent
from reliable_agents_labs.graph_store import build_neo4j_driver
from reliable_agents_labs.ingest import load_dependency_names, sync_packages
from reliable_agents_labs.models import build_embedding_client, build_model_client
from reliable_agents_labs.vector_store import build_qdrant_client

PASS_RATE_THRESHOLD = 0.75
RETRIEVAL_RECALL_THRESHOLD = 0.75


async def test_graph_rag_evaluation_meets_the_pass_rate_threshold():
    qdrant = build_qdrant_client()
    driver = build_neo4j_driver()
    embedder = build_embedding_client()
    model = build_model_client("answer_model")

    names = load_dependency_names()
    await sync_packages(names, qdrant, embedder)
    await load_dependency_graph(names, driver)

    try:
        results = await run_evaluation(
            dataset=STRUCTURAL_GOLDEN_DATASET,
            ask_fn=ask_graph_rag_agent,
            qdrant=qdrant,
            driver=driver,
            embedder=embedder,
            model_client=model,
        )
        passed = sum(r.passed for r in results)

        assert passed / len(results) >= PASS_RATE_THRESHOLD, [
            (r.question, r.expected, r.actual) for r in results if not r.passed
        ]
        assert retrieval_recall(results) >= RETRIEVAL_RECALL_THRESHOLD, [
            (r.question, r.expected, r.retrieved) for r in results if r.retrieval_hit is False
        ]
    finally:
        await driver.close()
