"""Tier 3: hybrid_search against real, local Qdrant and real, local
Neo4j together, one disposable collection and two disposable nodes,
scripted embedding vector only (no real model call, that stays tier 4,
chapter 15's own contract tests already cover the real embedder). See
chapter 20.
"""

import uuid

from qdrant_client.models import Distance, VectorParams

from reliable_agents_labs.graph_store import build_neo4j_driver, create_depends_on
from reliable_agents_labs.hybrid_retrieval import hybrid_search
from reliable_agents_labs.vector_store import build_qdrant_client, upsert_package
from tests.fakes import ScriptedEmbeddingClient


async def test_hybrid_search_combines_a_real_vector_hit_with_a_real_graph_edge():
    qdrant = build_qdrant_client()
    driver = build_neo4j_driver()
    collection = f"test_{uuid.uuid4().hex}"
    suffix = uuid.uuid4().hex[:8]
    target = f"target-{suffix}"
    dependent = f"dependent-{suffix}"
    await qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    try:
        await upsert_package(
            qdrant,
            package_id=1,
            name=target,
            summary="a real package, for this test only",
            vector=[1.0, 0.0, 0.0, 0.0],
            collection_name=collection,
        )
        await create_depends_on(driver, dependent, target)

        results = await hybrid_search(
            "irrelevant text, the embedding below is scripted",
            qdrant,
            driver,
            ScriptedEmbeddingClient([1.0, 0.0, 0.0, 0.0]),
            limit=1,
            collection_name=collection,
        )

        assert results[0]["name"] == target
        assert results[0]["dependents"] == [dependent]
    finally:
        await qdrant.delete_collection(collection)
        async with driver.session() as session:
            await session.run(
                "MATCH (p:Package) WHERE p.name IN $names DETACH DELETE p",
                names=[target, dependent],
            )
        await driver.close()
