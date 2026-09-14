"""Tier 3: integration test against a real, local Qdrant, running via
`docker compose up qdrant` (or CI's own `services:` container, see
.github/workflows/ci.yml). This is the taxonomy's own textbook case,
finally: a real local service this project runs itself. See chapter 12.
"""

import uuid

from qdrant_client.models import Distance, VectorParams

from reliable_agents_labs.vector_store import build_qdrant_client, search_packages, upsert_package


async def test_store_and_search_round_trip():
    client = build_qdrant_client()
    # A fresh, disposable collection per test run, real Qdrant, real
    # network calls, never touching the "packages" collection a reader's
    # own ingestion (chapter 13) fills in.
    collection = f"test_{uuid.uuid4().hex}"
    await client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    try:
        await upsert_package(
            client,
            package_id=1,
            name="alpha",
            summary="alpha",
            vector=[1.0, 0.0, 0.0, 0.0],
            collection_name=collection,
        )
        await upsert_package(
            client,
            package_id=2,
            name="beta",
            summary="beta",
            vector=[0.0, 1.0, 0.0, 0.0],
            collection_name=collection,
        )

        results = await search_packages(
            client, query_vector=[1.0, 0.0, 0.0, 0.0], limit=1, collection_name=collection
        )
    finally:
        await client.delete_collection(collection)

    assert results[0]["name"] == "alpha"
