"""Tier 3: integration test against real PyPI and real local Qdrant
together. See chapter 13.
"""

import uuid

from reliable_agents_labs.ingest import ingest_all
from reliable_agents_labs.models import build_embedding_client
from reliable_agents_labs.vector_store import build_qdrant_client


async def test_ingest_all_continues_past_one_bad_name():
    client = build_qdrant_client()
    embedder = build_embedding_client()
    collection = f"test_{uuid.uuid4().hex}"
    try:
        result = await ingest_all(
            ["httpx", f"not-a-real-package-{uuid.uuid4().hex}"],
            client,
            embedder,
            collection_name=collection,
        )
    finally:
        await client.delete_collection(collection)

    assert result["succeeded"] == ["httpx"]
    assert len(result["failed"]) == 1
