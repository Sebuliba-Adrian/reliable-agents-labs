"""Tier 3: sync_packages against real PyPI and real local Qdrant
together, chapter 14's idempotency guarantee end to end.
"""

import uuid

from reliable_agents_labs.ingest import sync_packages
from reliable_agents_labs.models import build_embedding_client
from reliable_agents_labs.vector_store import build_qdrant_client


async def test_running_sync_twice_does_not_duplicate():
    client = build_qdrant_client()
    embedder = build_embedding_client()
    collection = f"test_{uuid.uuid4().hex}"
    try:
        await sync_packages(["httpx"], client, embedder, collection_name=collection)
        first_count = (await client.get_collection(collection)).points_count

        await sync_packages(["httpx"], client, embedder, collection_name=collection)
        second_count = (await client.get_collection(collection)).points_count
    finally:
        await client.delete_collection(collection)

    assert first_count == 1
    assert second_count == 1


async def test_sync_prunes_a_name_no_longer_in_the_list():
    client = build_qdrant_client()
    embedder = build_embedding_client()
    collection = f"test_{uuid.uuid4().hex}"
    try:
        await sync_packages(["httpx", "fastapi"], client, embedder, collection_name=collection)
        result = await sync_packages(["httpx"], client, embedder, collection_name=collection)
        remaining = (await client.get_collection(collection)).points_count
    finally:
        await client.delete_collection(collection)

    assert result["pruned"] == ["fastapi"]
    assert remaining == 1
