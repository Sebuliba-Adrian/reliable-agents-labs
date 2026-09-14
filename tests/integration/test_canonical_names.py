"""Tier 3: the real bug chapter 16 found, reproduced and pinned down.
PyPI's canonical name for a dependency is not always the same string as
how it is spelled in pyproject.toml ("pyyaml" vs. the real "PyYAML").
`sync_packages` must not delete a package it just stored because the
input spelling and the stored spelling differ.
"""

import uuid

from reliable_agents_labs.ingest import sync_packages
from reliable_agents_labs.models import build_embedding_client
from reliable_agents_labs.vector_store import build_qdrant_client


async def test_sync_survives_a_name_pypi_spells_differently():
    client = build_qdrant_client()
    embedder = build_embedding_client()
    collection = f"test_{uuid.uuid4().hex}"
    try:
        result = await sync_packages(["pyyaml"], client, embedder, collection_name=collection)
        count = (await client.get_collection(collection)).points_count
    finally:
        await client.delete_collection(collection)

    assert result["succeeded"] == ["PyYAML"]
    assert result["pruned"] == []
    assert count == 1
