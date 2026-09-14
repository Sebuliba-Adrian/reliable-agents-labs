"""Chapter 13: the one-off fetch, embed, store calls from chapters 10-12,
turned into a real, repeatable pipeline over a real list of packages.
The dataset is this project's own main dependencies, not synthetic
examples, `pyproject.toml` names exactly what "ingesting real packages"
means for this book.
"""

import tomllib

import httpx
from qdrant_client import AsyncQdrantClient

from reliable_agents_labs.models import EmbeddingClient
from reliable_agents_labs.pypi import fetch_package_metadata
from reliable_agents_labs.vector_store import COLLECTION_NAME, ensure_collection, upsert_package


def load_dependency_names(pyproject_path: str = "pyproject.toml") -> list[str]:
    """Real package names straight from this project's own
    `pyproject.toml`, stripped of version specifiers and extras, e.g.
    `"uvicorn[standard]>=0.53.0"` becomes `"uvicorn"`.
    """
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)
    names = []
    for dep in config["project"]["dependencies"]:
        name = dep.split(">=")[0].split("==")[0].split("[")[0].strip()
        names.append(name)
    return names


async def ingest_package(
    name: str,
    client: AsyncQdrantClient,
    embedder: EmbeddingClient,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    """Fetch, embed, and store one package. Returns `False` instead of
    raising when the package cannot be fetched, chapter 13's own
    exercise: a package that no longer exists, or was never real,
    should not stop the whole pipeline over one bad name.
    """
    try:
        meta = await fetch_package_metadata(name)
    except httpx.HTTPStatusError:
        return False
    vector = await embedder.embed(meta.summary)
    await upsert_package(
        client,
        package_id=hash(meta.name) % (2**31),
        name=meta.name,
        summary=meta.summary,
        vector=vector,
        collection_name=collection_name,
    )
    return True


async def ingest_all(
    names: list[str],
    client: AsyncQdrantClient,
    embedder: EmbeddingClient,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """Ingest a whole list, real PyPI packages this project depends on
    by default, continuing past any individual failure. Returns a
    summary rather than raising, so a caller can decide what "mostly
    succeeded" should mean for their own use case.
    """
    await ensure_collection(client, collection_name=collection_name)
    succeeded = []
    failed = []
    for name in names:
        if await ingest_package(name, client, embedder, collection_name=collection_name):
            succeeded.append(name)
        else:
            failed.append(name)
    return {"succeeded": succeeded, "failed": failed}
