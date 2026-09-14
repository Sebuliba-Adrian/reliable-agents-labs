"""Chapters 13-14: the one-off fetch, embed, store calls from chapters
10-12, turned into a real, repeatable pipeline over a real list of
packages, and then made safe to actually repeat. The dataset is this
project's own main dependencies, not synthetic examples,
`pyproject.toml` names exactly what "ingesting real packages" means for
this book.
"""

import tomllib
import uuid

import httpx
from qdrant_client import AsyncQdrantClient

from reliable_agents_labs.models import EmbeddingClient
from reliable_agents_labs.pypi import fetch_package_metadata
from reliable_agents_labs.vector_store import COLLECTION_NAME, ensure_collection, upsert_package

# A fixed namespace for this project's own deterministic package ids
# (chapter 14). Any valid UUID works as a namespace, what matters is that
# it never changes, changing it would silently orphan every id already
# stored under the old one.
PACKAGE_ID_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


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


def package_point_id(name: str) -> str:
    """Chapter 14's fix. `uuid.uuid5` derives the same UUID from the same
    input every time, in every process, unlike Python's built-in
    `hash()`, which chapter 13 verified is randomized per process for
    strings. Same package name in, same id out, forever, which is what
    lets `upsert` actually overwrite instead of duplicate.
    """
    return str(uuid.uuid5(PACKAGE_ID_NAMESPACE, name))


async def ingest_package(
    name: str,
    client: AsyncQdrantClient,
    embedder: EmbeddingClient,
    collection_name: str = COLLECTION_NAME,
) -> str | None:
    """Fetch, embed, and store one package. Returns the package's real,
    PyPI-canonical name on success (which is not always the same string
    as `name`, see chapter 16), or `None` instead of raising when the
    package cannot be fetched, chapter 13's own exercise: a package that
    no longer exists, or was never real, should not stop the whole
    pipeline over one bad name.
    """
    try:
        meta = await fetch_package_metadata(name)
    except httpx.HTTPStatusError:
        return None
    vector = await embedder.embed(meta.summary)
    await upsert_package(
        client,
        package_id=package_point_id(meta.name),
        name=meta.name,
        summary=meta.summary,
        vector=vector,
        collection_name=collection_name,
    )
    return meta.name


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

    `succeeded` holds each package's real, canonical name, exactly what
    got stored, not the raw input string, see chapter 16 for why that
    distinction matters.
    """
    await ensure_collection(client, collection_name=collection_name)
    succeeded = []
    failed = []
    for name in names:
        canonical_name = await ingest_package(
            name, client, embedder, collection_name=collection_name
        )
        if canonical_name is not None:
            succeeded.append(canonical_name)
        else:
            failed.append(name)
    return {"succeeded": succeeded, "failed": failed}


async def prune_stale(
    current_names: list[str],
    client: AsyncQdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> list[str]:
    """The other half of idempotent ingestion: a package that stops being
    a dependency (removed from `pyproject.toml`) should stop being
    retrieved, not linger forever because nothing ever told the vector
    store it left. Deletes any stored point whose id does not match one
    of `current_names`' own deterministic ids, returns the names removed.
    """
    current_ids = {package_point_id(name) for name in current_names}
    stale_ids = []
    stale_names = []
    offset = None
    while True:
        records, offset = await client.scroll(
            collection_name=collection_name, limit=100, offset=offset, with_payload=True
        )
        for record in records:
            if str(record.id) not in current_ids:
                stale_ids.append(record.id)
                stale_names.append(record.payload.get("name", str(record.id)))
        if offset is None:
            break
    if stale_ids:
        await client.delete(collection_name=collection_name, points_selector=stale_ids)
    return stale_names


async def sync_packages(
    names: list[str],
    client: AsyncQdrantClient,
    embedder: EmbeddingClient,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """Ingest the current list and prune whatever no longer belongs, the
    two halves idempotent ingestion actually needs: running this twice
    in a row, or after `pyproject.toml` changes, leaves the collection
    matching `names` exactly, not "matching plus whatever was there
    before."
    """
    result = await ingest_all(names, client, embedder, collection_name=collection_name)
    # Prune against the *canonical* names actually stored, not the raw
    # input list. A dependency spelled differently than PyPI's own
    # canonical form (chapter 16: "pyyaml" vs. the real "PyYAML") would
    # otherwise compute a "keep" id that never matches what ingest_all
    # just stored, and prune_stale would delete it on every single sync.
    result["pruned"] = await prune_stale(
        result["succeeded"], client, collection_name=collection_name
    )
    return result
