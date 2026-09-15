"""Chapter 20: chapter 17 showed a real gap, a vector index cannot answer
"what connects to what." Chapters 18-19 gave that question a graph, but
built it as a separate system, queried on its own. This chapter puts
both to work on one question at once: vector search finds what a
question is actually about, then a graph traversal enriches each result
with what depends on it, handing a downstream agent both facts in a
single retrieval step.
"""

from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient

from reliable_agents_labs.graph_store import find_dependents
from reliable_agents_labs.models import EmbeddingClient
from reliable_agents_labs.vector_store import COLLECTION_NAME, search_packages


async def hybrid_search(
    question: str,
    qdrant: AsyncQdrantClient,
    driver: AsyncDriver,
    embedder: EmbeddingClient,
    limit: int = 3,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Chapter 15's retrieval call, then chapter 18's traversal, once per
    result, not a choice between the two. A result with no stored
    dependents, nothing else in this project's own graph depends on it,
    gets an empty list, not an error: the graph only knows what chapter
    19's ETL actually loaded into it.
    """
    query_vector = await embedder.embed(question)
    results = await search_packages(
        qdrant, query_vector, limit=limit, collection_name=collection_name
    )
    for result in results:
        result["dependents"] = await find_dependents(driver, result["name"])
    return results


def build_hybrid_context(results: list[dict]) -> str:
    """Turn `hybrid_search`'s output into the same kind of context block
    chapter 15's `_build_context` builds, one line per package, extended
    with a dependents clause whenever the graph actually has one.
    """
    lines = []
    for r in results:
        line = f"- {r['name']}: {r['summary']}"
        if r["dependents"]:
            line += f" (depended on by: {', '.join(sorted(r['dependents']))})"
        lines.append(line)
    return "\n".join(lines)
