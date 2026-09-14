"""Chapter 12: the same cosine-similarity comparison chapter 11 wrote by
hand, now handed to Qdrant, a real vector database, running as its own
container (see compose.yaml). Nothing about the underlying math changes,
`Distance.COSINE` below is `cosine_similarity` from chapter 11, computed
at a speed and scale a plain Python loop does not have.
"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "packages"

# gemini-embedding-001's real output size, verified in chapter 11
# (`len(response.data[0].embedding)`), not a guessed round number.
VECTOR_SIZE = 3072


def build_qdrant_client(url: str = "http://localhost:6333") -> AsyncQdrantClient:
    return AsyncQdrantClient(url=url)


async def ensure_collection(
    client: AsyncQdrantClient, collection_name: str = COLLECTION_NAME
) -> None:
    """Idempotent on purpose, chapter 14's whole subject. Safe to call
    every time the app starts, not just once by hand.
    """
    if not await client.collection_exists(collection_name):
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


async def upsert_package(
    client: AsyncQdrantClient,
    package_id: int | str,
    name: str,
    summary: str,
    vector: list[float],
    collection_name: str = COLLECTION_NAME,
) -> None:
    await client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=package_id, vector=vector, payload={"name": name, "summary": summary})
        ],
    )


async def search_packages(
    client: AsyncQdrantClient,
    query_vector: list[float],
    limit: int = 3,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    response = await client.query_points(
        collection_name=collection_name, query=query_vector, limit=limit
    )
    return [
        {
            "name": point.payload["name"],
            "summary": point.payload.get("summary", ""),
            "score": point.score,
        }
        for point in response.points
    ]
