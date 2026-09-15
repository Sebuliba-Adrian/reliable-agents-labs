"""Chapter 21: chapter 17's question, chapter 20's retrieval, chapter 15's
answer shape. `ask_rag_agent` retrieves then generates from a plain
vector-search context; this agent does the same thing with
`hybrid_search`'s enriched context instead, the only change needed to
close the gap chapter 17 opened.
"""

from reliable_agents_labs.graph_store import AsyncDriver, build_neo4j_driver
from reliable_agents_labs.hybrid_retrieval import build_hybrid_context, hybrid_search
from reliable_agents_labs.json_parsing import parse_json_object
from reliable_agents_labs.models import (
    EmbeddingClient,
    ModelClient,
    build_embedding_client,
    build_model_client,
)
from reliable_agents_labs.rag_agent import RAG_SYSTEM_PROMPT, RagAnswer
from reliable_agents_labs.vector_store import (
    COLLECTION_NAME,
    AsyncQdrantClient,
    build_qdrant_client,
)


async def ask_graph_rag_agent(
    question: str,
    qdrant: AsyncQdrantClient | None = None,
    driver: AsyncDriver | None = None,
    embedder: EmbeddingClient | None = None,
    model_client: ModelClient | None = None,
    limit: int = 3,
    collection_name: str = COLLECTION_NAME,
    on_retrieval=None,
) -> RagAnswer:
    """Same prompt, same answer shape, same citation contract as chapter
    15's `ask_rag_agent`. The only change is the context it builds from:
    `hybrid_search` instead of a plain vector-only `search_packages`
    call, so a structural fact, what depends on what, can now sit in the
    same context a semantic one always could.

    `on_retrieval` is the same seam chapter 15's `ask_rag_agent` added,
    reused here rather than reinvented: whatever `hybrid_search` actually
    returned, before `build_hybrid_context` flattens it into prompt text.
    """
    qdrant = qdrant or build_qdrant_client()
    driver = driver or build_neo4j_driver()
    embedder = embedder or build_embedding_client()
    model_client = model_client or build_model_client("answer_model")

    results = await hybrid_search(
        question, qdrant, driver, embedder, limit=limit, collection_name=collection_name
    )
    if on_retrieval is not None:
        on_retrieval(results)
    context = build_hybrid_context(results)

    user_prompt = f"Package information:\n{context}\n\nQuestion: {question}"
    result = await model_client.generate(system=RAG_SYSTEM_PROMPT, user=user_prompt)
    payload = parse_json_object(result.text)
    return RagAnswer.model_validate(payload)
