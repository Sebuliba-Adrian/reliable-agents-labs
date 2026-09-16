"""Chapter 15: retrieval as a step before generation, not a tool the
model calls (contrast chapter 6's `ask_reorder_agent_with_tools`, where
the model itself decided whether to call a tool). Here retrieval always
runs first, its results become part of the prompt, and the model never
sees the question without also seeing what was actually retrieved for
it. "Grounded" means the model's context literally contains the
retrieved text. "Cited" means the answer names which package it came
from, checkable against what was actually retrieved, not just asserted.
"""

from pydantic import BaseModel

from reliable_agents_labs.json_parsing import parse_json_object
from reliable_agents_labs.models import (
    EmbeddingClient,
    ModelClient,
    build_embedding_client,
    build_model_client,
)
from reliable_agents_labs.vector_store import (
    AsyncQdrantClient,
    build_qdrant_client,
    search_packages,
)


class RagAnswer(BaseModel):
    answer: str
    cited_packages: list[str] = []


RAG_SYSTEM_PROMPT = (
    "You are a package dependency assistant. Answer the question using "
    "ONLY the package information provided below, never your own general "
    "knowledge of these packages. Respond with a single JSON object, no "
    'markdown fences, no commentary: {"answer": "your response text", '
    '"cited_packages": ["exact package names you actually used"]}. '
    "cited_packages must contain only names that appear in the provided "
    "context. If the context does not contain enough information to "
    "answer, say so plainly in answer and leave cited_packages empty."
)


def _build_context(results: list[dict]) -> str:
    return "\n".join(f"- {r['name']}: {r['summary']}" for r in results)


async def ask_rag_agent(
    question: str,
    qdrant: AsyncQdrantClient | None = None,
    embedder: EmbeddingClient | None = None,
    model_client: ModelClient | None = None,
    limit: int = 3,
    on_retrieval=None,
) -> RagAnswer:
    """Embed the question, retrieve the closest packages, put exactly
    that retrieved text in the prompt, then ask for a cited answer.
    Unlike chapter 6's tool-calling agent, there is no decision point,
    every question retrieves, whether or not the result turns out to be
    relevant, the model itself is trusted to say so when it is not.

    Each dependency defaults to the real, config-driven one, same reason
    as every agent function since chapter 4: a test hands this scripted
    or fake versions of all three instead, and never touches the network.

    `on_retrieval`, defaulting to `None`, sees the raw retrieved results
    before they are flattened into prompt text: the final `RagAnswer`
    only ever carries what the model chose to cite, not what retrieval
    actually found, and telling a retrieval miss apart from a
    generation miss needs both.
    """
    qdrant = qdrant or build_qdrant_client()
    embedder = embedder or build_embedding_client()
    model_client = model_client or build_model_client("answer_model")

    query_vector = await embedder.embed(question)
    results = await search_packages(qdrant, query_vector, limit=limit)
    if on_retrieval is not None:
        on_retrieval(results)
    context = _build_context(results)

    user_prompt = f"Package information:\n{context}\n\nQuestion: {question}"
    result = await model_client.generate(system=RAG_SYSTEM_PROMPT, user=user_prompt)
    payload = parse_json_object(result.text)
    return RagAnswer.model_validate(payload)
