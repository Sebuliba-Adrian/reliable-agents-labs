"""Tier 4: provider contract test for the embedding side, same reason as
test_gemini_contract.py: a real, controlled call to catch upstream
schema/behavior changes early, not to test this codebase's own logic.
"""

import os

import pytest

from reliable_agents_labs.models import GeminiEmbeddingClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.",
)


async def test_gemini_embedding_returns_a_real_vector():
    client = GeminiEmbeddingClient()
    vector = await client.embed("FastAPI is a modern Python web framework.")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)
