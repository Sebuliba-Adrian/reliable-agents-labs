"""Tier 3: integration tests, against real local services (Qdrant, Neo4j,
etc.). None exist yet, Project 2's chapter 12 ("A Real Vector Database")
introduces the first one. This placeholder is explicitly skipped, not
faked as passing, so coverage is never silently overstated.
"""

import pytest


@pytest.mark.skip(
    reason="No local services exist yet. Project 2 (Production RAG) introduces "
    "the first one (Qdrant). Replace this file's contents then, don't just "
    "delete the skip."
)
def test_placeholder():
    pass
