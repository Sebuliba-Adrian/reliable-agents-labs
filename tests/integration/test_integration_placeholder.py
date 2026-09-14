"""Tier 3: integration tests against real *local* services specifically
(Qdrant, Neo4j, both still Docker containers this project would run
itself). None exist yet, Project 2's chapter 12 ("A Real Vector
Database") introduces the first one. `test_pypi.py`, in this same
directory since chapter 10, is also tier 3 but against a real external
service this project does not run, see its own docstring for why that
still belongs here. This placeholder is explicitly skipped, not faked as
passing, so coverage is never silently overstated for the local-service
case specifically.
"""

import pytest


@pytest.mark.skip(
    reason="No local services exist yet. Project 2's chapter 12 introduces "
    "the first one (Qdrant). Replace this file's contents then, don't just "
    "delete the skip."
)
def test_placeholder():
    pass
