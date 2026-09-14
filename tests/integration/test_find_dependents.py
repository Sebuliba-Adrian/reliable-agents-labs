"""Tier 3: real PyPI metadata, real dependency relationships. See
chapter 17.
"""

from reliable_agents_labs.dependency_graph import find_dependents
from reliable_agents_labs.ingest import load_dependency_names


async def test_find_dependents_of_pydantic_among_real_dependencies():
    names = load_dependency_names()
    dependents = await find_dependents("pydantic", names)
    # A stable subset, not the exact full set: PyPI's real dependency
    # graph can change over time the same way chapter 10 already warned
    # against pinning an exact version. fastapi and anthropic depending
    # on pydantic is about as durable a fact as this dataset has.
    assert {"anthropic", "fastapi"}.issubset(dependents)
    assert "PyYAML" not in dependents
