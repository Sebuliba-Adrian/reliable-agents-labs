"""Tier 1: unit test for load_dependency_names, no network, just parsing
this project's own real pyproject.toml.
"""

from reliable_agents_labs.ingest import load_dependency_names


def test_load_dependency_names_strips_specifiers_and_extras():
    names = load_dependency_names()
    assert "fastapi" in names
    assert "uvicorn" in names
    assert not any("[" in name or ">=" in name for name in names)
