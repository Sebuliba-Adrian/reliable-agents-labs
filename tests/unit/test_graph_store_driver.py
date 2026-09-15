"""Tier 1: unit test for build_neo4j_driver's defaults, no network, the
driver is lazy, connecting only happens on first real use.
"""

from reliable_agents_labs.graph_store import build_neo4j_driver


def test_build_neo4j_driver_returns_a_driver_without_connecting():
    driver = build_neo4j_driver(uri="bolt://localhost:7688", password="x")
    assert driver is not None
