"""Tier 3: the real ETL pipeline against real PyPI and real local Neo4j
together. Uses `six` and `wheel`, two small, stable, real packages
unrelated to this project's own dependency graph, purely so this test's
cleanup (deleting exactly those two nodes afterward) never touches real
project data chapter 19's own verification run already loaded. See
chapter 19.
"""

from reliable_agents_labs.graph_etl import find_transitive_dependents, load_dependency_graph
from reliable_agents_labs.graph_store import build_neo4j_driver, find_dependents


async def test_load_dependency_graph_stores_real_core_dependencies_only():
    driver = build_neo4j_driver()
    try:
        loaded = await load_dependency_graph(["six", "wheel"], driver)

        assert loaded["six"] == []
        assert "packaging" in loaded["wheel"]
        assert "wheel" in await find_dependents(driver, "packaging")
    finally:
        async with driver.session() as session:
            await session.run(
                "MATCH (p:Package) WHERE p.name IN $names DETACH DELETE p",
                names=["six", "wheel"],
            )
        await driver.close()


async def test_find_transitive_dependents_follows_more_than_one_hop():
    driver = build_neo4j_driver()
    try:
        # A synthetic two-hop chain: a real ETL call would need "b" to
        # have its own metadata fetched too, this test only checks the
        # traversal query itself, not the ETL, so it wires the chain by
        # hand with disposable node names, chapter 18's own convention.
        from reliable_agents_labs.graph_store import create_depends_on

        await create_depends_on(driver, "etl-test-a", "etl-test-b")
        await create_depends_on(driver, "etl-test-b", "etl-test-c")

        result = await find_transitive_dependents(driver, "etl-test-c")

        assert set(result) == {"etl-test-a", "etl-test-b"}
    finally:
        async with driver.session() as session:
            await session.run(
                "MATCH (p:Package) WHERE p.name IN $names DETACH DELETE p",
                names=["etl-test-a", "etl-test-b", "etl-test-c"],
            )
        await driver.close()
