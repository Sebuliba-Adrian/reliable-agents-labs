"""Tier 3: integration test against a real, local Neo4j, running via
`docker compose up neo4j` (or CI's own service container). Chapter 8's
own placeholder named Neo4j specifically as the second real local
service this taxonomy was waiting for. See chapter 18.

Neo4j Community Edition has no equivalent to Qdrant's disposable
per-test collection (multi-database is an Enterprise feature), so this
test names its own nodes distinctly and deletes them itself, rather than
risking collision with whatever chapter 19's real ETL later stores.
"""

import uuid

from reliable_agents_labs.graph_store import (
    build_neo4j_driver,
    count_depends_on_edges,
    create_depends_on,
    find_dependents,
)


async def test_find_dependents_via_a_real_graph_traversal():
    driver = build_neo4j_driver()
    suffix = uuid.uuid4().hex[:8]
    target = f"target-{suffix}"
    dependent = f"dependent-{suffix}"
    unrelated = f"unrelated-{suffix}"
    try:
        await create_depends_on(driver, dependent, target)
        await create_depends_on(driver, unrelated, "something-else")

        result = await find_dependents(driver, target)

        assert result == [dependent]
    finally:
        async with driver.session() as session:
            await session.run(
                "MATCH (p:Package) WHERE p.name IN $names DETACH DELETE p",
                names=[target, dependent, unrelated, "something-else"],
            )
        await driver.close()


async def test_create_depends_on_is_idempotent():
    driver = build_neo4j_driver()
    suffix = uuid.uuid4().hex[:8]
    a, b = f"a-{suffix}", f"b-{suffix}"
    try:
        before = await count_depends_on_edges(driver)
        await create_depends_on(driver, a, b)
        await create_depends_on(driver, a, b)
        after = await count_depends_on_edges(driver)

        assert after - before == 1
    finally:
        async with driver.session() as session:
            await session.run(
                "MATCH (p:Package) WHERE p.name IN $names DETACH DELETE p", names=[a, b]
            )
        await driver.close()
