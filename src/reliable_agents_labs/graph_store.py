"""Chapter 18: the same dependency relationship chapter 17 answered with
a plain Python loop over PyPI calls, now given a real home built for
exactly this shape of question. A node is a package, a `DEPENDS_ON`
relationship is a declared dependency edge, and `find_dependents` below
answers chapter 17's whole question in one query instead of one API
call per candidate package.
"""

import os

from neo4j import AsyncDriver, AsyncGraphDatabase


def build_neo4j_driver(
    uri: str | None = None, user: str = "neo4j", password: str | None = None
) -> AsyncDriver:
    uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7688")
    password = password or os.environ.get("NEO4J_PASSWORD", "reliable-agents-labs")
    return AsyncGraphDatabase.driver(uri, auth=(user, password))


async def create_package_node(driver: AsyncDriver, name: str) -> None:
    """`MERGE`, not `CREATE`: calling this twice for the same package
    must not produce two nodes. Idempotent for the same reason chapter
    14 made ingestion idempotent, running a script again should never
    depend on remembering whether it already ran.
    """
    async with driver.session() as session:
        await session.run("MERGE (p:Package {name: $name})", name=name)


async def create_depends_on(driver: AsyncDriver, dependent: str, dependency: str) -> None:
    """A real, declared edge: `dependent` requires `dependency`. Both
    endpoints are `MERGE`d too, so this can run before either node has
    been created explicitly, without duplicating one that already has.
    """
    async with driver.session() as session:
        await session.run(
            """
            MERGE (a:Package {name: $dependent})
            MERGE (b:Package {name: $dependency})
            MERGE (a)-[:DEPENDS_ON]->(b)
            """,
            dependent=dependent,
            dependency=dependency,
        )


async def count_depends_on_edges(driver: AsyncDriver) -> int:
    """A small verification helper, not part of the main mechanism:
    used to prove `create_depends_on` is actually idempotent, by
    counting real relationships before and after running the same edge
    twice.
    """
    async with driver.session() as session:
        result = await session.run(
            "MATCH (:Package)-[r:DEPENDS_ON]->(:Package) RETURN count(r) AS c"
        )
        record = await result.single()
        return record["c"]


async def find_dependents(driver: AsyncDriver, target: str) -> list[str]:
    """Chapter 17's question, answered by an actual graph traversal: one
    query, not one PyPI call per candidate package. This is the same
    relationship, `DEPENDS_ON`, that chapter 17 computed with a Python
    loop, now stored so the graph itself can be asked directly.
    """
    async with driver.session() as session:
        result = await session.run(
            "MATCH (a:Package)-[:DEPENDS_ON]->(b:Package {name: $target}) RETURN a.name AS name",
            target=target,
        )
        return [record["name"] async for record in result]
