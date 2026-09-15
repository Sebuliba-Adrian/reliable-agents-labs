"""Chapter 19: chapter 18 wrote five edges by hand. This is the real
pipeline: extract every one of this project's dependencies' own real,
declared dependencies from PyPI, and load the whole graph at once,
mirroring chapter 13's ingestion pipeline, ETL for a graph instead of a
vector store.
"""

from neo4j import AsyncDriver

from reliable_agents_labs.dependency_graph import _dependency_name
from reliable_agents_labs.graph_store import create_depends_on
from reliable_agents_labs.pypi import fetch_package_metadata


def _is_core_requirement(requirement: str) -> bool:
    """`requires_dist` mixes two different kinds of fact: dependencies a
    package always needs, and dependencies gated behind an optional
    extra (e.g. `fastapi[standard]`'s `jinja2`, only installed if a
    caller asks for that extra). A `DEPENDS_ON` edge that could not tell
    these apart would answer "what installs under this one extra" as if
    it were "what this package always requires", precise enough for a
    single check like chapter 17's, not precise enough for a graph meant
    to answer many different questions.
    """
    return "extra ==" not in requirement


async def load_dependency_graph(names: list[str], driver: AsyncDriver) -> dict[str, list[str]]:
    """For every package in `names`, fetch its real, current metadata and
    store a `DEPENDS_ON` edge to each of its own real core dependencies,
    the ones it always needs, not ones behind an optional extra.
    `create_depends_on`'s `MERGE`s make this safe to call again after
    `pyproject.toml` changes, the same idempotency chapter 14 built for
    ingestion, already built into chapter 18's own writes.

    Returns what was loaded: each package's real name mapped to its
    direct core dependency list, for a caller to inspect or verify
    against, the same shape chapter 13's `ingest_all` returns a summary
    in.
    """
    loaded = {}
    for name in names:
        meta = await fetch_package_metadata(name)
        core_requirements = [r for r in meta.requires_dist if _is_core_requirement(r)]
        deps = sorted({_dependency_name(r) for r in core_requirements})
        for dep in deps:
            await create_depends_on(driver, meta.name, dep)
        loaded[meta.name] = deps
    return loaded


async def find_transitive_dependents(driver: AsyncDriver, target: str) -> list[str]:
    """Every package that depends on `target`, directly or through any
    chain of other dependencies, one Cypher query, only possible because
    chapter 18's edges and this chapter's ETL already loaded the whole
    graph, not just the one target a hand-written script happened to ask
    about.
    """
    async with driver.session() as session:
        result = await session.run(
            "MATCH (a:Package)-[:DEPENDS_ON*1..]->(b:Package {name: $target}) "
            "RETURN DISTINCT a.name AS name",
            target=target,
        )
        return [record["name"] async for record in result]
