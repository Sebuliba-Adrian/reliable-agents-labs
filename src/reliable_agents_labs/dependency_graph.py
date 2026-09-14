"""Chapter 17 opens Project 3: the real relationship a vector index
cannot see. `PackageMetadata.requires_dist` has held the real, declared
dependency list for every ingested package since chapter 10, `upsert_package`
just never stored or embedded it, only the summary. Answering "who
depends on X" needs a real graph traversal, computed here directly
against PyPI first, before chapter 18 hands it to Neo4j.
"""

import re

from reliable_agents_labs.pypi import fetch_package_metadata


def _dependency_name(requirement: str) -> str:
    """The bare package name from one requires_dist entry, e.g.
    `"starlette>=0.46.0"` becomes `"starlette"`. Strips environment
    markers and version specifiers the same way chapter 13's
    `load_dependency_names` strips them from `pyproject.toml`'s own
    dependency strings, a different string shape, the same problem.
    """
    requirement = requirement.split(";")[0].strip()
    return re.split(r"[<>=!~\[\s]", requirement, maxsplit=1)[0]


async def find_dependents(target: str, candidate_names: list[str]) -> list[str]:
    """Which of `candidate_names` actually declare `target` as a real,
    current dependency, computed directly from PyPI's own metadata, no
    graph database yet. Two packages' declared dependency relationship
    has nothing to do with how similar their summaries read, which is
    exactly why a vector index cannot answer this question, only a real
    traversal of the actual relationship can.
    """
    dependents = []
    for name in candidate_names:
        meta = await fetch_package_metadata(name)
        deps = {_dependency_name(r) for r in meta.requires_dist}
        if target in deps:
            dependents.append(meta.name)
    return dependents
