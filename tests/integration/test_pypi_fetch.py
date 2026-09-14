"""Tier 3: integration tests. The taxonomy's own definition names "real
local services" as the example (Qdrant, Neo4j, both still chapters away).
PyPI's JSON API is not local, it is a real external system this project
does not run itself, but it fits the same tier for the same underlying
reason: this is not a scripted fake (tier 2) and not the LLM provider
(tier 4), it is a real dependency whose behavior this project's own code
cannot fully control, exactly what tier 3 exists to test. See chapter 10.
"""

from reliable_agents_labs.pypi import PackageMetadata, fetch_package_metadata


async def test_fetch_returns_real_current_data():
    metadata = await fetch_package_metadata("fastapi")
    assert isinstance(metadata, PackageMetadata)
    assert metadata.name == "fastapi"
    assert metadata.version
    assert any("starlette" in dep for dep in metadata.requires_dist)
