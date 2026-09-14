"""Tier 1: unit test for the shape of PackageMetadata itself, no network."""

from reliable_agents_labs.pypi import PackageMetadata


def test_requires_dist_defaults_to_empty():
    metadata = PackageMetadata(name="example", version="1.0.0", summary="An example.")
    assert metadata.requires_dist == []
