"""Tier 1: unit test for _is_core_requirement, no network."""

from reliable_agents_labs.graph_etl import _is_core_requirement


def test_plain_requirement_is_core():
    assert _is_core_requirement("starlette>=0.46.0") is True


def test_extra_gated_requirement_is_not_core():
    assert _is_core_requirement('jinja2>=3.1.5; extra == "standard"') is False
