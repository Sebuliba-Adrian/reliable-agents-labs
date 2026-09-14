"""Tier 1: unit tests for _dependency_name's parsing, no network."""

from reliable_agents_labs.dependency_graph import _dependency_name


def test_strips_version_specifier():
    assert _dependency_name("starlette>=0.46.0") == "starlette"


def test_strips_environment_marker():
    assert _dependency_name('httpx<1.0,>=0.15.4; extra == "standard"') == "httpx"


def test_strips_extras_bracket():
    assert _dependency_name("httpx[http2]>=0.20.0") == "httpx"
