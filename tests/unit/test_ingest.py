"""Tier 1: unit tests for load_dependency_names and package_point_id, no
network, just parsing this project's own real pyproject.toml and pure
UUID math.
"""

import subprocess
import sys

from reliable_agents_labs.ingest import load_dependency_names, package_point_id


def test_load_dependency_names_strips_specifiers_and_extras():
    names = load_dependency_names()
    assert "fastapi" in names
    assert "uvicorn" in names
    assert not any("[" in name or ">=" in name for name in names)


def test_package_point_id_is_stable_for_the_same_name():
    assert package_point_id("httpx") == package_point_id("httpx")


def test_package_point_id_differs_for_different_names():
    assert package_point_id("httpx") != package_point_id("fastapi")


def test_package_point_id_is_stable_across_separate_processes():
    """The real test chapter 13's `hash()` bug needed and never got: a
    single process trivially agrees with itself, `hash()` is stable
    within one process too. Only two genuinely separate interpreters,
    with independently randomized hash seeds, can tell stable apart from
    coincidentally-the-same.
    """
    code = (
        "from reliable_agents_labs.ingest import package_point_id; print(package_point_id('httpx'))"
    )
    ids = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(ids) == 1
