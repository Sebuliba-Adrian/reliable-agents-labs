"""Tier 1: unit test for pass_rate itself, no network, no model."""

from reliable_agents_labs.evaluation import EvalResult, pass_rate


def test_pass_rate_computes_the_fraction_passed():
    results = [
        EvalResult(question="q1", expected="a", actual=["a"], passed=True),
        EvalResult(question="q2", expected="b", actual=[], passed=False),
        EvalResult(question="q3", expected="c", actual=["c"], passed=True),
        EvalResult(question="q4", expected="d", actual=["d"], passed=True),
    ]
    assert pass_rate(results) == 0.75
