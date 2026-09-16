"""Tier 1: unit tests for pass_rate and retrieval_recall themselves, no
network, no model.
"""

from reliable_agents_labs.evaluation import EvalResult, pass_rate, retrieval_recall


def test_pass_rate_computes_the_fraction_passed():
    results = [
        EvalResult(question="q1", expected="a", actual=["a"], passed=True),
        EvalResult(question="q2", expected="b", actual=[], passed=False),
        EvalResult(question="q3", expected="c", actual=["c"], passed=True),
        EvalResult(question="q4", expected="d", actual=["d"], passed=True),
    ]
    assert pass_rate(results) == 0.75


def test_retrieval_recall_computes_the_fraction_actually_retrieved():
    results = [
        EvalResult(
            question="q1",
            expected="a",
            actual=["a"],
            passed=True,
            retrieved=["a", "b"],
            retrieval_hit=True,
        ),
        EvalResult(
            question="q2",
            expected="b",
            actual=[],
            passed=False,
            retrieved=["c", "d"],
            retrieval_hit=False,
        ),
        EvalResult(
            question="q3",
            expected="c",
            actual=["c"],
            passed=True,
            retrieved=["c"],
            retrieval_hit=True,
        ),
    ]
    assert retrieval_recall(results) == 2 / 3


def test_retrieval_recall_excludes_examples_with_no_expected_citation():
    results = [
        EvalResult(
            question="q1",
            expected="a",
            actual=["a"],
            passed=True,
            retrieved=["a"],
            retrieval_hit=True,
        ),
        EvalResult(
            question="q2",
            expected=None,
            actual=[],
            passed=True,
            retrieved=["z"],
            retrieval_hit=None,
        ),
    ]
    assert retrieval_recall(results) == 1.0


def test_retrieval_hit_can_differ_from_passed():
    """The whole point of this decomposition: retrieval found the right
    package, and the model still didn't cite it, a generation miss, not
    a retrieval miss. pass_rate would call this simply "failed"; only
    retrieval_recall shows retrieval was never the problem.
    """
    result = EvalResult(
        question="q1",
        expected="httpx",
        actual=[],
        passed=False,
        retrieved=["httpx", "requests"],
        retrieval_hit=True,
    )
    assert result.retrieval_hit is True
    assert result.passed is False
