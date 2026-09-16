"""Tier 5: `judge_faithfulness` against the real, live model, on two
deliberately constructed cases, one genuinely faithful, one genuinely
not. A judge that always says "faithful" would pass a scripted test
just as easily as a real one; only a live run against a real
counterexample proves it actually discriminates, chapter 13's own
"test the failure path deliberately" convention, applied here.
"""

from reliable_agents_labs.evaluation import judge_faithfulness
from reliable_agents_labs.models import build_model_client


async def test_judge_accepts_a_genuinely_faithful_answer():
    model = build_model_client("answer_model")

    result = await judge_faithfulness(
        question="What does httpx do?",
        context="httpx: a fully featured HTTP client for Python, with sync and async support.",
        answer="httpx is a Python HTTP client that supports both sync and async use.",
        model_client=model,
    )

    assert result.faithful is True, result.reasoning


async def test_judge_rejects_a_genuinely_unfaithful_answer():
    model = build_model_client("answer_model")

    result = await judge_faithfulness(
        question="What does httpx do?",
        context="httpx: a fully featured HTTP client for Python, with sync and async support.",
        answer=(
            "httpx is a full web application framework for building and "
            "deploying REST APIs and web servers."
        ),
        model_client=model,
    )

    assert result.faithful is False, result.reasoning
