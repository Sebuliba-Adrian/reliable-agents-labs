"""Tier 1: cost.py's own arithmetic, real pricing constants, no network,
no model. See chapter 31.
"""

from reliable_agents_labs.cost import TaskCostTracker, estimate_cost
from reliable_agents_labs.models import ModelResult


def _result(input_tokens: int, output_tokens: int) -> ModelResult:
    return ModelResult(
        text="x",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_id="fake",
        provider="fake",
    )


def test_estimate_cost_matches_the_real_published_rate():
    # 1,000,000 input tokens at $0.75/M, 1,000,000 output at $3.75/M.
    result = _result(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost(result) == 0.75 + 3.75


def test_estimate_cost_of_a_tiny_real_call():
    result = _result(input_tokens=126, output_tokens=44)
    assert round(estimate_cost(result), 6) == round(126 / 1e6 * 0.75 + 44 / 1e6 * 3.75, 6)


def test_task_cost_tracker_accumulates_across_multiple_calls():
    tracker = TaskCostTracker()
    tracker.track(_result(126, 44))
    tracker.track(_result(385, 107))

    assert tracker.call_count == 2
    assert tracker.total_cost == estimate_cost(_result(126, 44)) + estimate_cost(_result(385, 107))
