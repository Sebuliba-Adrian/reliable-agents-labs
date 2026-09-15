"""Tier 2: run_tool_loop's on_result hook, wired to a real TaskCostTracker,
against a scripted model, never a real one. See chapter 31.
"""

from reliable_agents_labs.agent_loop import run_tool_loop
from reliable_agents_labs.cost import TaskCostTracker, estimate_cost
from reliable_agents_labs.models import ModelResult
from tests.fakes import ScriptedModelClient

_TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {}}}


def _tool_call_result(input_tokens: int, output_tokens: int) -> ModelResult:
    from reliable_agents_labs.models import ToolCall

    return ModelResult(
        text="",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_id="scripted",
        provider="scripted",
        tool_calls=[ToolCall(id="call_0", name="lookup", arguments={"item": "a"})],
    )


def _text_result(input_tokens: int, output_tokens: int, text: str) -> ModelResult:
    return ModelResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_id="scripted",
        provider="scripted",
    )


async def test_on_result_sees_every_real_call_a_multi_turn_task_makes():
    model = ScriptedModelClient(
        [_tool_call_result(126, 44), _text_result(385, 107, "final answer")]
    )
    tracker = TaskCostTracker()

    answer = await run_tool_loop(
        "question",
        model,
        tools=[_TOOL],
        tool_fns={"lookup": lambda _args: "ok"},
        system="system",
        on_result=tracker.track,
    )

    assert answer == "final answer"
    assert tracker.call_count == 2
    expected = estimate_cost(_tool_call_result(126, 44)) + estimate_cost(
        _text_result(385, 107, "final answer")
    )
    assert tracker.total_cost == expected


async def test_no_on_result_still_works_exactly_as_before():
    model = ScriptedModelClient([_text_result(10, 5, "single answer")])

    answer = await run_tool_loop(
        "question", model, tools=[_TOOL], tool_fns={"lookup": lambda _args: "ok"}, system="system"
    )

    assert answer == "single answer"
