"""Tier 2: run_tool_loop's own orchestration tests, against a scripted
model, never a real one. See chapter 22.
"""

import pytest

from reliable_agents_labs.agent_loop import ToolLoopDidNotConverge, run_tool_loop
from reliable_agents_labs.models import ModelResult, ToolCall
from tests.fakes import ScriptedModelClient

_TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {}}}


def _tool_call_result(*calls: tuple[str, dict]) -> ModelResult:
    return ModelResult(
        text="",
        input_tokens=10,
        output_tokens=5,
        model_id="scripted",
        provider="scripted",
        tool_calls=[
            ToolCall(id=f"call_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ],
    )


def _text_result(text: str) -> ModelResult:
    return ModelResult(
        text=text, input_tokens=10, output_tokens=5, model_id="scripted", provider="scripted"
    )


async def test_tool_loop_runs_every_tool_call_in_a_single_turn():
    model = ScriptedModelClient(
        [
            _tool_call_result(("lookup", {"item": "a"}), ("lookup", {"item": "b"})),
            _text_result("both looked up"),
        ]
    )
    seen = []

    def lookup(arguments: dict) -> str:
        seen.append(arguments["item"])
        return "ok"

    answer = await run_tool_loop(
        "question", model, tools=[_TOOL], tool_fns={"lookup": lookup}, system="system"
    )

    assert answer == "both looked up"
    assert seen == ["a", "b"]


async def test_tool_loop_continues_across_more_than_one_turn():
    model = ScriptedModelClient(
        [
            _tool_call_result(("lookup", {"item": "a"})),
            _tool_call_result(("lookup", {"item": "b"})),
            _text_result("done after two rounds"),
        ]
    )
    seen = []

    def lookup(arguments: dict) -> str:
        seen.append(arguments["item"])
        return "ok"

    answer = await run_tool_loop(
        "question", model, tools=[_TOOL], tool_fns={"lookup": lookup}, system="system"
    )

    assert answer == "done after two rounds"
    assert seen == ["a", "b"]


async def test_tool_loop_raises_a_real_error_instead_of_returning_silently_empty():
    # The model never stops asking for tools, chapter 6's own bug turned
    # this into a silent empty string instead of a real, checkable error.
    model = ScriptedModelClient([_tool_call_result(("lookup", {"item": "x"})) for _ in range(5)])

    with pytest.raises(ToolLoopDidNotConverge):
        await run_tool_loop(
            "question",
            model,
            tools=[_TOOL],
            tool_fns={"lookup": lambda _args: "ok"},
            system="system",
            max_iterations=3,
        )
