"""Tier 2: the reorder agent's tool-calling path, against a scripted
model, never a real one. See chapter 6.
"""

import pytest

from reliable_agents_labs.models import ModelResult, ToolCall
from reliable_agents_labs.reorder_agent import ask_reorder_agent_with_tools
from tests.fakes import ScriptedModelClient


def _result(text: str = "", tool_calls: list[ToolCall] | None = None) -> ModelResult:
    return ModelResult(
        text=text,
        input_tokens=20,
        output_tokens=12,
        model_id="scripted",
        provider="scripted",
        tool_calls=tool_calls or [],
    )


async def test_tool_call_grounds_the_answer():
    fake = ScriptedModelClient(
        [
            _result(
                tool_calls=[
                    ToolCall(id="call_1", name="check_inventory", arguments={"sku": "SKU-1029"})
                ]
            ),
            _result(text="SKU-1029 has 4 units in stock, below its reorder point of 20."),
        ]
    )
    answer = await ask_reorder_agent_with_tools(
        "How many units of SKU-1029 are in stock?", client=fake
    )
    assert answer == "SKU-1029 has 4 units in stock, below its reorder point of 20."


async def test_skips_tool_when_unneeded():
    fake = ScriptedModelClient([_result(text="A safety stock buffer covers demand spikes.")])
    answer = await ask_reorder_agent_with_tools("What is safety stock, in general?", client=fake)
    assert answer == "A safety stock buffer covers demand spikes."


async def test_unexpected_tool_name_raises():
    fake = ScriptedModelClient(
        [_result(tool_calls=[ToolCall(id="call_1", name="delete_warehouse", arguments={})])]
    )
    with pytest.raises(ValueError, match="Unexpected tool call"):
        await ask_reorder_agent_with_tools("Anything", client=fake)


async def test_missing_sku_argument_raises():
    """Chapter 8's exercise: nothing stops a model from calling the right
    tool with the wrong argument shape. This is a distinct failure mode
    from `test_unexpected_tool_name_raises`, the tool name is correct
    this time, the payload is not.
    """
    fake = ScriptedModelClient(
        [_result(tool_calls=[ToolCall(id="call_1", name="check_inventory", arguments={})])]
    )
    with pytest.raises(ValueError, match="missing required argument 'sku'"):
        await ask_reorder_agent_with_tools("How many units are in stock?", client=fake)


async def test_unknown_sku_gets_final_answer():
    """The tool itself can come back empty, chapter 6's exercise: this is
    a distinct failure mode from `ask_reorder_agent_structured`'s parse
    errors, the model asked correctly, the data just isn't there.
    """
    fake = ScriptedModelClient(
        [
            _result(
                tool_calls=[
                    ToolCall(id="call_1", name="check_inventory", arguments={"sku": "SKU-0000"})
                ]
            ),
            _result(text="I don't have a record for SKU-0000 in inventory."),
        ]
    )
    answer = await ask_reorder_agent_with_tools("How many SKU-0000 do we have?", client=fake)
    assert "don't have a record" in answer
