"""Chapter 22: chapter 6's `ask_reorder_agent_with_tools` handles exactly
one tool call, then hands back one final answer. That is not a loop, it
is a fixed two-turn exchange, and it breaks the moment a real question
needs more than one round trip, verified live: a question naming two
SKUs makes the real model request two tool calls in the same turn,
chapter 6's code only ever looks at the first one, sends back a reply
missing the second tool's result, and the model responds by asking for
that missing tool call again, which chapter 6's code, already on its
one and only second turn, returns as `final.text`, an empty string,
silently, no exception at all.
"""

import json
from collections.abc import Callable

from reliable_agents_labs.models import ModelClient, ModelResult


class ToolLoopDidNotConverge(RuntimeError):
    """Raised when the model is still asking for tools after
    `max_iterations` turns. A real, explicit failure, not a silently
    wrong empty answer.
    """


def _fallback_call_dict(call) -> dict:
    """`call.raw` is the provider's own tool-call dict, always present
    for a real adapter (see `ToolCall`'s own docstring); this only runs
    for a scripted test double that never set it.
    """
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
    }


async def run_tool_loop(
    question: str,
    client: ModelClient,
    tools: list[dict],
    tool_fns: dict[str, Callable[[dict], str]],
    system: str,
    max_iterations: int = 5,
    on_result: Callable[[ModelResult], None] | None = None,
) -> str:
    """A real loop: keep calling the model and running every tool call it
    asks for, in the same turn or a later one, until it returns text
    instead of more tool calls. `tool_fns` maps a tool's name to a plain
    function that takes that call's arguments dict and returns the tool
    output as a string, one entry per tool named in `tools`.

    Unlike chapter 6's function, this handles any number of tool calls
    in a single turn, replayed as one assistant message naming all of
    them followed by one tool-result message per call, the shape every
    OpenAI-compatible provider, including Gemini's compatibility layer,
    expects back.

    `on_result` is optional and defaults to `None`, no behavior change
    for any existing caller. Chapter 31 passes one to see every real
    `ModelResult` this loop makes, not just the final answer text this
    function has only ever returned, real per-call token counts a
    multi-turn task would otherwise discard.
    """
    history: list[dict] = []
    for _ in range(max_iterations):
        result = await client.generate(system=system, user=question, tools=tools, history=history)
        if on_result is not None:
            on_result(result)
        if not result.tool_calls:
            return result.text

        assistant_calls = [call.raw or _fallback_call_dict(call) for call in result.tool_calls]
        history.append({"role": "assistant", "content": None, "tool_calls": assistant_calls})
        for call in result.tool_calls:
            output = tool_fns[call.name](call.arguments)
            history.append({"role": "tool", "tool_call_id": call.id, "content": output})

    raise ToolLoopDidNotConverge(
        f"model was still requesting tools after {max_iterations} iterations"
    )
