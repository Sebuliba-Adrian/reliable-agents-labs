"""Tier 2: run_reorder_workflow_traced's own orchestration tests,
scripted model, never a real network call. Whether a trace also
reaches Langfuse depends on LANGFUSE_PUBLIC_KEY being set, same
guarantee as chapter 9: without one, @observe's calls are harmless
no-ops. See chapter 28.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.observability import run_reorder_workflow_traced
from reliable_agents_labs.reorder_workflow import build_approval_workflow
from tests.fakes import ScriptedModelClient

_EMPTY_STATE = {
    "question": "restock SKU-1029?",
    "answer": "",
    "reorder": False,
    "approved": False,
    "note": "",
    "logged": False,
}


def _text_result(text: str) -> ModelResult:
    return ModelResult(
        text=text, input_tokens=10, output_tokens=5, model_id="scripted", provider="scripted"
    )


async def test_traced_run_returns_the_same_result_as_an_untraced_one():
    model = ScriptedModelClient(
        [_text_result("SKU-1029 is fine."), _text_result('{"reorder": false}')]
    )
    graph = build_approval_workflow(model_client=model, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "traced-test-1"}}

    result = await run_reorder_workflow_traced(graph, _EMPTY_STATE, config)

    assert result["reorder"] is False


async def test_a_resume_call_does_not_crash_the_tracer():
    # The exact real bug this chapter found live: the first version
    # called initial_state.get("question") unconditionally, and a
    # resume call passes a Command, not a dict, with no such method.
    model = ScriptedModelClient(
        [
            _text_result("SKU-1029 is low, reorder it."),
            _text_result('{"reorder": true}'),
        ]
    )
    graph = build_approval_workflow(model_client=model, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "traced-test-2"}}
    await run_reorder_workflow_traced(graph, _EMPTY_STATE, config)

    resumed = await run_reorder_workflow_traced(
        graph, Command(resume={"approved": True, "note": "ok"}), config
    )

    assert resumed["logged"] is True
