"""Tier 2: the approval workflow's own orchestration tests, a scripted
model and an in-memory checkpointer (`interrupt` requires one, but the
mechanism under test here is the pause/resume itself, not durability,
chapter 25 already covers real file persistence). See chapter 26.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import reliable_agents_labs.reorder_workflow as reorder_workflow
from reliable_agents_labs.models import ModelResult
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


async def test_a_reorder_proposal_pauses_before_logging_anything(monkeypatch):
    monkeypatch.setattr(reorder_workflow, "REORDER_LOG", [])
    model = ScriptedModelClient(
        [_text_result("SKU-1029 is low, reorder it."), _text_result('{"reorder": true}')]
    )
    graph = reorder_workflow.build_approval_workflow(
        model_client=model, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "test-pause"}}

    paused = await graph.ainvoke(_EMPTY_STATE, config)

    assert "__interrupt__" in paused
    assert reorder_workflow.REORDER_LOG == []


async def test_approval_resumes_and_logs(monkeypatch):
    monkeypatch.setattr(reorder_workflow, "REORDER_LOG", [])
    model = ScriptedModelClient(
        [_text_result("SKU-1029 is low, reorder it."), _text_result('{"reorder": true}')]
    )
    graph = reorder_workflow.build_approval_workflow(
        model_client=model, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "test-approve"}}
    await graph.ainvoke(_EMPTY_STATE, config)

    result = await graph.ainvoke(Command(resume={"approved": True, "note": "go ahead"}), config)

    assert result["logged"] is True
    assert result["note"] == "go ahead"
    assert len(reorder_workflow.REORDER_LOG) == 1


async def test_rejection_resumes_without_logging(monkeypatch):
    monkeypatch.setattr(reorder_workflow, "REORDER_LOG", [])
    model = ScriptedModelClient(
        [_text_result("SKU-1029 is low, reorder it."), _text_result('{"reorder": true}')]
    )
    graph = reorder_workflow.build_approval_workflow(
        model_client=model, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "test-reject"}}
    await graph.ainvoke(_EMPTY_STATE, config)

    result = await graph.ainvoke(
        Command(resume={"approved": False, "note": "budget freeze"}), config
    )

    assert result["logged"] is False
    assert reorder_workflow.REORDER_LOG == []
