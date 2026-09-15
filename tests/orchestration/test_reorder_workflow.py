"""Tier 2: the reorder workflow's own orchestration tests, against a
scripted model, never a real one. See chapter 24.
"""

import reliable_agents_labs.reorder_workflow as reorder_workflow
from reliable_agents_labs.models import ModelResult
from tests.fakes import ScriptedModelClient

_EMPTY_STATE = {"question": "", "answer": "", "reorder": False, "logged": False}


def _text_result(text: str) -> ModelResult:
    return ModelResult(
        text=text, input_tokens=10, output_tokens=5, model_id="scripted", provider="scripted"
    )


async def test_a_reorder_decision_logs_and_a_no_reorder_decision_does_not(monkeypatch):
    monkeypatch.setattr(reorder_workflow, "REORDER_LOG", [])
    model = ScriptedModelClient(
        [
            _text_result("SKU-1029 is low, you should reorder it."),
            _text_result('{"reorder": true}'),
        ]
    )
    graph = reorder_workflow.build_reorder_workflow(model_client=model)

    result = await graph.ainvoke({**_EMPTY_STATE, "question": "restock SKU-1029?"})

    assert result["reorder"] is True
    assert result["logged"] is True
    assert len(reorder_workflow.REORDER_LOG) == 1


async def test_a_negative_prose_mention_of_reorder_does_not_falsely_log(monkeypatch):
    # The exact real bug this chapter found live: the word "reorder"
    # appears in a negative sentence too. The structured decision call
    # is what keeps the routing correct, not string matching the prose.
    monkeypatch.setattr(reorder_workflow, "REORDER_LOG", [])
    model = ScriptedModelClient(
        [
            _text_result("SKU-2040 has plenty of stock, you do not need to reorder it."),
            _text_result('{"reorder": false}'),
        ]
    )
    graph = reorder_workflow.build_reorder_workflow(model_client=model)

    result = await graph.ainvoke({**_EMPTY_STATE, "question": "restock SKU-2040?"})

    assert result["reorder"] is False
    assert result["logged"] is False
    assert reorder_workflow.REORDER_LOG == []
