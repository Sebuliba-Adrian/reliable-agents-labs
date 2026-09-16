"""Tier 2: the reorder workflow's own orchestration tests, against a
scripted model, never a real one. See chapter 24, extended by chapter
27's self-correction loop: every pass through `ask_agent` now also
scripts one `evaluate_answer` judge call.
"""

import reliable_agents_labs.reorder_workflow as reorder_workflow
from reliable_agents_labs.models import ModelResult
from tests.fakes import ScriptedModelClient

_EMPTY_STATE = {"question": "", "answer": "", "reorder": False, "logged": False}


def _text_result(text: str) -> ModelResult:
    return ModelResult(
        text=text, input_tokens=10, output_tokens=5, model_id="scripted", provider="scripted"
    )


def _faithful_verdict() -> ModelResult:
    return _text_result('{"faithful": true, "reasoning": "matches the tool output"}')


async def test_a_reorder_decision_logs_and_a_no_reorder_decision_does_not(monkeypatch):
    monkeypatch.setattr(reorder_workflow, "REORDER_LOG", [])
    model = ScriptedModelClient(
        [
            _text_result("SKU-1029 is low, you should reorder it."),
            _text_result('{"reorder": true}'),
            _faithful_verdict(),
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
            _faithful_verdict(),
        ]
    )
    graph = reorder_workflow.build_reorder_workflow(model_client=model)

    result = await graph.ainvoke({**_EMPTY_STATE, "question": "restock SKU-2040?"})

    assert result["reorder"] is False
    assert result["logged"] is False
    assert reorder_workflow.REORDER_LOG == []


async def test_an_unfaithful_answer_retries_with_the_judges_feedback_before_logging(monkeypatch):
    monkeypatch.setattr(reorder_workflow, "REORDER_LOG", [])
    model = ScriptedModelClient(
        [
            _text_result("It might be worth reordering that one."),
            _text_result('{"reorder": true}'),
            _text_result('{"faithful": false, "reasoning": "does not cite a real quantity"}'),
            _text_result("SKU-1029 has 4 units on hand, below its reorder point of 20."),
            _text_result('{"reorder": true}'),
            _faithful_verdict(),
        ]
    )
    graph = reorder_workflow.build_reorder_workflow(model_client=model)

    result = await graph.ainvoke({**_EMPTY_STATE, "question": "restock SKU-1029?"})

    assert result["attempts"] == 2
    assert result["faithful"] is True
    assert result["logged"] is True
    assert len(reorder_workflow.REORDER_LOG) == 1


async def test_exhausting_the_correction_budget_still_routes_instead_of_looping_forever(
    monkeypatch,
):
    monkeypatch.setattr(reorder_workflow, "REORDER_LOG", [])
    unfaithful_verdict = _text_result(
        '{"faithful": false, "reasoning": "still does not cite a real quantity"}'
    )
    model = ScriptedModelClient(
        [
            _text_result("It might be worth reordering that one."),
            _text_result('{"reorder": true}'),
            unfaithful_verdict,
            _text_result("Still vague, but probably reorder it."),
            _text_result('{"reorder": true}'),
            unfaithful_verdict,
        ]
    )
    graph = reorder_workflow.build_reorder_workflow(model_client=model)

    result = await graph.ainvoke({**_EMPTY_STATE, "question": "restock SKU-1029?"})

    # MAX_CORRECTION_ATTEMPTS=2: two attempts, still unfaithful, and the
    # graph routes on the last real reorder decision anyway rather than
    # retrying a third time or hanging. A real, explicit budget, chapter
    # 22's own max_iterations discipline applied to this loop instead.
    assert result["attempts"] == 2
    assert result["faithful"] is False
    assert result["logged"] is True
    assert len(reorder_workflow.REORDER_LOG) == 1
