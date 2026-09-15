"""Tier 3: real persistence, a real SQLite file on disk, scripted
model so the point being tested (checkpointing, not the model) is the
only thing that can fail. Chapter 22's own honest gap: kill the
process partway through a run and everything is gone. This test opens
a fresh checkpointer connection, with no `ainvoke` call at all, and
recovers a prior run's state anyway. See chapter 25.
"""

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.reorder_workflow import build_reorder_workflow
from tests.fakes import ScriptedModelClient

_EMPTY_STATE = {"question": "restock SKU-1029?", "answer": "", "reorder": False, "logged": False}


def _text_result(text: str) -> ModelResult:
    return ModelResult(
        text=text, input_tokens=10, output_tokens=5, model_id="scripted", provider="scripted"
    )


async def test_state_survives_a_fresh_checkpointer_connection_to_the_same_file(tmp_path):
    db_path = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "sku-1029-check"}}

    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        model = ScriptedModelClient(
            [_text_result("SKU-1029 is low, reorder it."), _text_result('{"reorder": true}')]
        )
        graph = build_reorder_workflow(model_client=model, checkpointer=saver)
        await graph.ainvoke(_EMPTY_STATE, config)

    # A brand new connection, no `ainvoke` call anywhere in this block,
    # standing in for a fresh process that never ran the agent at all.
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        graph = build_reorder_workflow(model_client=ScriptedModelClient([]), checkpointer=saver)
        state = await graph.aget_state(config)

        assert state.values["reorder"] is True
        assert state.values["logged"] is True
        assert "SKU-1029" in state.values["answer"]
