"""Chapter 24: chapter 22's real tool loop and chapter 23's real
StateGraph shape, combined. `ask_agent` is a node exactly like
chapter 23's `check_stock`, it just happens to call a real model
instead of a plain dict lookup. A model call is not a special case
LangGraph needs to know about, it's one more thing a node can do.
"""

from typing import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from reliable_agents_labs.agent_loop import run_tool_loop
from reliable_agents_labs.inventory import run_check_inventory_tool
from reliable_agents_labs.json_parsing import parse_json_object
from reliable_agents_labs.models import ModelClient, build_model_client
from reliable_agents_labs.reorder_agent import CHECK_INVENTORY_TOOL, TOOL_SYSTEM_PROMPT

DEFAULT_CHECKPOINT_DB = "reorder_checkpoints.sqlite"


def build_checkpointer(path: str = DEFAULT_CHECKPOINT_DB):
    """Chapter 25: a real, durable checkpointer, one real file on disk
    instead of the process's own memory. Returns an async context
    manager, `async with build_checkpointer() as saver:`, the exact
    shape `AsyncSqliteSaver.from_conn_string` already has.
    """
    return AsyncSqliteSaver.from_conn_string(path)


DECISION_SYSTEM_PROMPT = (
    "You will be given an assistant's answer to an inventory question. "
    "Respond with a single JSON object, no markdown fences, no commentary, "
    'matching this exact shape: {"reorder": true or false}. Set reorder to '
    "true only if the answer recommends reordering at least one item."
)

# A real, if minimal, side effect: an in-memory log a caller can inspect
# after a run. Chapter 25 gives this a durable home; for now it exists
# purely to prove the conditional edge below actually chose a branch.
REORDER_LOG: list[str] = []


class ReorderWorkflowState(TypedDict):
    question: str
    answer: str
    reorder: bool
    logged: bool


def log_reorder(state: ReorderWorkflowState) -> dict:
    REORDER_LOG.append(f"{state['question']!r} -> {state['answer']!r}")
    return {"logged": True}


def _route_on_answer(state: ReorderWorkflowState) -> str:
    """A conditional edge, same shape as chapter 23's `route_on_stock`:
    reads the state so far and picks a branch by name. Routes on
    `state["reorder"]`, a real boolean, never on whether the word
    "reorder" happens to appear somewhere in the answer's own prose,
    "you do not need to reorder" contains that word too.
    """
    return "log_reorder" if state["reorder"] else "skip"


def _build_ask_agent_node(model_client: ModelClient):
    """Chapter 24's node, factored out so chapter 26's approval workflow
    can reuse it unchanged rather than duplicating it.
    """

    async def ask_agent(state: ReorderWorkflowState) -> dict:
        answer = await run_tool_loop(
            state["question"],
            model_client,
            tools=[CHECK_INVENTORY_TOOL],
            tool_fns={"check_inventory": run_check_inventory_tool},
            system=TOOL_SYSTEM_PROMPT,
        )
        # The free-text answer above is for a human to read, never for
        # this graph to branch on. A second, structured call turns it
        # into the one real fact the routing function actually needs,
        # chapter 5's own lesson: never keyword-sniff a prose answer.
        decision_result = await model_client.generate(system=DECISION_SYSTEM_PROMPT, user=answer)
        decision = parse_json_object(decision_result.text)
        return {"answer": answer, "reorder": decision["reorder"]}

    return ask_agent


def build_reorder_workflow(model_client: ModelClient | None = None, checkpointer=None):
    """Two nodes, one conditional edge. `ask_agent` wraps chapter 22's
    `run_tool_loop`, a closure over `model_client` since a node function
    only ever receives the graph's own state, never extra arguments.

    `checkpointer` is optional and defaults to `None`, an uncompiled-
    with-persistence graph, the exact shape chapter 24 already tested.
    Chapter 25 passes a real one.
    """
    model_client = model_client or build_model_client("answer_model")
    ask_agent = _build_ask_agent_node(model_client)

    builder = StateGraph(ReorderWorkflowState)
    builder.add_node("ask_agent", ask_agent)
    builder.add_node("log_reorder", log_reorder)
    builder.add_edge(START, "ask_agent")
    builder.add_conditional_edges(
        "ask_agent", _route_on_answer, {"log_reorder": "log_reorder", "skip": END}
    )
    builder.add_edge("log_reorder", END)
    return builder.compile(checkpointer=checkpointer)


class ApprovalWorkflowState(TypedDict):
    question: str
    answer: str
    reorder: bool
    approved: bool
    note: str
    logged: bool


def await_approval(state: ApprovalWorkflowState) -> dict:
    """Chapter 26: a real pause point. `interrupt()` stops this node
    exactly where it's called, persists everything needed to resume
    via whatever checkpointer the graph was compiled with, and hands
    the payload below to whoever reads the paused state. Resuming with
    `Command(resume={"approved": ..., "note": ...})` makes `interrupt()`
    return that same dict, as if it had been a normal function call.
    """
    decision = interrupt({"question": state["question"], "answer": state["answer"]})
    return {"approved": decision["approved"], "note": decision.get("note", "")}


def _route_on_approval(state: ApprovalWorkflowState) -> str:
    return "log_reorder" if state["approved"] else "skip"


def build_approval_workflow(model_client: ModelClient | None = None, checkpointer=None):
    """The same `ask_agent` node chapter 24 built, with one real pause
    point inserted before a reorder actually gets logged: a human, not
    the model, has the final say on whether to act on it. Only the
    branch that would take an irreversible action pauses; a "no reorder
    needed" answer reaches `END` without ever interrupting anyone.
    """
    model_client = model_client or build_model_client("answer_model")
    ask_agent = _build_ask_agent_node(model_client)

    builder = StateGraph(ApprovalWorkflowState)
    builder.add_node("ask_agent", ask_agent)
    builder.add_node("await_approval", await_approval)
    builder.add_node("log_reorder", log_reorder)
    builder.add_edge(START, "ask_agent")
    builder.add_conditional_edges(
        "ask_agent", _route_on_answer, {"log_reorder": "await_approval", "skip": END}
    )
    builder.add_conditional_edges(
        "await_approval", _route_on_approval, {"log_reorder": "log_reorder", "skip": END}
    )
    builder.add_edge("log_reorder", END)
    return builder.compile(checkpointer=checkpointer)
