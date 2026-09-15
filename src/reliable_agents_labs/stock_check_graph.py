"""Chapter 23: LangGraph's mental model, applied to the smallest real
version of this book's own domain question. Three ideas, and nothing
else: a node is a plain function that returns a partial state update, an
edge decides what runs next (an edge to a fixed target, or a conditional
edge that picks one of several based on the state so far), and `compile`
turns a builder into something that actually runs. No model call yet,
no tool loop yet, chapter 24 extends this exact shape into the real
reorder workflow.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from reliable_agents_labs.inventory import check_inventory


class ReorderState(TypedDict):
    sku: str
    quantity: int
    reorder_point: int
    decision: str


def check_stock(state: ReorderState) -> dict:
    """A node: a plain function, state in, a partial update out.
    LangGraph merges whatever this returns into the running state, it
    never has to return the whole thing back.
    """
    record = check_inventory(state["sku"])
    if record is None:
        return {"quantity": 0, "reorder_point": 0}
    return {"quantity": record.quantity, "reorder_point": record.reorder_point}


def route_on_stock(state: ReorderState) -> str:
    """A conditional edge's routing function: reads the state so far,
    returns the name of whichever branch should run next. Nothing about
    this function touches the graph itself, it's a plain decision.
    """
    if state["quantity"] < state["reorder_point"]:
        return "reorder"
    return "no_reorder"


def reorder(state: ReorderState) -> dict:
    return {
        "decision": (
            f"reorder {state['sku']}: {state['quantity']} on hand, "
            f"below reorder point {state['reorder_point']}"
        )
    }


def no_reorder(state: ReorderState) -> dict:
    return {"decision": f"no reorder needed for {state['sku']}"}


def build_stock_check_graph():
    """Three nodes, one fixed edge, one conditional edge. `compile()` is
    what turns this builder into something with `.invoke()`; the builder
    itself never runs anything.
    """
    builder = StateGraph(ReorderState)
    builder.add_node("check_stock", check_stock)
    builder.add_node("reorder", reorder)
    builder.add_node("no_reorder", no_reorder)
    builder.add_edge(START, "check_stock")
    builder.add_conditional_edges(
        "check_stock", route_on_stock, {"reorder": "reorder", "no_reorder": "no_reorder"}
    )
    builder.add_edge("reorder", END)
    builder.add_edge("no_reorder", END)
    return builder.compile()
