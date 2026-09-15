"""Chapter 9: tracing the reorder agent, taught OpenTelemetry-concepts
first, not "learn Langfuse's API."

Langfuse is not a bespoke tracing service with its own wire format. Under
the hood it is an OpenTelemetry backend: constructing the Langfuse client
points a standard OTel TracerProvider at Langfuse's own OTLP endpoint.
`@observe` is a thin decorator over OTel's own span mechanism, using
OTel's context propagation to nest calls into one trace, it is not a
replacement for OTel, it is Langfuse's answer to "what do I actually call
to start a span." A raw, unmarked `tracer.start_as_current_span` call
will not show up in Langfuse, its exporter only forwards spans it
recognizes as one of its own observation types, `@observe` is what marks
a span that way.
"""

from langfuse import get_client, observe
from reliable_agents_labs.models import ModelClient
from reliable_agents_labs.reorder_agent import ask_reorder_agent_with_tools


@observe(name="ask_reorder_agent_with_tools", capture_input=False, capture_output=True)
async def ask_reorder_agent_traced(question: str, client: ModelClient | None = None) -> str:
    """The same `ask_reorder_agent_with_tools` chapters 6 and 7 already
    call, wrapped so a real call shows up in Langfuse. Nothing about
    chapter 6's function changes, tracing is added around it, not inside
    it.

    `capture_input=False` is not optional here, it is the fix for a real
    bug this chapter found: `@observe`'s default captures every argument,
    including `client`, and a real `GeminiOpenAICompatibleClient` embeds
    the live API key in its own internal state. Left on default, the
    first real call sent that key straight into the trace store. Setting
    input explicitly, one line below, records the question a human
    actually asked, and nothing else.
    """
    get_client().update_current_span(input=question)
    return await ask_reorder_agent_with_tools(question, client=client)


@observe(name="reorder_workflow")
async def run_reorder_workflow_traced(graph, initial_state, config: dict) -> dict:
    """Chapter 28: wraps a compiled workflow graph's own `ainvoke`, one
    root span per real run. `ask_agent` and `log_reorder`, inside
    `reorder_workflow.py`, are `@observe`d too, so this one traced call
    produces a real nested trace: the graph's own node structure, not a
    flat list, without needing LangChain's own callback system at all.
    A LangGraph node is just a function; the exact decorator chapter 9
    already used works on it unchanged.

    `initial_state` is a plain dict on a first call, but a real resume
    call passes a `Command` instead (chapter 26), which has no
    `question` key at all. A first version of this function called
    `.get("question")` unconditionally and crashed on exactly that
    real resume call, caught live before this ever reached print.
    """
    question = None
    if isinstance(initial_state, dict):
        question = initial_state.get("question")
    get_client().update_current_span(input={"question": question})
    return await graph.ainvoke(initial_state, config)
