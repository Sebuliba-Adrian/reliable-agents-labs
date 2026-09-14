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
