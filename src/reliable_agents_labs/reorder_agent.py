"""The reorder agent, Project 1 and Project 4's system.

Chapter 4 starts here: a plain model call, wrapped in application code,
with a config-driven client (see models.build_model_client), no
structured output yet (chapter 5), no tools yet (chapter 6). This module
grows chapter by chapter through the rest of Project 1, then again in
Project 4 once LangGraph turns it into a stateful workflow.
"""

from reliable_agents_labs.models import ModelClient, build_model_client

SYSTEM_PROMPT = (
    "You are a warehouse assistant for a small parts distributor. "
    "Answer questions about inventory as best you can. If you do not have "
    "real data to answer a question, say so plainly rather than guessing."
)


async def ask_reorder_agent(question: str, client: ModelClient | None = None) -> str:
    """The first version of the reorder agent: one model call, no memory,
    no tools, no structured output. Chapter 4's whole point is that this
    is not enough on its own, later chapters in Project 1 build on exactly
    this function.

    `client` is optional and defaults to the real, config-driven adapter.
    Accepting it as a parameter (instead of hardcoding the real client
    inside this function) is what makes tier-2 orchestration tests
    possible: a test can pass a ScriptedModelClient here instead, and
    never make a real network call to verify this function's own logic.
    """
    if client is None:
        client = build_model_client("answer_model")
    result = await client.generate(system=SYSTEM_PROMPT, user=question)
    return result.text
