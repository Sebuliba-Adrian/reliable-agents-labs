"""The reorder agent, Project 1 and Project 4's system.

Chapter 4 starts here: a plain model call, wrapped in application code,
with a config-driven client (see models.build_model_client), no
structured output yet (chapter 5), no tools yet (chapter 6). This module
grows chapter by chapter through the rest of Project 1, then again in
Project 4 once LangGraph turns it into a stateful workflow.
"""

import json

from pydantic import BaseModel

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


class ReorderAnswer(BaseModel):
    """Chapter 5's whole point: a shape calling code can act on, instead of
    prose it would have to guess at. `can_answer` is the field that
    matters most, it is what lets later chapters (6 onward) decide whether
    to hand the question to a tool instead of trusting the model's own
    text.
    """

    can_answer: bool
    answer: str


STRUCTURED_SYSTEM_PROMPT = (
    "You are a warehouse assistant for a small parts distributor. "
    "Respond with a single JSON object, and nothing else, no markdown "
    'fences, no commentary, matching this exact shape: {"can_answer": '
    'true or false, "answer": "your response text"}. Set can_answer to '
    "false when you do not have real data to answer the question; still "
    "fill in answer with an honest explanation of what you would need."
)


def _parse_json_object(text: str) -> dict:
    """Models occasionally wrap JSON in markdown fences even when told not
    to. Strip those defensively before parsing, rather than letting a
    cosmetic wrapper turn into a hard failure.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


async def ask_reorder_agent_structured(
    question: str, client: ModelClient | None = None
) -> ReorderAnswer:
    """Chapter 5's version: the same question, but the model must return a
    typed shape instead of free text. `can_answer` is what makes this
    useful, code can branch on it directly, no keyword-sniffing a prose
    answer for phrases like "I do not have access".

    Raises `json.JSONDecodeError` if the model's text is not valid JSON,
    or `pydantic.ValidationError` if it is valid JSON but does not match
    `ReorderAnswer`'s shape. Chapter 5's exercise asks you to trigger both.
    """
    if client is None:
        client = build_model_client("answer_model")
    result = await client.generate(system=STRUCTURED_SYSTEM_PROMPT, user=question)
    payload = _parse_json_object(result.text)
    return ReorderAnswer.model_validate(payload)
