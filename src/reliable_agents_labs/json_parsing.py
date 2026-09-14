"""Shared helper, first written for chapter 5's reorder agent, reused by
chapter 15's RAG agent: parsing a model's text reply as JSON when it was
asked to return exactly that. Not provider-specific, not agent-specific,
just what every structured-output prompt in this book needs once a model
wraps its JSON in markdown fences anyway.
"""

import json


def parse_json_object(text: str) -> dict:
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
