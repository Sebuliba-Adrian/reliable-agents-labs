"""Tier 4: provider contract tests. A real, controlled call to the actual
model API, to catch upstream schema/behavior changes early, not to test
this codebase's own logic (that's tiers 1-2).

This is the one tier that needs a real credential. Phase 3's gate in the
Book 2 roadmap is explicit: this test must actually PASS against the real
Gemini key before Project 1's first "Build it" chapter is written, a
skip is not a substitute for that, it just keeps the test suite from
failing outright for contributors who haven't configured a key yet.
"""

import os

import pytest

from reliable_agents_labs.models import GeminiOpenAICompatibleClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set. Copy .env.example to .env and fill it in "
    "to actually close Phase 3's gate, this test must pass for real before "
    "Project 1's first chapter, not just be present.",
)


async def test_gemini_generates_real_text():
    client = GeminiOpenAICompatibleClient()
    result = await client.generate(
        system="Reply with exactly one word: the color of the sky on a clear day.",
        user="What color?",
    )
    assert result.text.strip()
    assert result.provider == "gemini"
    assert result.input_tokens > 0
