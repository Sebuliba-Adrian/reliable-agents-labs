"""Placeholder orchestration scaffold. Phase 4 (real chapter content) builds
the actual reorder-agent control flow here, chapter by chapter. This
minimal `run_once` exists only so Phase 3 can prove the five-tier test
taxonomy, especially tier 2 (agent orchestration against a scripted fake
model), is genuinely wired before any real chapter code lands on top of it.
"""

from reliable_agents_labs.models import ModelClient


async def run_once(client: ModelClient, system: str, user: str) -> str:
    """The simplest possible thing worth calling "orchestration": call the
    model once, return its text. Real chapters replace this with the
    reorder agent's actual control flow (tool calling, then the LangGraph
    stateful workflow in Project 4), this stays only as the tier-2 test's
    subject until then.
    """
    result = await client.generate(system=system, user=user)
    return result.text
