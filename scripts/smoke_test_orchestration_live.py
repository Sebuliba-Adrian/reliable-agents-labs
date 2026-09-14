"""Live smoke test: proves the REAL Gemini client and the REAL orchestration
path (run_once) actually work together, not just independently.

tests/contract verifies the raw client alone. tests/orchestration verifies
run_once's control flow, but only against a scripted fake. Neither proves
the two combine correctly end to end, this script does, following the
same pattern as Book 1's scripts/smoke_test_openai_compatible.py.

Deliberately kept OUT of the pytest suite (real network call, not free,
not deterministic).

Usage:
    uv run python scripts/smoke_test_orchestration_live.py
"""

import asyncio

from reliable_agents_labs.models import GeminiOpenAICompatibleClient
from reliable_agents_labs.orchestration import run_once


async def main() -> None:
    client = GeminiOpenAICompatibleClient()
    text = await run_once(
        client,
        system="You are a terse warehouse assistant. Reply in one short sentence.",
        user="We have 4 units of SKU-1029 left. Is that low?",
    )
    print(f"model: {client._model_id}")
    print(f"response: {text!r}")
    assert text.strip(), "got an empty response"
    print("\nOK: real Gemini call flowed through the real orchestration path end to end.")


if __name__ == "__main__":
    asyncio.run(main())
