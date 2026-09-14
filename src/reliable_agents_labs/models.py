"""The tiny provider-abstraction seam for this book.

This is deliberately NOT a general-purpose model router (that's what
LiteLLM/LangChain do, and reaching for one here would undercut the whole
point: teaching the durable mechanism, not adding another framework to
learn). `ModelClient` is just enough of a seam to let:

  1. A deterministic, scripted fake stand in during agent-orchestration
     tests (tier 2 of the five-tier taxonomy), so control-flow tests never
     make a real network call.
  2. A reader switch from the default provider (Gemini, via its
     OpenAI-compatible endpoint) to Anthropic's native SDK by changing
     `config/models.yaml`, not by rewriting application code.

Gemini is the default because a live key is available for real
smoke-testing of every example in this book as it's written and as CI runs
it, not because it's the "best" provider, that question is explicitly out
of scope (see book-playbook/kdp-publishing-playbook.md's sibling notes on
avoiding framework/vendor churn).
"""

from __future__ import annotations

import os
from typing import Protocol

from pydantic import BaseModel


class ModelResult(BaseModel):
    """What every adapter returns, regardless of provider."""

    text: str
    input_tokens: int
    output_tokens: int
    model_id: str
    provider: str


class ModelClient(Protocol):
    """The provider seam. Two real adapters implement this (Gemini, Anthropic);
    a third, `ScriptedModelClient` in tests/, implements it for deterministic
    orchestration tests. Application code only ever depends on this Protocol,
    never on a concrete provider SDK directly.
    """

    async def generate(self, *, system: str, user: str) -> ModelResult: ...


class GeminiOpenAICompatibleClient:
    """Default adapter: Gemini via its OpenAI-compatible endpoint, using the
    standard `openai` SDK pointed at a different base_url. This is real,
    tested code, not a stub, since a live GEMINI_API_KEY is available.
    """

    def __init__(self, model_id: str | None = None) -> None:
        from openai import AsyncOpenAI  # local import: keep the seam thin

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._model_id = model_id or os.environ.get("GEMINI_MODEL_ID", "gemini-3.6-flash")

    async def generate(self, *, system: str, user: str) -> ModelResult:
        response = await self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        return ModelResult(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model_id=self._model_id,
            provider="gemini",
        )


class AnthropicClient:
    """Second adapter: Anthropic's native SDK. Not the default (see module
    docstring), but a real, working alternative selectable via
    config/models.yaml, not a documented-but-unimplemented placeholder.
    """

    def __init__(self, model_id: str | None = None) -> None:
        from anthropic import AsyncAnthropic  # local import: keep the seam thin

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = AsyncAnthropic(api_key=api_key)
        self._model_id = model_id or os.environ.get("ANTHROPIC_MODEL_ID", "claude-sonnet-5")

    async def generate(self, *, system: str, user: str) -> ModelResult:
        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model_id=self._model_id,
            provider="anthropic",
        )


def build_model_client(role: str, config_path: str = "config/models.yaml") -> ModelClient:
    """Construct the right adapter for a named role (e.g. "answer_model"),
    reading provider/model choice from config, never hardcoded in
    application code. See config/models.yaml.
    """
    import yaml

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    role_config = config[role]
    provider = role_config["provider"]
    model_id = role_config.get("model_id")

    if provider == "gemini":
        return GeminiOpenAICompatibleClient(model_id=model_id)
    if provider == "anthropic":
        return AnthropicClient(model_id=model_id)
    raise ValueError(f"Unknown provider {provider!r} for role {role!r}")
