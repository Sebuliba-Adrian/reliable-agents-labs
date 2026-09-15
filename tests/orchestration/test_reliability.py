"""Tier 2: RetryingModelClient's own orchestration tests, real openai
exception classes, a scripted flaky client, never a real network call.
See chapter 27.
"""

import httpx2
import pytest
from openai import APIConnectionError

from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.reliability import RetryingModelClient
from tests.fakes import FlakyModelClient

_REQUEST = httpx2.Request("POST", "https://example.com")


def _result(text: str) -> ModelResult:
    return ModelResult(text=text, input_tokens=5, output_tokens=5, model_id="fake", provider="fake")


async def test_recovers_from_a_real_transient_error():
    flaky = FlakyModelClient(
        APIConnectionError(request=_REQUEST), fail_times=2, result=_result("recovered")
    )
    client = RetryingModelClient(flaky, max_attempts=5)

    result = await client.generate(system="s", user="u")

    assert result.text == "recovered"
    assert flaky.calls == 3


async def test_reraises_after_exhausting_max_attempts():
    flaky = FlakyModelClient(
        APIConnectionError(request=_REQUEST), fail_times=10, result=_result("never reached")
    )
    client = RetryingModelClient(flaky, max_attempts=3)

    with pytest.raises(APIConnectionError):
        await client.generate(system="s", user="u")

    assert flaky.calls == 3


async def test_a_permanent_error_is_not_retried():
    flaky = FlakyModelClient(ValueError("bad request"), fail_times=10, result=_result("x"))
    client = RetryingModelClient(flaky, max_attempts=5)

    with pytest.raises(ValueError):
        await client.generate(system="s", user="u")

    assert flaky.calls == 1
