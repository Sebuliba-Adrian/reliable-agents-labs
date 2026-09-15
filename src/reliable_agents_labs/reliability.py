"""Chapter 27: closes Project 4. Every model call since chapter 4 has
assumed the network just works. `run_tool_loop`, `ask_agent`, the
approval workflow, none of them retry anything, a single transient
connection error anywhere in a run fails the whole thing. This wraps
any real `ModelClient` with retry-with-backoff on exactly the errors a
real provider SDK already classifies as transient, never on a real,
permanent error.
"""

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from reliable_agents_labs.models import ModelClient, ModelResult

# Exactly the categories the openai SDK itself calls out as transient:
# a dropped connection, a timeout, a provider-side 5xx, or a rate limit.
# A real, permanent error, a bad request or an invalid key, is not in
# this tuple on purpose: retrying one of those three more times wastes
# real time and real API cost for a failure that will never succeed.
TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class RetryingModelClient:
    """Wraps any real `ModelClient`. `max_attempts` and the backoff
    schedule are real, explicit tradeoffs, the same kind chapter 22's
    `max_iterations` already was: too few attempts gives up on a
    genuinely transient blip, too many turns a real outage into a long,
    silent hang before the caller ever finds out.
    """

    def __init__(self, client: ModelClient, max_attempts: int = 3) -> None:
        self._client = client
        self._max_attempts = max_attempts

    async def generate(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> ModelResult:
        @retry(
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=0.5, max=8),
            reraise=True,
        )
        async def _call() -> ModelResult:
            return await self._client.generate(
                system=system, user=user, tools=tools, history=history
            )

        return await _call()
