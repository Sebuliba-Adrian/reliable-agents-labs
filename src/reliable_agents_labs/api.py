"""An HTTP layer over the reorder agent. Chapter 7: the same
`ask_reorder_agent_with_tools` from chapter 6, reachable over the
network instead of only from a Python script.

`get_model_client` is FastAPI's own dependency-injection mechanism doing
exactly what the `client` parameter has done since chapter 4: letting a
test substitute a scripted, fake client without a real network call. The
mechanism is FastAPI-specific, the reason for it is not.
"""

import asyncio
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from reliable_agents_labs.models import ModelClient, build_model_client
from reliable_agents_labs.observability import ask_reorder_agent_traced

app = FastAPI(title="Reorder Agent API")

# Chapter 30: a real, explicit bound, not "however long a caller feels
# like sending." Generous for any real question this domain has ever
# asked, far short of what a real model call would charge real money
# to process and a real context window to hold.
MAX_QUESTION_LENGTH = 2000

# A real, explicit ceiling. Without one, a single slow or hung real
# model call holds this endpoint's HTTP connection open indefinitely,
# verified live: a fake client sleeping 3 seconds made this endpoint
# wait the full 3 seconds with no upper bound at all.
REQUEST_TIMEOUT_SECONDS = 30

# Chapter 30's third real gap: /ask never checked who was calling it.
# A shared key is not real identity or per-user permissions, just the
# smallest real step from "anyone" to "anyone holding the one key this
# deployment issued." `None` by default is deliberate: a deployment
# that forgets to set this fails closed, rejecting every caller,
# rather than silently accepting all of them.
API_KEY = os.environ.get("API_KEY")


def get_model_client() -> ModelClient:
    return build_model_client("answer_model")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="missing or invalid API key")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
async def ask(request: AskRequest, client: ModelClient = Depends(get_model_client)) -> AskResponse:
    try:
        answer = await asyncio.wait_for(
            ask_reorder_agent_traced(request.question, client=client),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="request timed out") from exc
    return AskResponse(answer=answer)
