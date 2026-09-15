"""An HTTP layer over the reorder agent. Chapter 7: the same
`ask_reorder_agent_with_tools` from chapter 6, reachable over the
network instead of only from a Python script.

`get_model_client` is FastAPI's own dependency-injection mechanism doing
exactly what the `client` parameter has done since chapter 4: letting a
test substitute a scripted, fake client without a real network call. The
mechanism is FastAPI-specific, the reason for it is not.
"""

import asyncio

from fastapi import Depends, FastAPI, HTTPException
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


def get_model_client() -> ModelClient:
    return build_model_client("answer_model")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, client: ModelClient = Depends(get_model_client)) -> AskResponse:
    try:
        answer = await asyncio.wait_for(
            ask_reorder_agent_traced(request.question, client=client),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="request timed out") from exc
    return AskResponse(answer=answer)
