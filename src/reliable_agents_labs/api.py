"""An HTTP layer over the reorder agent. Chapter 7: the same
`ask_reorder_agent_with_tools` from chapter 6, reachable over the
network instead of only from a Python script.

`get_model_client` is FastAPI's own dependency-injection mechanism doing
exactly what the `client` parameter has done since chapter 4: letting a
test substitute a scripted, fake client without a real network call. The
mechanism is FastAPI-specific, the reason for it is not.
"""

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from reliable_agents_labs.models import ModelClient, build_model_client
from reliable_agents_labs.reorder_agent import ask_reorder_agent_with_tools

app = FastAPI(title="Reorder Agent API")


def get_model_client() -> ModelClient:
    return build_model_client("answer_model")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, client: ModelClient = Depends(get_model_client)) -> AskResponse:
    answer = await ask_reorder_agent_with_tools(request.question, client=client)
    return AskResponse(answer=answer)
