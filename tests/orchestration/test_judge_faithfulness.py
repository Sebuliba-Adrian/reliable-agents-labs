"""Tier 2: `judge_faithfulness`'s own orchestration test, against a
scripted model, never a real one. Proves the parsing and plumbing work,
not that a real model judges correctly, that is `tests/evals`'s job.
"""

from reliable_agents_labs.evaluation import judge_faithfulness
from reliable_agents_labs.models import ModelResult
from tests.fakes import ScriptedModelClient


def _model_result(text: str) -> ModelResult:
    return ModelResult(
        text=text, input_tokens=30, output_tokens=15, model_id="scripted", provider="scripted"
    )


async def test_judge_parses_a_faithful_verdict():
    model = ScriptedModelClient(
        [_model_result('{"faithful": true, "reasoning": "The answer only restates the context."}')]
    )

    result = await judge_faithfulness(
        question="What does httpx do?",
        context="httpx: a fully featured HTTP client for Python.",
        answer="httpx is an HTTP client for Python.",
        model_client=model,
    )

    assert result.faithful is True
    assert "context" in result.reasoning


async def test_judge_parses_an_unfaithful_verdict():
    model = ScriptedModelClient(
        [
            _model_result(
                '{"faithful": false, "reasoning": '
                '"The answer claims httpx is a web framework, the context never says that."}'
            )
        ]
    )

    result = await judge_faithfulness(
        question="What does httpx do?",
        context="httpx: a fully featured HTTP client for Python.",
        answer="httpx is a full web framework for building servers.",
        model_client=model,
    )

    assert result.faithful is False
    assert "web framework" in result.reasoning
