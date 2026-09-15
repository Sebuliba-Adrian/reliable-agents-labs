"""Chapter 16: tier 5, evaluation. A golden dataset with a known-correct
answer per question, scored against the real `ask_rag_agent`. Unlike
tiers 1-4, a single failure here is not necessarily a bug, model output
is not perfectly reproducible, this tier is scored as a pass rate against
a threshold, not asserted pass/fail question by question.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel

from reliable_agents_labs.json_parsing import parse_json_object
from reliable_agents_labs.models import ModelClient
from reliable_agents_labs.rag_agent import ask_rag_agent


@dataclass
class GoldenExample:
    question: str
    # None means: no package should be cited, this question has no
    # correct answer in the ingested data, and saying so is the right
    # answer, chapter 15's own verified finding, now made repeatable.
    expected_citation: str | None


GOLDEN_DATASET: list[GoldenExample] = [
    GoldenExample("What can I use to parse YAML files?", "PyYAML"),
    GoldenExample("What library talks to the Qdrant vector database?", "qdrant-client"),
    GoldenExample("What handles LLM observability and tracing?", "langfuse"),
    GoldenExample("What loads environment variables from a .env file?", "python-dotenv"),
    GoldenExample("What is the official Python library for Anthropic's API?", "anthropic"),
    GoldenExample("What ASGI server is described as lightning-fast?", "uvicorn"),
    GoldenExample("What library does data validation using type hints?", "pydantic"),
    GoldenExample("What package handles Kubernetes deployments?", None),
]

# Chapter 21: the same question chapter 17 could only answer with an
# honest refusal. Scored against `ask_graph_rag_agent`, not the plain
# `ask_rag_agent` this dataset was originally written for, see
# `evaluate_example`'s `ask_fn` parameter.
STRUCTURAL_GOLDEN_DATASET: list[GoldenExample] = [
    GoldenExample(
        "If pydantic had a breaking change, which of my other packages would be affected?",
        "fastapi",
    ),
    GoldenExample("What package handles Kubernetes deployments?", None),
]


@dataclass
class EvalResult:
    question: str
    expected: str | None
    actual: list[str]
    passed: bool
    # `passed` alone cannot tell a retrieval miss (the expected package
    # never came back from search) apart from a generation miss (it
    # came back, the model just didn't cite it). `retrieved` and
    # `retrieval_hit` come from `on_retrieval`, not from `actual`, which
    # only ever carried what the model chose to cite.
    retrieved: list[str] = field(default_factory=list)
    # None when expected_citation is itself None: there is nothing for
    # retrieval to have found, so "did retrieval find it" doesn't apply.
    retrieval_hit: bool | None = None


async def evaluate_example(example: GoldenExample, ask_fn=ask_rag_agent, **kwargs) -> EvalResult:
    """`ask_fn` defaults to chapter 15's plain vector-only agent, the one
    this dataset and this function were both written against. Chapter 21
    passes `ask_graph_rag_agent` instead, same scoring logic, a different
    agent doing the retrieving.
    """
    retrieved_names: list[str] = []

    def _capture_retrieval(results: list[dict]) -> None:
        # `hybrid_search` (chapter 20) puts a package name in front of
        # the model two ways: as a top-level result, or nested in
        # another result's own `dependents`, per `build_hybrid_context`.
        # A retrieval check that only reads `name` would call this a
        # miss even when the model could see the package right there in
        # its own context, a real gap this evaluation itself found live
        # the first time it ran against `ask_graph_rag_agent`, not a
        # hypothetical.
        for r in results:
            retrieved_names.append(r["name"])
            retrieved_names.extend(r.get("dependents", []))

    answer = await ask_fn(example.question, on_retrieval=_capture_retrieval, **kwargs)
    if example.expected_citation is None:
        passed = answer.cited_packages == []
        retrieval_hit = None
    else:
        retrieval_hit = example.expected_citation in retrieved_names
        passed = example.expected_citation in answer.cited_packages
    return EvalResult(
        question=example.question,
        expected=example.expected_citation,
        actual=answer.cited_packages,
        retrieved=retrieved_names,
        retrieval_hit=retrieval_hit,
        passed=passed,
    )


async def run_evaluation(
    dataset: list[GoldenExample] = GOLDEN_DATASET, ask_fn=ask_rag_agent, **kwargs
) -> list[EvalResult]:
    return [await evaluate_example(example, ask_fn=ask_fn, **kwargs) for example in dataset]


def pass_rate(results: list[EvalResult]) -> float:
    return sum(r.passed for r in results) / len(results)


def retrieval_recall(results: list[EvalResult]) -> float:
    """The fraction of examples, among those with a real expected
    citation, where retrieval actually returned it, regardless of
    whether the model went on to cite it. Examples with no expected
    citation carry `retrieval_hit=None` and are excluded, the same way
    they were never scored as a retrieval question in the first place.
    """
    scored = [r for r in results if r.retrieval_hit is not None]
    return sum(r.retrieval_hit for r in scored) / len(scored)


JUDGE_SYSTEM_PROMPT = (
    "You are a strict faithfulness judge. You will be given a question, "
    "a context, and an answer that claims to be grounded in that "
    "context. Decide whether every factual claim the answer makes is "
    "actually supported by the context. An answer that adds a claim the "
    "context never made is not faithful, even if that claim happens to "
    "be true in the real world, and an answer that contradicts the "
    "context is not faithful either. Respond with a single JSON object, "
    "no markdown fences, no commentary: "
    '{"faithful": true or false, "reasoning": "one sentence explaining '
    'your verdict"}.'
)


class JudgeResult(BaseModel):
    faithful: bool
    reasoning: str


async def judge_faithfulness(
    question: str, context: str, answer: str, model_client: ModelClient
) -> JudgeResult:
    """`evaluate_example` above only scores a question this book already
    wrote a `GoldenExample` for, a known question with a known-correct
    citation. Production traffic asks questions nobody anticipated,
    there is no golden answer to check against. A second model call,
    scoring the first model's answer against the same context it was
    given rather than against a pre-written expectation, works on any
    question, known or not.

    `context` and `answer` are passed in directly rather than produced by
    calling an agent internally, so this works against `ask_rag_agent`,
    `ask_graph_rag_agent`, or any future agent's output, without this
    function needing to know which one produced them.
    """
    user_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer: {answer}"
    result = await model_client.generate(system=JUDGE_SYSTEM_PROMPT, user=user_prompt)
    payload = parse_json_object(result.text)
    return JudgeResult.model_validate(payload)
