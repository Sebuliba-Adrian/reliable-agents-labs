"""Chapter 16: tier 5, evaluation. A golden dataset with a known-correct
answer per question, scored against the real `ask_rag_agent`. Unlike
tiers 1-4, a single failure here is not necessarily a bug, model output
is not perfectly reproducible, this tier is scored as a pass rate against
a threshold, not asserted pass/fail question by question.
"""

from dataclasses import dataclass

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


@dataclass
class EvalResult:
    question: str
    expected: str | None
    actual: list[str]
    passed: bool


async def evaluate_example(example: GoldenExample, **kwargs) -> EvalResult:
    answer = await ask_rag_agent(example.question, **kwargs)
    if example.expected_citation is None:
        passed = answer.cited_packages == []
    else:
        passed = example.expected_citation in answer.cited_packages
    return EvalResult(
        question=example.question,
        expected=example.expected_citation,
        actual=answer.cited_packages,
        passed=passed,
    )


async def run_evaluation(
    dataset: list[GoldenExample] = GOLDEN_DATASET, **kwargs
) -> list[EvalResult]:
    return [await evaluate_example(example, **kwargs) for example in dataset]


def pass_rate(results: list[EvalResult]) -> float:
    return sum(r.passed for r in results) / len(results)
