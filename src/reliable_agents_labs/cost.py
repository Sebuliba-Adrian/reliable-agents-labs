"""Chapter 31: `gemini-3.6-flash`'s own real, current standard pricing,
Google's own published rate through 2026-12-31: $0.75 per million
input tokens, $3.75 per million output tokens (doubling on 2027-01-01,
a real, dated fact worth re-checking rather than assuming forever).
Every `ModelResult` since chapter 4 has carried real `input_tokens`
and `output_tokens`; this turns those into a real dollar figure
instead of an abstract count nobody ever added up.
"""

from dataclasses import dataclass, field

from reliable_agents_labs.models import ModelResult

INPUT_COST_PER_MILLION_TOKENS = 0.75
OUTPUT_COST_PER_MILLION_TOKENS = 3.75


def estimate_cost(result: ModelResult) -> float:
    input_cost = result.input_tokens / 1_000_000 * INPUT_COST_PER_MILLION_TOKENS
    output_cost = result.output_tokens / 1_000_000 * OUTPUT_COST_PER_MILLION_TOKENS
    return input_cost + output_cost


@dataclass
class TaskCostTracker:
    """A real, if minimal, accumulator: pass `track` as `run_tool_loop`'s
    `on_result` callback to see the real dollar cost of an entire
    multi-turn task, not just its final answer's own last call.
    """

    results: list[ModelResult] = field(default_factory=list)

    def track(self, result: ModelResult) -> None:
        self.results.append(result)

    @property
    def total_cost(self) -> float:
        return sum(estimate_cost(r) for r in self.results)

    @property
    def call_count(self) -> int:
        return len(self.results)
