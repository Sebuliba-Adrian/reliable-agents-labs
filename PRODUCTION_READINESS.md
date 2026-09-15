# Production Readiness

A real, honest checklist for this project, written from chapter 32 of
*Building Reliable AI Agents*. Every line below names the exact chapter
that verified it, live, against a real service. Nothing here is
aspirational; anything not checked is named as a real, current gap,
not glossed over.

## Verified, live, across 31 chapters

- **Structured output over free text** (ch. 5): typed answers a caller
  can branch on, not prose to keyword-sniff.
- **Real tool calling, multi-turn** (ch. 6, 22): `run_tool_loop` runs
  every tool call a turn makes, not just the first, and converges or
  raises a real, explicit error.
- **Containerized, pinned by digest** (ch. 9): the reorder agent's own
  Dockerfile, `python:3.13-slim` pinned to its real digest.
- **Idempotent ingestion** (ch. 14): re-running sync never duplicates
  or silently deletes data spelled differently than a canonical name.
- **Grounded, cited retrieval** (ch. 15): answers cite exactly what was
  retrieved, and refuse honestly when nothing relevant was found.
- **A real evaluation tier** (ch. 16, 21, 29): a golden dataset, a pass
  rate, a regression check against the last recorded run rather than a
  fixed floor, and retrieval recall scored separately from citation
  accuracy, catching a retrieval miss instead of only ever reporting
  it as an undifferentiated failure.
- **A faithfulness judge for questions with no golden answer** (ch.
  16): `judge_faithfulness` scores an answer against its own context
  rather than a pre-written expectation, verified live to actually
  discriminate a faithful answer from a fabricated one, not just
  assumed to.
- **A real graph, not a re-derived one** (ch. 18, 19): dependency edges
  stored once, queried many times, loaded by a real ETL pipeline.
- **Hybrid retrieval** (ch. 20, 21): vector search and graph traversal
  in one call, closing a real gap plain similarity search cannot.
- **Persistence across a real process restart** (ch. 25): a real
  SQLite checkpoint file, verified across two genuinely separate
  process invocations.
- **A real human-in-the-loop checkpoint** (ch. 26): `interrupt()` on
  the one branch that takes an irreversible action, never on every
  branch.
- **Retry with backoff on real transient errors** (ch. 27): exactly the
  categories a provider's own SDK calls transient, never a permanent
  error retried for no reason.
- **Self-correction on a converged-but-wrong answer** (ch. 27): an
  evaluator node reuses chapter 16's faithfulness judge to catch an
  answer that drifted from its own tool output, retries with the
  judge's own feedback, and gives up gracefully on a real, explicit
  budget rather than looping forever.
- **Tracing without a heavy dependency** (ch. 28): `@observe` on plain
  LangGraph node functions, no LangChain callback integration package.
- **Input, timeout, and API-key guardrails on the HTTP boundary**
  (ch. 30): a real length limit, a real request timeout, and a real
  key check, all three missing until found live, verified against all
  three real cases the last one adds: no key, the wrong key, the real
  key.
- **Real, dated cost accounting** (ch. 31): a real per-token price,
  checked directly, turned into a real dollar figure per task.

## Real, current gaps, named rather than hidden

- **A shared key, not real identity or per-user permissions.**
  Chapter 30's `require_api_key` answers "does this caller hold the
  one key this deployment issued," not "which caller is this" or
  "what is this caller allowed to do." Distinguishing callers, scoping
  what each one can reach, and rotating a compromised key without
  redeploying every client are Book 3's job.
- **SQLite checkpointing, not a production database.** Chapter 25's
  own `AsyncSqliteSaver` docstring says so directly: real production
  write load needs PostgreSQL, an interface swap this project never
  made because Project 4 never ran at that scale.
- **The Dockerfile only covers Project 1.** It containerizes the
  reorder agent's own tool-calling loop; it was never extended to run
  the RAG pipeline, the graph, or the full LangGraph workflow.
- **No per-caller rate limiting.** Chapter 30 bounds one request's own
  size and duration; nothing bounds how many requests one caller can
  send.
- **Cost accounting covers model calls only.** Chapter 31's own
  `estimate_cost` prices `ModelResult`, not Qdrant's, Neo4j's, or
  Langfuse's own real infrastructure cost to run.
- **Single-region, single-instance everywhere.** No project in this
  book ever ran more than one instance of anything.

## Real, final numbers

```
114 passed, 0 skipped, 1 warning in 218.51s
```

Five tiers, 25 real source modules, one companion repository, verified
against real, running services throughout: Gemini, Qdrant, Neo4j,
Langfuse. `uv sync --locked` resolves cleanly from a fresh clone,
verified the same day this file was written.
