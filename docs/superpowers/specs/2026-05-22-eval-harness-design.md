# JobPilot v3 — Item 1: Eval Harness Design

**Status:** Draft for user review
**Date:** 2026-05-22
**Scope:** v3 item 1 only ("Eval harness") per [`PLAN-update.md`](../../../PLAN-update.md#1-eval-harness). Items 2–5 (observability, HITL, retrieval quality, memory) are out of scope here but called out where they touch this design.

---

## 1. Goal

Make prompt, model, and retrieval changes **measurable** before demos or interviews. One command runs the evaluator (and optionally the tailor) against a labelled fixture set and writes a JSONL row per case plus a Markdown report. A `--baseline` flag turns the same command into a regression checker.

The harness wraps the existing production agents without modifying them.

## 2. Non-goals (deferred)

| Concern | Where it lives |
|---|---|
| Per-step / per-LLM-call run logs with `run_id` propagation | v3 item 2 (observability) |
| HITL approval gates between graph nodes | v3 item 3 (checkpointing) |
| Hybrid retrieval (BM25 + rerank) and LLM-as-judge citation verifier | v3 item 4 (retrieval quality) |
| User-feedback-driven style memory | v3 item 5 (memory loop) |
| Discord, scraper, FastAPI | v2 backlog |

Item 1 ships **deterministic** metrics only. LLM-judge metrics arrive with item 4.

## 3. Decisions taken during brainstorming

| Decision | Choice |
|---|---|
| Fixture sourcing | **Bootstrap-then-curate.** Start with the 3 existing sample JDs + you add real listings over time. First run freezes "actual" as initial "expected"; you correct from there. Grow to 15–30 incrementally. |
| Cost tracking | **Tokens + USD via a pricing YAML** (`evals/model_prices.yaml`). Unknown models report cost as `null` not error. |
| Comparison surface | **Single command, optional `--baseline` flag.** No separate `eval-compare` subcommand. |
| Labelling scope | Keep `expected_decision`, `expected_score_band`, `key_evidence_chunk_substrings`, `required_risk_flags`, `disallowed_risk_flags`. Drop `disallowed_unsupported_claims` (requires LLM judge → item 4). |
| Default execution | `--no-tailor` (evaluator only) by default to keep eval cost low. `--tailor` opts in. |
| CI gate | Include `--fail-under <accuracy>` flag now. Wiring it into GitHub Actions is a follow-up. |
| Frozen baselines | Live under `evals/runs/_baseline_<label>/` and are committed by exception. The default `evals/runs/` directory is gitignored. |

## 4. Approach (rejected alternatives)

We picked the **domain-modeled harness** (a small `evals/` package with one file per concern) over:

- **Single-file `runner.py`** — simpler now but predicted to bloat once item 2 (tracing) and item 4 (citation verifier) plug in.
- **Adding a judge node to the production LangGraph** — would force every prod run to detect eval mode; tight coupling rejected.

## 5. Module layout

```
src/jobpilot/
├── llm/client.py            # + LLMClient.record() context manager (additive only)
├── cli.py                   # + eval-batch command (additive only)
└── evals/                   # all new
    ├── __init__.py
    ├── fixtures.py          # JD + label loader; pure I/O
    ├── pricing.py           # load model_prices.yaml; tokens → USD
    ├── metrics.py           # per-case + aggregate metric computation
    ├── runner.py            # one batch: iterate fixtures → invoke graph → collect EvalRecord
    ├── report.py            # Markdown writer (single-run + comparison views)
    └── compare.py           # diff two JSONL runs → ComparisonReport

evals/                        # new top-level data dir
├── jobs/                     # JD bodies, one per file (git-tracked)
├── labels/                   # expected outcomes, one per JD (git-tracked)
├── model_prices.yaml         # token → USD config (git-tracked)
├── runs/                     # results land here (git-IGNORED by default)
│   └── <ISO timestamp>/
│       ├── results.jsonl
│       ├── report.md
│       └── meta.yaml
└── README.md
```

**Production code (`agents/`, `rag/`, `cli.py:eval`, `cli.py:run`) is unchanged.** The harness invokes the existing `build_graph()` and reads `AgentState`.

### 5.1 Data flow

```
fixtures.load_eval_set(evals/)        →  list[EvalCase]
   ▼
runner.run_batch(cases, settings)     →  BatchResult       (wraps build_graph + LLMClient.record)
   ▼
metrics.score(BatchResult, prices)    →  ScoredBatch
   ▼
report.write(scored, run_dir)         →  evals/runs/<ts>/{results.jsonl, report.md, meta.yaml}
   ▼  (if --baseline supplied)
compare.diff(baseline_jsonl, scored)  →  delta-augmented report.md
```

### 5.2 Required additive change to `LLMClient`

`src/jobpilot/llm/client.py` gains one method. No existing signature changes. Evaluator/tailor agents are oblivious.

```python
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

@dataclass
class CallTelemetry:
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

class LLMClient:
    # ... existing fields ...

    @contextmanager
    def record(self) -> Iterator[list[CallTelemetry]]:
        """Capture per-call telemetry for the duration of the block.

        Nested blocks are not supported in item 1 (asserts if entered twice).
        """
        ...
```

`complete_structured()` is wrapped internally so it appends a `CallTelemetry` entry whenever a recording is active. Token counts come from `response.usage`; latency from a `time.monotonic()` wrap.

The eval runner uses it like:

```python
with llm.record() as calls:
    out = await graph.ainvoke(state)
# calls is list[CallTelemetry], one entry per LLM call inside the graph
```

## 6. On-disk data model

### 6.1 Job descriptions: `evals/jobs/<stem>.txt`

Plain text, the same format as today's `data/sample_jobs/`. The 3 existing JDs are moved (`git mv data/sample_jobs/*.txt evals/jobs/`), `data/sample_jobs/` is removed, and any test that referenced it is updated.

### 6.2 Expected outcomes: `evals/labels/<stem>.yaml`

One file per JD. Only `expected_decision` is required.

```yaml
expected_decision: apply              # apply | maybe | skip   (required)
expected_score_band: [70, 90]         # [min, max] inclusive   (optional)
key_evidence_chunk_substrings:        # at least one must appear in cited chunks (optional)
  - "RAG"
  - "LangGraph"
required_risk_flags: []               # substrings that MUST appear in risk_flags (optional)
disallowed_risk_flags: []             # substrings that MUST NOT appear (optional)
notes: ""                             # human notes; not scored
```

Semantics for optional fields: omitted or empty list → metric is **N/A** for that case and is excluded from aggregate denominators.

### 6.3 Pricing config: `evals/model_prices.yaml`

```yaml
# USD per 1k tokens; key is the LiteLLM model id
"claude-sonnet-4-6":
  input: 0.003
  output: 0.015
"claude-haiku-4-5-20251001":
  input: 0.001
  output: 0.005
```

`pricing.cost(model, in_tokens, out_tokens) -> float | None` returns `None` for unknown models; reports show `n/a`.

### 6.4 Pydantic schemas

Defined in `evals/fixtures.py` so loaders are type-checked end-to-end:

```python
class ExpectedOutcome(BaseModel):
    expected_decision: Decision                                  # imported from models.schemas
    expected_score_band: tuple[int, int] | None = None
    key_evidence_chunk_substrings: list[str] = []
    required_risk_flags: list[str] = []
    disallowed_risk_flags: list[str] = []
    notes: str = ""

class EvalCase(BaseModel):
    stem: str                       # filename stem; primary key
    jd: JobDescription              # reused from models.schemas
    expected: ExpectedOutcome
```

`load_eval_set(jobs_glob, labels_dir)` returns `list[EvalCase]`. JDs without a matching label file emit a warning and are excluded (not an error).

`expected_score_band` is declared as `tuple[int, int]` in Python but YAML deserialises lists; the model includes a validator that coerces a 2-element list into a tuple and rejects anything else with a clear error.

## 7. Metrics

### 7.1 Per-case (deterministic)

| Metric | Formula | N/A when |
|---|---|---|
| `decision_correct` | `eval.decision == expected.expected_decision` | never |
| `score_in_band` | `band[0] ≤ eval.score ≤ band[1]` | `expected_score_band` omitted |
| `score_abs_error` | `abs(eval.score - midpoint(band))` where `midpoint = (band[0] + band[1]) / 2` (float) | `expected_score_band` omitted |
| `citation_evidence_ok` | `any(sub.lower() in cited_texts.lower() for sub in ...)` | list empty/omitted |
| `required_risk_flags_present` | all required substrings present in `risk_flags` | list empty/omitted |
| `disallowed_risk_flags_absent` | none of the disallowed substrings appear in `risk_flags` | list empty/omitted |
| `cited_chunks_retrieved` | `set(cited_chunk_ids) ⊆ set(retrieved_chunk_ids)` | never (sanity check) |
| `latency_ms` | wall time around `graph.ainvoke(state)` | never |
| `input_tokens` / `output_tokens` | sum over `CallTelemetry` | never |
| `estimated_usd` | `pricing.cost(model, in, out)` | unknown model → `null` |

### 7.2 Aggregate

- `decision_accuracy` = mean of `decision_correct` over **non-errored** cases
- `score_mae`, `score_in_band_rate` — computed only over cases that had a `expected_score_band` (denominator reported)
- `citation_evidence_pass_rate`, `required_risk_flags_pass_rate`, `disallowed_risk_flags_pass_rate` — each over cases that had the corresponding non-empty label list (denominator reported)
- `p50_latency_ms`, `p95_latency_ms`
- `total_input_tokens`, `total_output_tokens`, `total_usd`, `mean_usd_per_known_case` (USD averaged over cases with a known model price)
- `n_cases`, `n_errors` (cases that raised during `graph.ainvoke` are bucketed separately and do **not** count as a "wrong decision")

## 8. Run outputs

### 8.1 `results.jsonl` — one row per case

```json
{
  "stem": "canva_fullstack",
  "ts": "2026-05-22T14:30:00Z",
  "prompt_version": "v1",
  "model": "claude-sonnet-4-6",
  "expected": { "decision": "apply", "score_band": [70, 90], "key_evidence_chunk_substrings": ["RAG"], "required_risk_flags": [], "disallowed_risk_flags": [] },
  "actual": {
    "decision": "apply",
    "score": 82,
    "reasoning": "...",
    "cited_chunk_ids": ["chunk_03", "chunk_07"],
    "risk_flags": []
  },
  "retrieved_chunk_ids": ["chunk_03", "chunk_07", "chunk_11", "..."],
  "metrics": {
    "decision_correct": true,
    "score_in_band": true,
    "score_abs_error": 2,
    "citation_evidence_ok": true,
    "required_risk_flags_present": null,
    "disallowed_risk_flags_absent": null,
    "cited_chunks_retrieved": true
  },
  "telemetry": {
    "latency_ms": 4231,
    "calls": [
      { "model": "claude-sonnet-4-6", "input_tokens": 3104, "output_tokens": 287, "latency_ms": 4180 }
    ],
    "input_tokens": 3104,
    "output_tokens": 287,
    "estimated_usd": 0.0136
  },
  "error": null
}
```

**Errored case shape:** if `graph.ainvoke(state)` raises, the row is written with `actual: null`, `metrics: null`, `retrieved_chunk_ids: []`, `telemetry: { latency_ms: <time until raise>, calls: [], input_tokens: 0, output_tokens: 0, estimated_usd: null }`, and `error: { "type": "<exception class>", "message": "<str(exc)>" }`. The run continues to the next case; errors appear in their own section of the report and do **not** count toward `decision_accuracy` (numerator or denominator).

**Failure case definition:** a case is a *Failure* (appears in the Failures section) iff `error is null` AND at least one **applicable** metric is false. A metric is "applicable" when its corresponding label field was non-empty (i.e. not N/A). A case that passes every applicable metric does not appear in Failures.

### 8.2 `report.md` — single run

```markdown
# JobPilot Eval — 2026-05-22T14:30:00Z

**Config:** model=`claude-sonnet-4-6`, prompt_version=`v1`, n_cases=8

## Summary
| metric                          | value          |
| ------------------------------- | -------------- |
| decision_accuracy               | 7/8 (87.5%)    |
| score_mae                       | 6.3 (n=5)      |
| score_in_band_rate              | 4/5 (80%)      |
| citation_evidence_pass_rate     | 3/4 (75%)      |
| required_risk_flags_pass_rate   | 2/2 (100%)    |
| disallowed_risk_flags_pass_rate | 1/1 (100%)    |
| p50 latency / p95               | 3.9s / 7.2s    |
| total tokens (in / out)         | 24,832 / 2,104 |
| total cost                      | $0.106         |
| errors                          | 0              |

## Failures (3)
### canva_fullstack — score out of band
- expected_decision=apply, got=apply ✓
- expected_score_band=[70,90], got=64 ✗
- reasoning excerpt: "…strong RAG match but flagged frontend gap…"
- cited: chunk_03, chunk_07

### openai_backend — wrong decision
...

## Errors (0)
*(none)*
```

Failures are sorted by `stem` so report diffs are stable across runs.

### 8.3 `report.md` — with `--baseline`

The single-run report gains two sections at the top:

```markdown
## Delta vs baseline (2026-05-21T09:00:00Z, model=claude-sonnet-4-6, prompt=v1)
| metric            | baseline | candidate | Δ       |
| ----------------- | -------- | --------- | ------- |
| decision_accuracy | 6/8      | 7/8       | +12.5pp |
| score_mae         | 8.1      | 6.3       | -1.8    |
| total_usd         | $0.142   | $0.106    | -$0.036 |

## Regressions (1)  # passed in baseline but fail in candidate
### anthropic_research_eng — decision flipped
- baseline: decision=apply, score=78  ✓
- candidate: decision=maybe, score=66 ✗
- diff: decision apply→maybe, score 78→66
```

The Regressions section satisfies the plan's acceptance check: *"failure cases that explain what changed and why it matters."* Cases present in only one of the two runs are listed under "Added" / "Removed" subsections.

### 8.4 `meta.yaml`

```yaml
run_id: 2026-05-22T14-30-00
ts: 2026-05-22T14:30:00Z
model: claude-sonnet-4-6
prompt_version: v1
n_cases: 8
settings:
  retrieval_k: 8
  score_threshold: 70
git_sha: 85ee26a              # from git rev-parse HEAD if available
```

`git_sha` is best-effort; missing git → field omitted.

## 9. CLI surface

Added to `src/jobpilot/cli.py`. The existing `version`, `ingest`, `eval`, `run` commands are unchanged.

```bash
jobpilot eval-batch [JOBS_GLOB] [OPTIONS]
```

| Flag | Default | Purpose |
|---|---|---|
| `JOBS_GLOB` (positional) | `evals/jobs/*.txt` | Which JD files to run |
| `--labels-dir PATH` | `evals/labels/` | Where to look for `<stem>.yaml`. Missing label → case skipped with warning, not an error |
| `--prompt-version TEXT` | `v1` | Forwarded to `EvaluatorAgent(prompt_version=...)` and tailor |
| `--model TEXT` | `settings.llm_model` | Override `LLM_MODEL` for this run; written to `meta.yaml` |
| `--baseline PATH` | none | If set, report includes delta + regressions section |
| `--run-dir PATH` | `evals/runs/<ISO ts>/` | Where outputs go |
| `--tailor / --no-tailor` | `--no-tailor` | Run only the evaluator by default. `--tailor` runs the full graph (no docx written) |
| `--fail-under FLOAT` | none | Exit non-zero if `decision_accuracy < value`. CI/regression gate |
| `--prices PATH` | `evals/model_prices.yaml` | Pricing config path |
| `--concurrency INT` | `1` | Run N cases in parallel via `asyncio.gather`. Default 1 keeps ordering deterministic |

Stdout stays terse: prints the summary table from `report.md` plus the path to the full report.

## 10. Bootstrap workflow

How the harness goes from "3 unlabeled JDs" to 15–30 curated cases over time:

1. **One-time move** as part of implementation:
   - `git mv data/sample_jobs/*.txt evals/jobs/`
   - Remove `data/sample_jobs/` and update any test references
2. **Starter labels** — one YAML per JD with only `expected_decision` filled in. The implementer drafts these from the JD content; user verifies during spec review.
3. **First batch run**:
   ```bash
   uv run jobpilot eval-batch
   ```
   Produces `evals/runs/<ts>/{results.jsonl, report.md, meta.yaml}`.
4. **Curate from the report.** For each JD where the model's score looks right, copy it into `expected_score_band = [score-10, score+10]`. Where the model cites garbage, add `key_evidence_chunk_substrings`. Where you spot a model-paranoia case, add `disallowed_risk_flags`. Re-run.
5. **Grow the set.** Drop a `.txt` in `evals/jobs/` + a `.yaml` (minimum: `expected_decision`) in `evals/labels/`. Harness picks it up automatically.
6. **Freeze a baseline** when a run feels right:
   ```bash
   cp -r evals/runs/2026-05-22T14-30-00 evals/runs/_baseline_v1
   git add evals/runs/_baseline_v1
   ```
   Future runs compare with `--baseline evals/runs/_baseline_v1/results.jsonl`.

`evals/runs/_baseline_*/` is committed by exception; `evals/runs/` is otherwise gitignored. The leading underscore is the "track this one" marker, documented in `evals/README.md`.

## 11. Testing strategy

### 11.1 Unit tests (`tests/unit/`)

| File | Covers |
|---|---|
| `test_evals_fixtures.py` | YAML → `ExpectedOutcome`; defaults; missing label file → warning case; bad YAML → clear error |
| `test_evals_pricing.py` | Known model → correct USD; unknown model → `None`; zero tokens → `0.0`; schema validation |
| `test_evals_metrics.py` | All deterministic per-case checks; aggregate math (denominators, p95 on small samples); errored cases excluded from accuracy denominator |
| `test_evals_report.py` | Markdown rendering from a hand-built `ScoredBatch`; failure-only listing; with/without baseline branches; sorted-by-stem determinism |
| `test_evals_compare.py` | JSONL diff: regressions, improvements, added/removed cases; identical runs → empty regressions |
| `test_llm_client_record.py` | `LLMClient.record()`: accumulates within block, isolates between blocks, captures input/output tokens + latency from a fake SDK; nested entry asserts |

### 11.2 Integration test (`tests/integration/test_eval_batch.py`)

Boots the full harness with:
- A `tmp_path`-based `evals/` containing 2 JD files + 2 label files
- A `FakeLLMClient` (extends the existing test fake) that returns canned `EvaluationResult` per JD and emits fake `response.usage`
- The real `RagStore` against a tmp Chroma dir seeded with 3 fixture chunks
- The Typer app via `CliRunner`

Assertions:
- Exit 0
- `results.jsonl` exists, has 2 rows, parses per schema
- `report.md` contains "decision_accuracy"
- `meta.yaml` records prompt_version + model
- One case passes all metrics, the other has a deliberate decision flip → appears under "Failures", not "Errors"
- `--fail-under 1.0` → exit 1; `--fail-under 0.0` → exit 0
- `--baseline <first-run>` produces a report with a "Delta vs baseline" section

### 11.3 Test fixtures (`tests/fixtures/evals.py`)

- `make_eval_case(stem, decision, score_band, ...)`
- `make_jsonl_row(...)` for compare tests
- 2-chunk pre-built `RagStore` factory

### 11.4 Coverage

Project gate stays at 70%. New `evals/` modules should land near 90% (mostly pure functions). The integration test backstops runner + CLI behavior.

### 11.5 Out of scope for item 1

- Real LLM calls inside tests (cost; bootstrap step 3 is the first real call)
- `--concurrency > 1` (Chroma thread-safety not verified; flag as v3 item 2 concern)
- Markdown visual diff (assertions are structural)

## 12. Acceptance check

Quoting the plan: *"one command produces a summary table plus failure cases that explain what changed and why it matters."* This design satisfies the check via:

- `jobpilot eval-batch` (single command)
- Summary table in `report.md` (Section 8.2)
- `--baseline` enables Regressions section that itemises pass→fail flips with diffs (Section 8.3)

## 13. Open follow-ups (NOT part of item 1)

- Wire `--fail-under` into a GitHub Actions workflow (item 2 concern, ships with observability so the CI run can publish run logs)
- Add per-step JSONL inside `evals/runs/<ts>/` (item 2)
- Replace deterministic `citation_evidence_ok` with an LLM-judge that validates each tailored claim against retrieved chunks (item 4)
- Concurrency > 1 once Chroma thread-safety is verified (item 2)
