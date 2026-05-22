# v3 Item 1 — Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `jobpilot eval-batch` — a CLI that runs the existing evaluator (and optionally tailor) against a labelled fixture set, writes a JSONL row per case plus a Markdown report, and supports baseline-vs-candidate regression comparison.

**Architecture:** New `src/jobpilot/evals/` package (fixtures, pricing, metrics, runner, report, compare) wraps the unchanged production agents via the existing `build_graph()`. One additive `LLMClient.record()` context manager captures per-call token + latency telemetry. Top-level `evals/` data directory holds JD files, labels, pricing config, and run artifacts.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, PyYAML, pytest, the existing LangGraph/Chroma/LiteLLM stack.

**Spec:** [`docs/superpowers/specs/2026-05-22-eval-harness-design.md`](../specs/2026-05-22-eval-harness-design.md)

---

## File Structure

**New files:**
- `src/jobpilot/evals/__init__.py`
- `src/jobpilot/evals/fixtures.py` — `EvalCase`, `ExpectedOutcome`, `load_eval_set()`
- `src/jobpilot/evals/pricing.py` — `load_prices()`, `cost()`
- `src/jobpilot/evals/metrics.py` — per-case + aggregate scoring, Pydantic DTOs (`PerCaseMetrics`, `BatchAggregates`, `EvalRecord`, `ScoredBatch`, `BatchConfig`)
- `src/jobpilot/evals/runner.py` — `run_batch()` invokes graph + collects records
- `src/jobpilot/evals/report.py` — `write_results_jsonl()`, `write_meta_yaml()`, `write_report_md()`
- `src/jobpilot/evals/compare.py` — `load_scored_batch()`, `diff_runs()`
- `evals/jobs/*.txt` — 3 JDs moved from `data/sample_jobs/`
- `evals/labels/*.yaml` — starter labels for those 3 JDs
- `evals/model_prices.yaml`
- `evals/README.md`
- `tests/fixtures/evals.py` — builders for `EvalCase`, `EvalRecord`, etc.
- `tests/unit/test_llm_client_record.py`
- `tests/unit/test_evals_fixtures.py`
- `tests/unit/test_evals_pricing.py`
- `tests/unit/test_evals_metrics.py`
- `tests/unit/test_evals_report.py`
- `tests/unit/test_evals_compare.py`
- `tests/integration/test_eval_batch.py`

**Modified files:**
- `src/jobpilot/llm/client.py` — add `CallTelemetry`, `record()` context manager; capture `response.usage` in `complete_structured()`
- `src/jobpilot/cli.py` — add `eval-batch` Typer command
- `.gitignore` — `evals/runs/` excluded; `evals/runs/_baseline_*/` allowed
- `README.md` — short pointer to new `evals/` workflow (one paragraph)

**Removed files / paths:**
- `data/sample_jobs/` — moved into `evals/jobs/`

---

## Task 1: Move sample JDs into evals/ and update .gitignore

**Files:**
- Move: `data/sample_jobs/canva_fullstack.txt` → `evals/jobs/canva_fullstack.txt`
- Move: `data/sample_jobs/openai_backend.txt` → `evals/jobs/openai_backend.txt`
- Move: `data/sample_jobs/anthropic_research_eng.txt` → `evals/jobs/anthropic_research_eng.txt`
- Modify: `.gitignore`
- Audit references: any test or doc that mentions `data/sample_jobs/`

- [ ] **Step 1: Verify current references to `data/sample_jobs/`**

Run: `grep -rn "data/sample_jobs" --include="*.py" --include="*.md" --include="*.yaml" --include="Makefile" .`

Expected: matches in `PLAN-update.md`, `PLAN.md`, `README.md`, possibly `Makefile`. Note them — they'll need editing in Step 4.

- [ ] **Step 2: Move the three JD files**

```bash
mkdir -p evals/jobs
git mv data/sample_jobs/canva_fullstack.txt evals/jobs/canva_fullstack.txt
git mv data/sample_jobs/openai_backend.txt evals/jobs/openai_backend.txt
git mv data/sample_jobs/anthropic_research_eng.txt evals/jobs/anthropic_research_eng.txt
rmdir data/sample_jobs
```

- [ ] **Step 3: Update `.gitignore`**

Replace the trailing `.env*` block keeping prior entries, and add a new "Eval harness" stanza before the "Env" block:

```
# Eval harness — run artifacts are local except explicit baselines
evals/runs/
!evals/runs/_baseline_*/
```

The negation pattern is git's "ignore everything in `evals/runs/` except subdirectories whose name starts with `_baseline_`". Verify with `git check-ignore -v evals/runs/whatever` after.

- [ ] **Step 4: Update any docs/tests/Makefile references**

For each file from Step 1 output, replace `data/sample_jobs/` → `evals/jobs/`. Likely files: `README.md`, `Makefile` (if it references a sample JD). Do NOT modify `PLAN.md` or `PLAN-update.md` (they're historical records, gitignored anyway per `.gitignore`).

- [ ] **Step 5: Run existing tests to make sure nothing breaks**

Run: `uv run pytest -q`
Expected: same number of tests pass as before (no test references `data/sample_jobs/` as a hard path, but verify).

If a test fails because of the move, update the test path. If no test fails, good — the JDs were sample-only.

- [ ] **Step 6: Commit**

```bash
git add evals/jobs/ .gitignore README.md Makefile
git rm -r data/sample_jobs/ 2>/dev/null || true
git commit -m "chore(evals): move data/sample_jobs/ → evals/jobs/ and gitignore evals/runs/"
```

---

## Task 2: Add `CallTelemetry` and `LLMClient.record()` context manager

**Files:**
- Modify: `src/jobpilot/llm/client.py`
- Test: `tests/unit/test_llm_client_record.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_llm_client_record.py`:

```python
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from jobpilot.config import Settings
from jobpilot.llm.client import CallTelemetry, LLMClient
from jobpilot.models.schemas import EvaluationResult


def _fake_response(payload: dict[str, Any], in_tok: int, out_tok: int) -> MagicMock:
    """Mirror the existing test fakes in test_llm_client.py but include usage."""
    response = MagicMock()
    tool_call = MagicMock()
    tool_call.function.name = "submit_evaluation"
    tool_call.function.arguments = json.dumps(payload)
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [tool_call]
    response.usage.prompt_tokens = in_tok
    response.usage.completion_tokens = out_tok
    return response


def _payload() -> dict[str, Any]:
    return {
        "score": 50,
        "decision": "maybe",
        "reasoning": "ok",
        "cited_chunk_ids": [],
        "risk_flags": [],
    }


def test_record_captures_token_counts_and_latency(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = _fake_response(_payload(), 100, 20)
    client = LLMClient(settings=settings, sdk=sdk)

    with client.record() as calls:
        client.complete_structured(
            system="s", user="u", cached_context=None,
            schema=EvaluationResult,
            tool_name="submit_evaluation", tool_description="d",
        )

    assert len(calls) == 1
    t = calls[0]
    assert isinstance(t, CallTelemetry)
    assert t.model == settings.llm_model
    assert t.input_tokens == 100
    assert t.output_tokens == 20
    assert t.latency_ms >= 0  # real wall time


def test_record_accumulates_multiple_calls(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = [
        _fake_response(_payload(), 100, 20),
        _fake_response(_payload(), 200, 30),
    ]
    client = LLMClient(settings=settings, sdk=sdk)

    with client.record() as calls:
        for _ in range(2):
            client.complete_structured(
                system="s", user="u", cached_context=None,
                schema=EvaluationResult,
                tool_name="submit_evaluation", tool_description="d",
            )

    assert [(c.input_tokens, c.output_tokens) for c in calls] == [(100, 20), (200, 30)]


def test_record_isolates_blocks(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.side_effect = [
        _fake_response(_payload(), 100, 20),
        _fake_response(_payload(), 200, 30),
    ]
    client = LLMClient(settings=settings, sdk=sdk)

    with client.record() as a:
        client.complete_structured(
            system="s", user="u", cached_context=None,
            schema=EvaluationResult,
            tool_name="submit_evaluation", tool_description="d",
        )
    with client.record() as b:
        client.complete_structured(
            system="s", user="u", cached_context=None,
            schema=EvaluationResult,
            tool_name="submit_evaluation", tool_description="d",
        )

    assert len(a) == 1 and len(b) == 1
    assert a[0].input_tokens == 100
    assert b[0].input_tokens == 200


def test_record_outside_block_does_not_record(settings: Settings) -> None:
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = _fake_response(_payload(), 100, 20)
    client = LLMClient(settings=settings, sdk=sdk)
    # No exception, and no recording state lingers.
    client.complete_structured(
        system="s", user="u", cached_context=None,
        schema=EvaluationResult,
        tool_name="submit_evaluation", tool_description="d",
    )
    with client.record() as calls:
        client.complete_structured(
            system="s", user="u", cached_context=None,
            schema=EvaluationResult,
            tool_name="submit_evaluation", tool_description="d",
        )
    assert len(calls) == 1  # not 2 — first call was before record() started


def test_record_rejects_nested_entry(settings: Settings) -> None:
    sdk = MagicMock()
    client = LLMClient(settings=settings, sdk=sdk)
    with client.record():
        with pytest.raises(RuntimeError, match="nested"):
            with client.record():
                pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_llm_client_record.py -v`
Expected: ImportError on `CallTelemetry` / `record` — not yet defined.

- [ ] **Step 3: Add the implementation**

Modify `src/jobpilot/llm/client.py`. At the top, add the new imports + dataclass:

```python
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, TypeVar
```

(Replace the existing `from typing import Any, TypeVar` line accordingly.)

After the `_RETRY_EXC` line and before `class LLMClient:`, add:

```python
@dataclass(frozen=True)
class CallTelemetry:
    """Per-LLM-call usage and latency, captured by LLMClient.record()."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
```

Inside `LLMClient.__init__`, add a recording slot:

```python
def __init__(self, *, settings: Settings, sdk: Any | None = None) -> None:
    self._settings = settings
    self._sdk: Any = sdk or OpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )
    self._active_recording: list[CallTelemetry] | None = None
```

Add the context manager method below `__init__`:

```python
@contextmanager
def record(self) -> Iterator[list[CallTelemetry]]:
    """Capture per-call telemetry for the duration of the block.

    Nested entry is not supported in item 1 and raises RuntimeError.
    """
    if self._active_recording is not None:
        raise RuntimeError("LLMClient.record() blocks cannot be nested")
    bucket: list[CallTelemetry] = []
    self._active_recording = bucket
    try:
        yield bucket
    finally:
        self._active_recording = None
```

Modify `complete_structured` to time the SDK call and append telemetry when a recording is active. Find the `log.debug("llm.request", ...)` + `response = self._sdk.chat.completions.create(...)` block and replace with:

```python
log.debug("llm.request", model=self._settings.llm_model, tool=tool_name)
t0 = time.monotonic()
response = self._sdk.chat.completions.create(
    model=self._settings.llm_model,
    max_tokens=max_tokens,
    messages=messages,
    tools=[tool],
    tool_choice={"type": "function", "function": {"name": tool_name}},
)
latency_ms = (time.monotonic() - t0) * 1000.0

if self._active_recording is not None:
    usage = getattr(response, "usage", None)
    self._active_recording.append(
        CallTelemetry(
            model=self._settings.llm_model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
        )
    )
```

(Keep the existing tool-call parsing block below this — only the request line and the recording append are new.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm_client_record.py tests/unit/test_llm_client.py -v`
Expected: all tests pass. (The existing `test_llm_client.py` should still pass — we didn't change the public contract.)

- [ ] **Step 5: Run the full suite + types + lint**

Run: `uv run pytest -q && uv run mypy src && uv run ruff check .`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/jobpilot/llm/client.py tests/unit/test_llm_client_record.py
git commit -m "feat(llm): add CallTelemetry + LLMClient.record() context manager"
```

---

## Task 3: Eval fixtures loader (`evals/fixtures.py`)

**Files:**
- Create: `src/jobpilot/evals/__init__.py`
- Create: `src/jobpilot/evals/fixtures.py`
- Create: `tests/fixtures/evals.py` (test helper builders)
- Test: `tests/unit/test_evals_fixtures.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_evals_fixtures.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from jobpilot.evals.fixtures import EvalCase, ExpectedOutcome, load_eval_set


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_eval_set_returns_cases_with_jds_and_labels(tmp_path: Path) -> None:
    _write(tmp_path / "jobs/a.txt", "JD A body")
    _write(tmp_path / "labels/a.yaml", "expected_decision: apply\nexpected_score_band: [70, 90]\n")

    cases = load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")

    assert len(cases) == 1
    case = cases[0]
    assert case.stem == "a"
    assert case.jd.body == "JD A body"
    assert case.jd.source == "a.txt"
    assert case.expected.expected_decision == "apply"
    assert case.expected.expected_score_band == (70, 90)


def test_load_eval_set_defaults_optional_fields(tmp_path: Path) -> None:
    _write(tmp_path / "jobs/a.txt", "JD")
    _write(tmp_path / "labels/a.yaml", "expected_decision: skip\n")

    cases = load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")

    e = cases[0].expected
    assert e.expected_decision == "skip"
    assert e.expected_score_band is None
    assert e.key_evidence_chunk_substrings == []
    assert e.required_risk_flags == []
    assert e.disallowed_risk_flags == []
    assert e.notes == ""


def test_load_eval_set_skips_jds_without_labels(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write(tmp_path / "jobs/a.txt", "JD A")
    _write(tmp_path / "jobs/b.txt", "JD B")
    _write(tmp_path / "labels/a.yaml", "expected_decision: apply\n")
    # No labels/b.yaml — should warn and skip, not raise.

    cases = load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")

    assert [c.stem for c in cases] == ["a"]


def test_load_eval_set_rejects_invalid_yaml(tmp_path: Path) -> None:
    _write(tmp_path / "jobs/a.txt", "JD")
    _write(tmp_path / "labels/a.yaml", "expected_decision: not_a_decision\n")
    with pytest.raises(ValueError, match="a.yaml"):
        load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")


def test_score_band_must_be_two_ints(tmp_path: Path) -> None:
    _write(tmp_path / "jobs/a.txt", "JD")
    _write(tmp_path / "labels/a.yaml", "expected_decision: apply\nexpected_score_band: [70]\n")
    with pytest.raises(ValueError, match="score_band"):
        load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")


def test_expected_outcome_model_directly() -> None:
    e = ExpectedOutcome(expected_decision="apply", expected_score_band=[80, 95])  # type: ignore[arg-type]
    assert e.expected_score_band == (80, 95)
    assert isinstance(e, ExpectedOutcome)


def test_load_eval_set_sorts_by_stem(tmp_path: Path) -> None:
    for stem in ("c", "a", "b"):
        _write(tmp_path / f"jobs/{stem}.txt", stem)
        _write(tmp_path / f"labels/{stem}.yaml", "expected_decision: apply\n")
    cases = load_eval_set(jobs_glob=str(tmp_path / "jobs/*.txt"), labels_dir=tmp_path / "labels")
    assert [c.stem for c in cases] == ["a", "b", "c"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_evals_fixtures.py -v`
Expected: ModuleNotFoundError on `jobpilot.evals.fixtures`.

- [ ] **Step 3: Create the package + module**

Create `src/jobpilot/evals/__init__.py` (empty file).

Create `src/jobpilot/evals/fixtures.py`:

```python
"""Load JD + label pairs into typed EvalCase objects for the harness."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jobpilot.logging_setup import get_logger
from jobpilot.models.schemas import Decision, JobDescription

log = get_logger(__name__)


class ExpectedOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    expected_decision: Decision
    expected_score_band: tuple[int, int] | None = None
    key_evidence_chunk_substrings: list[str] = Field(default_factory=list)
    required_risk_flags: list[str] = Field(default_factory=list)
    disallowed_risk_flags: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("expected_score_band", mode="before")
    @classmethod
    def _coerce_band(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        raise ValueError("expected_score_band must be a 2-element [min, max] list of ints")


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    stem: str
    jd: JobDescription
    expected: ExpectedOutcome


def load_eval_set(*, jobs_glob: str, labels_dir: Path) -> list[EvalCase]:
    """Load every JD in `jobs_glob` paired with `labels_dir/<stem>.yaml`.

    JDs without a matching label file are skipped with a warning. Bad YAML or
    invalid schema raises ValueError mentioning the offending file.
    """
    cases: list[EvalCase] = []
    for jd_path_str in sorted(glob.glob(jobs_glob)):
        jd_path = Path(jd_path_str)
        stem = jd_path.stem
        label_path = labels_dir / f"{stem}.yaml"
        if not label_path.exists():
            log.warning("eval.label_missing", stem=stem, label_path=str(label_path))
            continue
        try:
            data = yaml.safe_load(label_path.read_text(encoding="utf-8")) or {}
            expected = ExpectedOutcome.model_validate(data)
        except Exception as exc:
            raise ValueError(f"invalid label file {label_path.name}: {exc}") from exc
        jd = JobDescription(source=jd_path.name, body=jd_path.read_text(encoding="utf-8"))
        cases.append(EvalCase(stem=stem, jd=jd, expected=expected))
    return cases
```

Create `tests/fixtures/evals.py` (test helper builders — used by later tasks):

```python
"""Builders for eval-harness Pydantic objects, kept small + composable."""

from __future__ import annotations

from jobpilot.evals.fixtures import EvalCase, ExpectedOutcome
from jobpilot.models.schemas import Decision, JobDescription


def make_expected(
    *,
    decision: Decision = "apply",
    score_band: tuple[int, int] | None = None,
    key_evidence: list[str] | None = None,
    required_flags: list[str] | None = None,
    disallowed_flags: list[str] | None = None,
) -> ExpectedOutcome:
    return ExpectedOutcome(
        expected_decision=decision,
        expected_score_band=score_band,
        key_evidence_chunk_substrings=key_evidence or [],
        required_risk_flags=required_flags or [],
        disallowed_risk_flags=disallowed_flags or [],
    )


def make_eval_case(
    stem: str = "case_1",
    *,
    jd_body: str = "Some JD",
    decision: Decision = "apply",
    score_band: tuple[int, int] | None = None,
    **expected_kwargs,
) -> EvalCase:
    return EvalCase(
        stem=stem,
        jd=JobDescription(source=f"{stem}.txt", body=jd_body),
        expected=make_expected(decision=decision, score_band=score_band, **expected_kwargs),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_evals_fixtures.py -v`
Expected: 7 tests pass.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/jobpilot/evals/__init__.py src/jobpilot/evals/fixtures.py tests/unit/test_evals_fixtures.py tests/fixtures/evals.py
git commit -m "feat(evals): fixtures loader (EvalCase, ExpectedOutcome, load_eval_set)"
```

---

## Task 4: Pricing config (`evals/pricing.py`)

**Files:**
- Create: `src/jobpilot/evals/pricing.py`
- Test: `tests/unit/test_evals_pricing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_evals_pricing.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from jobpilot.evals.pricing import PriceTable, cost, load_prices


def _write_prices(tmp_path: Path) -> Path:
    p = tmp_path / "model_prices.yaml"
    p.write_text(
        """
"claude-sonnet-4-6":
  input: 0.003
  output: 0.015
"claude-haiku-4-5-20251001":
  input: 0.001
  output: 0.005
""",
        encoding="utf-8",
    )
    return p


def test_load_prices_returns_typed_table(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    assert isinstance(prices, PriceTable)
    assert prices.usd_per_1k("claude-sonnet-4-6") == (0.003, 0.015)


def test_cost_for_known_model(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    # 1000 input @ $0.003/1k = $0.003; 1000 output @ $0.015/1k = $0.015; total $0.018
    assert cost("claude-sonnet-4-6", 1000, 1000, prices=prices) == pytest.approx(0.018)


def test_cost_handles_zero_tokens(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    assert cost("claude-sonnet-4-6", 0, 0, prices=prices) == 0.0


def test_cost_unknown_model_returns_none(tmp_path: Path) -> None:
    prices = load_prices(_write_prices(tmp_path))
    assert cost("unknown-model-xyz", 1000, 1000, prices=prices) is None


def test_load_prices_rejects_negative(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text('"m":\n  input: -0.001\n  output: 0.01\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be >= 0"):
        load_prices(p)


def test_load_prices_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prices(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_evals_pricing.py -v`
Expected: ModuleNotFoundError on `jobpilot.evals.pricing`.

- [ ] **Step 3: Implement `pricing.py`**

Create `src/jobpilot/evals/pricing.py`:

```python
"""Model pricing config: tokens → USD."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: float = Field(ge=0)
    output: float = Field(ge=0)

    @field_validator("input", "output")
    @classmethod
    def _check_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be >= 0")
        return v


class PriceTable(BaseModel):
    model_config = ConfigDict(frozen=True)

    prices: dict[str, ModelPrice]

    def usd_per_1k(self, model: str) -> tuple[float, float] | None:
        mp = self.prices.get(model)
        return (mp.input, mp.output) if mp else None


def load_prices(path: Path) -> PriceTable:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    parsed: dict[str, ModelPrice] = {}
    for model, entry in raw.items():
        parsed[model] = ModelPrice.model_validate(entry)
    return PriceTable(prices=parsed)


def cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    prices: PriceTable,
) -> float | None:
    per_1k = prices.usd_per_1k(model)
    if per_1k is None:
        return None
    in_price, out_price = per_1k
    return (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_evals_pricing.py -v`
Expected: 6 tests pass.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/jobpilot/evals/pricing.py tests/unit/test_evals_pricing.py
git commit -m "feat(evals): pricing config (load_prices, cost)"
```

---

## Task 5: Author starter label files + model_prices.yaml + evals/README.md

**Files:**
- Create: `evals/labels/canva_fullstack.yaml`
- Create: `evals/labels/openai_backend.yaml`
- Create: `evals/labels/anthropic_research_eng.yaml`
- Create: `evals/model_prices.yaml`
- Create: `evals/README.md`

No tests in this task — it's data + docs. The fixtures loader from Task 3 already validates these YAMLs on load.

- [ ] **Step 1: Read each JD to draft an `expected_decision`**

Run: `cat evals/jobs/*.txt`

Use this rubric to assign decisions (only `expected_decision` is required — leave score band + evidence lists empty in the starter labels; the bootstrap workflow fills them in after the first run):

| JD | Owner profile context | Starter decision rationale |
|---|---|---|
| `canva_fullstack.txt` | Owner has TypeScript + Python + AWS + RAG/LangGraph experience; JD asks for exactly that | `apply` |
| `openai_backend.txt` | Read it. If owner profile clearly matches → `apply`; if mid-level/uncertain → `maybe` | author judgment |
| `anthropic_research_eng.txt` | Read it. Likely `maybe` (research roles often want ML pubs the owner may not have) or `apply` | author judgment |

If unsure on any JD, default to `maybe`. The user will correct during spec review.

- [ ] **Step 2: Write the three label files**

`evals/labels/canva_fullstack.yaml`:

```yaml
# Starter label. Score band, evidence, and risk-flag substrings are filled in
# after the first eval-batch run (see evals/README.md bootstrap workflow).
expected_decision: apply
notes: "Starter label — TypeScript + Python + AWS + RAG match; owner verifies after first run."
```

`evals/labels/openai_backend.yaml`:

```yaml
expected_decision: <your draft>   # apply | maybe | skip — see Task 5 Step 1
notes: "Starter label — owner verifies after first run."
```

`evals/labels/anthropic_research_eng.yaml`:

```yaml
expected_decision: <your draft>
notes: "Starter label — owner verifies after first run."
```

Replace `<your draft>` with the chosen decision string. Do NOT commit the placeholder.

- [ ] **Step 3: Write `evals/model_prices.yaml`**

```yaml
# USD per 1k tokens. Key is the LiteLLM model id (matches LLM_MODEL in .env).
# Add new models here as you start using them; unknown models report cost = null.
"claude-sonnet-4-6":
  input: 0.003
  output: 0.015
"claude-haiku-4-5-20251001":
  input: 0.001
  output: 0.005
"claude-opus-4-7":
  input: 0.015
  output: 0.075
```

(Prices are approximate placeholders — owner should verify against current Anthropic pricing at first use. Wrong numbers don't break anything; they just make `total_usd` directionally wrong.)

- [ ] **Step 4: Write `evals/README.md`**

```markdown
# evals/ — JobPilot evaluation harness

This directory holds the labelled fixture set + run artifacts for `jobpilot eval-batch`.

## Layout

- `jobs/<stem>.txt` — JD bodies, one per file. Git-tracked.
- `labels/<stem>.yaml` — expected outcomes paired by filename stem. Git-tracked.
- `model_prices.yaml` — USD per 1k tokens, keyed by LiteLLM model id. Git-tracked.
- `runs/<ISO-timestamp>/` — run outputs (`results.jsonl`, `report.md`, `meta.yaml`). **Gitignored.**
- `runs/_baseline_<label>/` — frozen baselines, committed by exception (the leading underscore is the "track this" marker; see `.gitignore`).

## Label file format

Only `expected_decision` is required. All other fields are optional; omitted/empty fields make the corresponding metric N/A for that case.

```yaml
expected_decision: apply              # apply | maybe | skip (REQUIRED)
expected_score_band: [70, 90]         # [min, max] inclusive
key_evidence_chunk_substrings:        # at least one must appear in cited chunks
  - "RAG"
required_risk_flags: []               # substrings that MUST appear in eval.risk_flags
disallowed_risk_flags: []             # substrings that MUST NOT appear in eval.risk_flags
notes: ""                             # human notes; not scored
```

## Bootstrap workflow

1. Run `uv run jobpilot eval-batch` against the starter labels.
2. Open `evals/runs/<latest>/report.md`. For each JD where the model's score "looks right", copy it into `expected_score_band = [score - 10, score + 10]`.
3. Where the model cites garbage, add `key_evidence_chunk_substrings` listing words that should appear in the right evidence.
4. Where the model hallucinates a gap, add `disallowed_risk_flags`. Where it misses an obvious one, add `required_risk_flags`.
5. Re-run. Iterate until labels reflect what you actually want the model to do.
6. Add new JDs by dropping `<stem>.txt` in `jobs/` and `<stem>.yaml` in `labels/` (minimum: `expected_decision`). Target 15–30 JDs over time.

## Freezing a baseline

```bash
cp -r evals/runs/<latest> evals/runs/_baseline_v1
git add evals/runs/_baseline_v1
git commit -m "evals: freeze baseline v1"
```

Then compare future runs with `--baseline evals/runs/_baseline_v1/results.jsonl`.

## Commands

```bash
# Default: run all evals/jobs/*.txt with current LLM_MODEL + prompt v1
uv run jobpilot eval-batch

# Override model or prompt version
uv run jobpilot eval-batch --model claude-haiku-4-5-20251001 --prompt-version v2

# Compare against a baseline
uv run jobpilot eval-batch --baseline evals/runs/_baseline_v1/results.jsonl

# Also run the tailor (4× slower; off by default)
uv run jobpilot eval-batch --tailor

# CI gate: fail if decision_accuracy drops below 0.8
uv run jobpilot eval-batch --fail-under 0.8
```
```

- [ ] **Step 5: Verify the YAMLs parse**

Run: `uv run python -c "from pathlib import Path; from jobpilot.evals.fixtures import load_eval_set; print(load_eval_set(jobs_glob='evals/jobs/*.txt', labels_dir=Path('evals/labels')))"`

Expected: prints 3 `EvalCase(...)` entries, no errors.

- [ ] **Step 6: Commit**

```bash
git add evals/labels/ evals/model_prices.yaml evals/README.md
git commit -m "feat(evals): starter labels + model_prices.yaml + README"
```

---

## Task 6: Metrics module (`evals/metrics.py`)

**Files:**
- Create: `src/jobpilot/evals/metrics.py`
- Test: `tests/unit/test_evals_metrics.py`

This task introduces the core Pydantic DTOs (`PerCaseMetrics`, `BatchAggregates`, `EvalRecord`, `BatchConfig`, `ScoredBatch`, `ActualOutcome`, `CaseTelemetry`, `CaseError`, `CallTelemetryDTO`) that downstream tasks consume.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_evals_metrics.py`:

```python
from __future__ import annotations

import pytest

from jobpilot.evals.fixtures import ExpectedOutcome
from jobpilot.evals.metrics import (
    ActualOutcome,
    BatchAggregates,
    CallTelemetryDTO,
    CaseTelemetry,
    EvalRecord,
    PerCaseMetrics,
    aggregate,
    score_case,
)


def _actual(decision="apply", score=80, cited=("c1",), risk_flags=()) -> ActualOutcome:
    return ActualOutcome(
        decision=decision,
        score=score,
        reasoning="ok",
        cited_chunk_ids=list(cited),
        risk_flags=list(risk_flags),
    )


def test_score_case_decision_correct() -> None:
    expected = ExpectedOutcome(expected_decision="apply")
    m = score_case(
        expected=expected,
        actual=_actual(decision="apply"),
        retrieved_chunk_ids=["c1"],
        cited_chunk_texts={"c1": "anything"},
    )
    assert m.decision_correct is True


def test_score_case_decision_wrong() -> None:
    expected = ExpectedOutcome(expected_decision="apply")
    m = score_case(
        expected=expected,
        actual=_actual(decision="maybe"),
        retrieved_chunk_ids=["c1"],
        cited_chunk_texts={"c1": "x"},
    )
    assert m.decision_correct is False


def test_score_band_in_and_out() -> None:
    expected = ExpectedOutcome(expected_decision="apply", expected_score_band=(70, 90))  # type: ignore[arg-type]
    m_in = score_case(expected=expected, actual=_actual(score=80),
                      retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    assert m_in.score_in_band is True
    assert m_in.score_abs_error == pytest.approx(0.0)  # midpoint = 80

    m_out = score_case(expected=expected, actual=_actual(score=60),
                       retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    assert m_out.score_in_band is False
    assert m_out.score_abs_error == pytest.approx(20.0)


def test_score_band_na_when_omitted() -> None:
    expected = ExpectedOutcome(expected_decision="apply", expected_score_band=None)
    m = score_case(expected=expected, actual=_actual(score=50),
                   retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    assert m.score_in_band is None
    assert m.score_abs_error is None


def test_citation_evidence_ok_substring_match() -> None:
    expected = ExpectedOutcome(
        expected_decision="apply",
        key_evidence_chunk_substrings=["RAG", "LangGraph"],
    )
    m_pass = score_case(expected=expected, actual=_actual(cited=("c1",)),
                        retrieved_chunk_ids=["c1"],
                        cited_chunk_texts={"c1": "Built a RAG pipeline using LangGraph"})
    assert m_pass.citation_evidence_ok is True

    m_fail = score_case(expected=expected, actual=_actual(cited=("c1",)),
                        retrieved_chunk_ids=["c1"],
                        cited_chunk_texts={"c1": "Kubernetes cluster operations"})
    assert m_fail.citation_evidence_ok is False


def test_citation_evidence_na_when_empty() -> None:
    expected = ExpectedOutcome(expected_decision="apply", key_evidence_chunk_substrings=[])
    m = score_case(expected=expected, actual=_actual(cited=("c1",)),
                   retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "anything"})
    assert m.citation_evidence_ok is None


def test_required_risk_flags() -> None:
    expected = ExpectedOutcome(
        expected_decision="apply", required_risk_flags=["management experience"]
    )
    ok = score_case(expected=expected,
                    actual=_actual(risk_flags=("needs 10+ years management experience",)),
                    retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    fail = score_case(expected=expected,
                      actual=_actual(risk_flags=()),
                      retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    assert ok.required_risk_flags_present is True
    assert fail.required_risk_flags_present is False


def test_disallowed_risk_flags() -> None:
    expected = ExpectedOutcome(
        expected_decision="apply", disallowed_risk_flags=["AWS experience"]
    )
    bad = score_case(expected=expected,
                     actual=_actual(risk_flags=("no AWS experience",)),
                     retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    good = score_case(expected=expected,
                      actual=_actual(risk_flags=()),
                      retrieved_chunk_ids=["c1"], cited_chunk_texts={"c1": "x"})
    assert bad.disallowed_risk_flags_absent is False
    assert good.disallowed_risk_flags_absent is True


def test_cited_chunks_subset_of_retrieved() -> None:
    expected = ExpectedOutcome(expected_decision="apply")
    ok = score_case(expected=expected, actual=_actual(cited=("c1",)),
                    retrieved_chunk_ids=["c1", "c2"], cited_chunk_texts={"c1": "x"})
    bad = score_case(expected=expected, actual=_actual(cited=("ghost",)),
                     retrieved_chunk_ids=["c1"], cited_chunk_texts={})
    assert ok.cited_chunks_retrieved is True
    assert bad.cited_chunks_retrieved is False


def _record(stem: str, *, decision="apply", expected_decision="apply",
            score=80, score_band=(70, 90), error=False) -> EvalRecord:
    expected = ExpectedOutcome(expected_decision=expected_decision, expected_score_band=score_band)  # type: ignore[arg-type]
    if error:
        return EvalRecord(
            stem=stem, ts="2026-05-22T00:00:00Z", prompt_version="v1", model="m",
            expected=expected, actual=None, retrieved_chunk_ids=[],
            metrics=None,
            telemetry=CaseTelemetry(latency_ms=10.0, calls=[], input_tokens=0,
                                    output_tokens=0, estimated_usd=None),
            error=None,  # type: ignore[arg-type] -- assigned below
        ).model_copy(update={"error": __import__("jobpilot.evals.metrics", fromlist=["CaseError"]).CaseError(type="X", message="boom")})
    actual = ActualOutcome(decision=decision, score=score, reasoning="r",
                           cited_chunk_ids=["c1"], risk_flags=[])
    metrics = PerCaseMetrics(
        decision_correct=(decision == expected_decision),
        score_in_band=(score_band[0] <= score <= score_band[1]),
        score_abs_error=abs(score - (score_band[0] + score_band[1]) / 2),
        citation_evidence_ok=None,
        required_risk_flags_present=None,
        disallowed_risk_flags_absent=None,
        cited_chunks_retrieved=True,
    )
    return EvalRecord(
        stem=stem, ts="2026-05-22T00:00:00Z", prompt_version="v1", model="m",
        expected=expected, actual=actual, retrieved_chunk_ids=["c1"],
        metrics=metrics,
        telemetry=CaseTelemetry(
            latency_ms=1000.0,
            calls=[CallTelemetryDTO(model="m", input_tokens=100, output_tokens=20, latency_ms=900.0)],
            input_tokens=100, output_tokens=20, estimated_usd=0.001,
        ),
        error=None,
    )


def test_aggregate_basic_counts() -> None:
    records = [
        _record("a", decision="apply", expected_decision="apply", score=80),
        _record("b", decision="maybe", expected_decision="apply", score=60),
        _record("c", error=True),
    ]
    agg = aggregate(records)
    assert isinstance(agg, BatchAggregates)
    assert agg.n_cases == 3
    assert agg.n_errors == 1
    # decision accuracy excludes errored cases from numerator and denominator
    assert agg.decision_correct_count == 1
    assert agg.decision_total == 2
    assert agg.decision_accuracy == pytest.approx(0.5)


def test_aggregate_score_mae_only_over_banded_cases() -> None:
    r1 = _record("a", score=80)  # in band [70,90], midpoint 80, abs_err 0
    r2 = _record("b", score=60)  # out of band, abs_err 20
    agg = aggregate([r1, r2])
    assert agg.score_mae_n == 2
    assert agg.score_mae == pytest.approx(10.0)


def test_aggregate_p50_p95_latency() -> None:
    records = [_record(f"r{i}", score=80) for i in range(10)]
    # All latencies are 1000.0 in the fixture
    agg = aggregate(records)
    assert agg.p50_latency_ms == pytest.approx(1000.0)
    assert agg.p95_latency_ms == pytest.approx(1000.0)


def test_aggregate_total_usd_sums_with_unknown_model_safe() -> None:
    r1 = _record("a", score=80)
    # mutate one record to have unknown-model cost
    r2 = _record("b", score=80).model_copy(
        update={"telemetry": _record("b", score=80).telemetry.model_copy(update={"estimated_usd": None})}
    )
    agg = aggregate([r1, r2])
    # None entries are skipped; sum is the one known cost
    assert agg.total_usd == pytest.approx(0.001)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_evals_metrics.py -v`
Expected: ModuleNotFoundError on `jobpilot.evals.metrics`.

- [ ] **Step 3: Implement `metrics.py`**

Create `src/jobpilot/evals/metrics.py`:

```python
"""Eval-harness Pydantic DTOs + per-case and aggregate scoring."""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jobpilot.evals.fixtures import ExpectedOutcome
from jobpilot.models.schemas import Decision


class CallTelemetryDTO(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class CaseTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)
    latency_ms: float
    calls: list[CallTelemetryDTO] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    estimated_usd: float | None


class CaseError(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: str
    message: str


class ActualOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    decision: Decision
    score: int
    reasoning: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class PerCaseMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)
    decision_correct: bool
    score_in_band: bool | None
    score_abs_error: float | None
    citation_evidence_ok: bool | None
    required_risk_flags_present: bool | None
    disallowed_risk_flags_absent: bool | None
    cited_chunks_retrieved: bool


class EvalRecord(BaseModel):
    """One row in `results.jsonl`."""
    model_config = ConfigDict(frozen=True)

    stem: str
    ts: str
    prompt_version: str
    model: str
    expected: ExpectedOutcome
    actual: ActualOutcome | None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    metrics: PerCaseMetrics | None
    telemetry: CaseTelemetry
    error: CaseError | None


class BatchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: str
    ts: str
    model: str
    prompt_version: str
    settings: dict[str, Any] = Field(default_factory=dict)
    git_sha: str | None = None


class BatchAggregates(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_cases: int
    n_errors: int
    decision_accuracy: float
    decision_correct_count: int
    decision_total: int
    score_mae: float | None
    score_mae_n: int
    score_in_band_rate: float | None
    score_in_band_count: int
    score_band_total: int
    citation_evidence_pass_rate: float | None
    citation_evidence_count: int
    citation_evidence_total: int
    required_risk_flags_pass_rate: float | None
    required_risk_flags_count: int
    required_risk_flags_total: int
    disallowed_risk_flags_pass_rate: float | None
    disallowed_risk_flags_count: int
    disallowed_risk_flags_total: int
    p50_latency_ms: float
    p95_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_usd: float | None
    mean_usd_per_case: float | None


class ScoredBatch(BaseModel):
    model_config = ConfigDict(frozen=True)
    config: BatchConfig
    records: list[EvalRecord]
    aggregates: BatchAggregates


# ---- scoring ----------------------------------------------------------------


def score_case(
    *,
    expected: ExpectedOutcome,
    actual: ActualOutcome,
    retrieved_chunk_ids: list[str],
    cited_chunk_texts: dict[str, str],
) -> PerCaseMetrics:
    """Compute deterministic per-case metrics. See spec §7.1."""
    decision_correct = actual.decision == expected.expected_decision

    if expected.expected_score_band is not None:
        lo, hi = expected.expected_score_band
        midpoint = (lo + hi) / 2.0
        score_in_band: bool | None = lo <= actual.score <= hi
        score_abs_error: float | None = abs(actual.score - midpoint)
    else:
        score_in_band = None
        score_abs_error = None

    if expected.key_evidence_chunk_substrings:
        cited_joined = " ".join(cited_chunk_texts.values()).lower()
        citation_evidence_ok: bool | None = any(
            sub.lower() in cited_joined for sub in expected.key_evidence_chunk_substrings
        )
    else:
        citation_evidence_ok = None

    flags_joined = " ".join(actual.risk_flags).lower()
    if expected.required_risk_flags:
        required_present: bool | None = all(
            r.lower() in flags_joined for r in expected.required_risk_flags
        )
    else:
        required_present = None

    if expected.disallowed_risk_flags:
        disallowed_absent: bool | None = not any(
            d.lower() in flags_joined for d in expected.disallowed_risk_flags
        )
    else:
        disallowed_absent = None

    cited_chunks_retrieved = set(actual.cited_chunk_ids).issubset(set(retrieved_chunk_ids))

    return PerCaseMetrics(
        decision_correct=decision_correct,
        score_in_band=score_in_band,
        score_abs_error=score_abs_error,
        citation_evidence_ok=citation_evidence_ok,
        required_risk_flags_present=required_present,
        disallowed_risk_flags_absent=disallowed_absent,
        cited_chunks_retrieved=cited_chunks_retrieved,
    )


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. Returns 0.0 for empty input."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def aggregate(records: list[EvalRecord]) -> BatchAggregates:
    """Aggregate per-case records into batch-level metrics. See spec §7.2."""
    n_cases = len(records)
    errored = [r for r in records if r.error is not None]
    ok = [r for r in records if r.error is None and r.metrics is not None]
    n_errors = len(errored)

    decision_correct_count = sum(1 for r in ok if r.metrics and r.metrics.decision_correct)
    decision_total = len(ok)
    decision_accuracy = (decision_correct_count / decision_total) if decision_total else 0.0

    score_records = [r for r in ok if r.metrics and r.metrics.score_in_band is not None]
    score_in_band_count = sum(1 for r in score_records if r.metrics and r.metrics.score_in_band)
    score_band_total = len(score_records)
    score_in_band_rate = (score_in_band_count / score_band_total) if score_band_total else None
    score_mae_values = [r.metrics.score_abs_error for r in score_records
                        if r.metrics and r.metrics.score_abs_error is not None]
    score_mae = (sum(score_mae_values) / len(score_mae_values)) if score_mae_values else None

    def _flag_pass_rate(attr: str) -> tuple[float | None, int, int]:
        rs = [r for r in ok if r.metrics and getattr(r.metrics, attr) is not None]
        passes = sum(1 for r in rs if r.metrics and getattr(r.metrics, attr))
        total = len(rs)
        rate = (passes / total) if total else None
        return rate, passes, total

    cit_rate, cit_pass, cit_tot = _flag_pass_rate("citation_evidence_ok")
    req_rate, req_pass, req_tot = _flag_pass_rate("required_risk_flags_present")
    dis_rate, dis_pass, dis_tot = _flag_pass_rate("disallowed_risk_flags_absent")

    latencies = [r.telemetry.latency_ms for r in records]
    p50 = _percentile(latencies, 0.5)
    p95 = _percentile(latencies, 0.95)

    total_input_tokens = sum(r.telemetry.input_tokens for r in records)
    total_output_tokens = sum(r.telemetry.output_tokens for r in records)
    known_costs = [r.telemetry.estimated_usd for r in records if r.telemetry.estimated_usd is not None]
    total_usd = sum(known_costs) if known_costs else None
    mean_usd = (total_usd / len(known_costs)) if known_costs else None

    return BatchAggregates(
        n_cases=n_cases,
        n_errors=n_errors,
        decision_accuracy=decision_accuracy,
        decision_correct_count=decision_correct_count,
        decision_total=decision_total,
        score_mae=score_mae,
        score_mae_n=score_band_total,
        score_in_band_rate=score_in_band_rate,
        score_in_band_count=score_in_band_count,
        score_band_total=score_band_total,
        citation_evidence_pass_rate=cit_rate,
        citation_evidence_count=cit_pass,
        citation_evidence_total=cit_tot,
        required_risk_flags_pass_rate=req_rate,
        required_risk_flags_count=req_pass,
        required_risk_flags_total=req_tot,
        disallowed_risk_flags_pass_rate=dis_rate,
        disallowed_risk_flags_count=dis_pass,
        disallowed_risk_flags_total=dis_tot,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_usd=total_usd,
        mean_usd_per_case=mean_usd,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_evals_metrics.py -v`
Expected: all tests pass.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/jobpilot/evals/metrics.py tests/unit/test_evals_metrics.py
git commit -m "feat(evals): per-case + aggregate metrics with Pydantic DTOs"
```

---

## Task 7: Report module — single-run JSONL + Markdown + meta.yaml

**Files:**
- Create: `src/jobpilot/evals/report.py`
- Test: `tests/unit/test_evals_report.py`

This task implements the single-run path. Baseline-aware extension lands in Task 9.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_evals_report.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from jobpilot.evals.fixtures import ExpectedOutcome
from jobpilot.evals.metrics import (
    ActualOutcome,
    BatchAggregates,
    BatchConfig,
    CallTelemetryDTO,
    CaseTelemetry,
    EvalRecord,
    PerCaseMetrics,
    ScoredBatch,
)
from jobpilot.evals.report import (
    write_meta_yaml,
    write_report_md,
    write_results_jsonl,
)


def _record(stem: str, *, decision="apply", expected_decision="apply", score=80,
            error=False) -> EvalRecord:
    expected = ExpectedOutcome(expected_decision=expected_decision, expected_score_band=(70, 90))  # type: ignore[arg-type]
    base = dict(
        stem=stem, ts="2026-05-22T00:00:00Z", prompt_version="v1", model="m",
        expected=expected, retrieved_chunk_ids=["c1"],
        telemetry=CaseTelemetry(
            latency_ms=1000.0,
            calls=[CallTelemetryDTO(model="m", input_tokens=100, output_tokens=20, latency_ms=900.0)],
            input_tokens=100, output_tokens=20, estimated_usd=0.001,
        ),
    )
    if error:
        from jobpilot.evals.metrics import CaseError
        return EvalRecord(**base, actual=None, metrics=None,
                          error=CaseError(type="RuntimeError", message="boom"))
    actual = ActualOutcome(decision=decision, score=score, reasoning="r",
                           cited_chunk_ids=["c1"], risk_flags=[])
    metrics = PerCaseMetrics(
        decision_correct=(decision == expected_decision),
        score_in_band=(70 <= score <= 90),
        score_abs_error=abs(score - 80),
        citation_evidence_ok=None,
        required_risk_flags_present=None,
        disallowed_risk_flags_absent=None,
        cited_chunks_retrieved=True,
    )
    return EvalRecord(**base, actual=actual, metrics=metrics, error=None)


def _scored(records: list[EvalRecord]) -> ScoredBatch:
    from jobpilot.evals.metrics import aggregate
    return ScoredBatch(
        config=BatchConfig(run_id="2026-05-22T00-00-00", ts="2026-05-22T00:00:00Z",
                           model="claude-sonnet-4-6", prompt_version="v1",
                           settings={"retrieval_k": 8}, git_sha="deadbeef"),
        records=records,
        aggregates=aggregate(records),
    )


def test_write_results_jsonl_one_row_per_record(tmp_path: Path) -> None:
    scored = _scored([_record("a", score=80), _record("b", score=60)])
    out = tmp_path / "results.jsonl"
    write_results_jsonl(scored, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert {r["stem"] for r in rows} == {"a", "b"}
    assert rows[0]["model"] == "claude-sonnet-4-6"
    assert rows[0]["telemetry"]["estimated_usd"] == 0.001


def test_write_meta_yaml(tmp_path: Path) -> None:
    scored = _scored([_record("a", score=80)])
    out = tmp_path / "meta.yaml"
    write_meta_yaml(scored, out)
    body = out.read_text(encoding="utf-8")
    assert "run_id: 2026-05-22T00-00-00" in body
    assert "model: claude-sonnet-4-6" in body
    assert "n_cases: 1" in body


def test_report_md_contains_summary(tmp_path: Path) -> None:
    scored = _scored([_record("a", score=80), _record("b", decision="maybe", score=60)])
    out = tmp_path / "report.md"
    write_report_md(scored, out, baseline=None)
    body = out.read_text(encoding="utf-8")
    assert "JobPilot Eval" in body
    assert "decision_accuracy" in body
    assert "score_mae" in body
    assert "1/2" in body  # one of two correct


def test_report_md_lists_failures_sorted_by_stem(tmp_path: Path) -> None:
    # Two failing cases: 'z' (score out of band) + 'a' (decision wrong)
    records = [
        _record("z", score=60),                              # out-of-band
        _record("a", decision="maybe"),                      # wrong decision
        _record("b", score=80),                              # passes
    ]
    scored = _scored(records)
    out = tmp_path / "report.md"
    write_report_md(scored, out, baseline=None)
    body = out.read_text(encoding="utf-8")
    # Sorted by stem: 'a' appears before 'z'
    a_idx = body.find("### a ")
    z_idx = body.find("### z ")
    assert 0 < a_idx < z_idx
    assert "### b " not in body  # passing cases not in failures


def test_report_md_lists_errors(tmp_path: Path) -> None:
    scored = _scored([_record("err", error=True), _record("ok", score=80)])
    out = tmp_path / "report.md"
    write_report_md(scored, out, baseline=None)
    body = out.read_text(encoding="utf-8")
    assert "Errors (1)" in body
    assert "err" in body
    assert "RuntimeError" in body


def test_report_md_no_baseline_section(tmp_path: Path) -> None:
    scored = _scored([_record("a", score=80)])
    out = tmp_path / "report.md"
    write_report_md(scored, out, baseline=None)
    body = out.read_text(encoding="utf-8")
    assert "Delta vs baseline" not in body
    assert "Regressions" not in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_evals_report.py -v`
Expected: ModuleNotFoundError on `jobpilot.evals.report`.

- [ ] **Step 3: Implement `report.py`**

Create `src/jobpilot/evals/report.py`:

```python
"""Write run artifacts: results.jsonl, report.md, meta.yaml.

Baseline-aware comparison sections are added in Task 9 (see compare.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from jobpilot.evals.metrics import EvalRecord, ScoredBatch

if TYPE_CHECKING:
    from jobpilot.evals.compare import ComparisonReport


def write_results_jsonl(scored: ScoredBatch, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [r.model_dump_json() for r in scored.records]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_meta_yaml(scored: ScoredBatch, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "run_id": scored.config.run_id,
        "ts": scored.config.ts,
        "model": scored.config.model,
        "prompt_version": scored.config.prompt_version,
        "n_cases": scored.aggregates.n_cases,
        "settings": dict(scored.config.settings),
    }
    if scored.config.git_sha:
        payload["git_sha"] = scored.config.git_sha
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


# ---- Markdown ---------------------------------------------------------------


def _fmt_pct(rate: float | None) -> str:
    return f"{rate * 100:.1f}%" if rate is not None else "n/a"


def _fmt_usd(amount: float | None) -> str:
    return f"${amount:.3f}" if amount is not None else "n/a"


def _is_failure(rec: EvalRecord) -> bool:
    if rec.error is not None or rec.metrics is None:
        return False
    m = rec.metrics
    checks = [
        m.decision_correct,
        m.score_in_band,
        m.citation_evidence_ok,
        m.required_risk_flags_present,
        m.disallowed_risk_flags_absent,
        m.cited_chunks_retrieved,
    ]
    return any(c is False for c in checks)


def _failure_lines(rec: EvalRecord) -> list[str]:
    assert rec.actual is not None and rec.metrics is not None
    out = [f"### {rec.stem}"]
    m = rec.metrics
    out.append(
        f"- expected_decision={rec.expected.expected_decision}, "
        f"got={rec.actual.decision} {'✓' if m.decision_correct else '✗'}"
    )
    if rec.expected.expected_score_band is not None:
        band = rec.expected.expected_score_band
        out.append(
            f"- expected_score_band=[{band[0]},{band[1]}], "
            f"got={rec.actual.score} {'✓' if m.score_in_band else '✗'}"
        )
    if m.citation_evidence_ok is False:
        out.append(f"- citation_evidence_ok=✗ (cited: {', '.join(rec.actual.cited_chunk_ids)})")
    if m.required_risk_flags_present is False:
        out.append(f"- required_risk_flags missing (got: {rec.actual.risk_flags})")
    if m.disallowed_risk_flags_absent is False:
        out.append(f"- disallowed_risk_flags present (got: {rec.actual.risk_flags})")
    if m.cited_chunks_retrieved is False:
        out.append(
            f"- cited_chunks_retrieved=✗ (cited {rec.actual.cited_chunk_ids} not in retrieved)"
        )
    excerpt = (rec.actual.reasoning or "").strip().split("\n", 1)[0][:160]
    out.append(f'- reasoning excerpt: "{excerpt}"')
    return out


def write_report_md(
    scored: ScoredBatch,
    out: Path,
    *,
    baseline: "ScoredBatch | None" = None,
    comparison: "ComparisonReport | None" = None,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    agg = scored.aggregates
    cfg = scored.config

    lines: list[str] = []
    lines.append(f"# JobPilot Eval — {cfg.ts}\n")
    lines.append(
        f"**Config:** model=`{cfg.model}`, prompt_version=`{cfg.prompt_version}`, "
        f"n_cases={agg.n_cases}\n"
    )

    if baseline is not None and comparison is not None:
        from jobpilot.evals.compare import render_comparison_section
        lines.append(render_comparison_section(baseline, scored, comparison))

    # Summary table
    lines.append("## Summary")
    rows = [
        ("decision_accuracy",
         f"{agg.decision_correct_count}/{agg.decision_total} ({_fmt_pct(agg.decision_accuracy)})"),
        ("score_mae",
         f"{agg.score_mae:.1f} (n={agg.score_mae_n})" if agg.score_mae is not None else "n/a"),
        ("score_in_band_rate",
         f"{agg.score_in_band_count}/{agg.score_band_total} ({_fmt_pct(agg.score_in_band_rate)})"
         if agg.score_band_total else "n/a"),
        ("citation_evidence_pass_rate",
         f"{agg.citation_evidence_count}/{agg.citation_evidence_total} "
         f"({_fmt_pct(agg.citation_evidence_pass_rate)})"
         if agg.citation_evidence_total else "n/a"),
        ("required_risk_flags_pass_rate",
         f"{agg.required_risk_flags_count}/{agg.required_risk_flags_total} "
         f"({_fmt_pct(agg.required_risk_flags_pass_rate)})"
         if agg.required_risk_flags_total else "n/a"),
        ("disallowed_risk_flags_pass_rate",
         f"{agg.disallowed_risk_flags_count}/{agg.disallowed_risk_flags_total} "
         f"({_fmt_pct(agg.disallowed_risk_flags_pass_rate)})"
         if agg.disallowed_risk_flags_total else "n/a"),
        ("p50 latency / p95",
         f"{agg.p50_latency_ms / 1000.0:.1f}s / {agg.p95_latency_ms / 1000.0:.1f}s"),
        ("total tokens (in / out)", f"{agg.total_input_tokens:,} / {agg.total_output_tokens:,}"),
        ("total cost", _fmt_usd(agg.total_usd)),
        ("errors", str(agg.n_errors)),
    ]
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Failures, sorted by stem
    failures = sorted([r for r in scored.records if _is_failure(r)], key=lambda r: r.stem)
    lines.append(f"## Failures ({len(failures)})")
    if not failures:
        lines.append("*(none)*")
    for rec in failures:
        lines.extend(_failure_lines(rec))
        lines.append("")

    # Errors
    errors = sorted([r for r in scored.records if r.error is not None], key=lambda r: r.stem)
    lines.append(f"## Errors ({len(errors)})")
    if not errors:
        lines.append("*(none)*")
    else:
        for rec in errors:
            assert rec.error is not None
            lines.append(f"### {rec.stem}")
            lines.append(f"- {rec.error.type}: {rec.error.message}")
            lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_evals_report.py -v`
Expected: all tests pass.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: green. (The `from jobpilot.evals.compare import ...` inside `write_report_md` is wrapped in a TYPE_CHECKING / local import so it doesn't fail when compare.py doesn't exist yet — Task 8 creates it.)

If `mypy` complains about the missing module reference, leave the TYPE_CHECKING import commented until Task 8, then uncomment.

- [ ] **Step 6: Commit**

```bash
git add src/jobpilot/evals/report.py tests/unit/test_evals_report.py
git commit -m "feat(evals): write_results_jsonl, write_meta_yaml, write_report_md (single-run)"
```

---

## Task 8: Compare module (`evals/compare.py`)

**Files:**
- Create: `src/jobpilot/evals/compare.py`
- Test: `tests/unit/test_evals_compare.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_evals_compare.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import yaml

from jobpilot.evals.compare import diff_runs, load_scored_batch
from jobpilot.evals.fixtures import ExpectedOutcome
from jobpilot.evals.metrics import (
    ActualOutcome,
    BatchConfig,
    CaseTelemetry,
    EvalRecord,
    PerCaseMetrics,
    ScoredBatch,
    aggregate,
)


def _rec(stem: str, *, decision="apply", score=80, expected_decision="apply") -> EvalRecord:
    expected = ExpectedOutcome(expected_decision=expected_decision, expected_score_band=(70, 90))  # type: ignore[arg-type]
    actual = ActualOutcome(decision=decision, score=score, reasoning="r",
                           cited_chunk_ids=["c1"], risk_flags=[])
    metrics = PerCaseMetrics(
        decision_correct=(decision == expected_decision),
        score_in_band=(70 <= score <= 90),
        score_abs_error=abs(score - 80),
        citation_evidence_ok=None,
        required_risk_flags_present=None,
        disallowed_risk_flags_absent=None,
        cited_chunks_retrieved=True,
    )
    return EvalRecord(
        stem=stem, ts="2026-05-22T00:00:00Z", prompt_version="v1", model="m",
        expected=expected, actual=actual, retrieved_chunk_ids=["c1"], metrics=metrics,
        telemetry=CaseTelemetry(latency_ms=1.0, calls=[], input_tokens=10,
                                output_tokens=2, estimated_usd=0.0001),
        error=None,
    )


def _scored(records: list[EvalRecord], *, model="m", run_id="rid") -> ScoredBatch:
    return ScoredBatch(
        config=BatchConfig(run_id=run_id, ts="2026-05-22T00:00:00Z", model=model,
                           prompt_version="v1", settings={}, git_sha=None),
        records=records,
        aggregates=aggregate(records),
    )


def test_diff_runs_identical_returns_empty_regressions_and_improvements() -> None:
    a = _scored([_rec("x", score=80)])
    b = _scored([_rec("x", score=80)])
    cmp = diff_runs(a, b)
    assert cmp.regressions == []
    assert cmp.improvements == []
    assert cmp.added_cases == []
    assert cmp.removed_cases == []


def test_diff_runs_detects_regression() -> None:
    baseline = _scored([_rec("x", decision="apply", score=80)])              # pass
    candidate = _scored([_rec("x", decision="maybe", score=60)])             # fail
    cmp = diff_runs(baseline, candidate)
    assert len(cmp.regressions) == 1
    reg = cmp.regressions[0]
    assert reg.stem == "x"
    assert reg.baseline_decision == "apply" and reg.candidate_decision == "maybe"
    assert reg.baseline_score == 80 and reg.candidate_score == 60


def test_diff_runs_detects_improvement() -> None:
    baseline = _scored([_rec("x", decision="maybe", score=60)])              # fail
    candidate = _scored([_rec("x", decision="apply", score=80)])             # pass
    cmp = diff_runs(baseline, candidate)
    assert len(cmp.improvements) == 1


def test_diff_runs_added_and_removed_cases() -> None:
    baseline = _scored([_rec("a", score=80), _rec("b", score=80)])
    candidate = _scored([_rec("b", score=80), _rec("c", score=80)])
    cmp = diff_runs(baseline, candidate)
    assert cmp.added_cases == ["c"]
    assert cmp.removed_cases == ["a"]


def test_load_scored_batch_reads_jsonl_and_meta(tmp_path: Path) -> None:
    run_dir = tmp_path / "rid"
    run_dir.mkdir()
    rec = _rec("x", score=80)
    (run_dir / "results.jsonl").write_text(rec.model_dump_json() + "\n", encoding="utf-8")
    (run_dir / "meta.yaml").write_text(
        yaml.safe_dump({
            "run_id": "rid", "ts": "2026-05-22T00:00:00Z", "model": "m",
            "prompt_version": "v1", "n_cases": 1, "settings": {},
        }, sort_keys=False),
        encoding="utf-8",
    )
    scored = load_scored_batch(run_dir / "results.jsonl")
    assert scored.config.run_id == "rid"
    assert len(scored.records) == 1
    assert scored.aggregates.n_cases == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_evals_compare.py -v`
Expected: ModuleNotFoundError on `jobpilot.evals.compare`.

- [ ] **Step 3: Implement `compare.py`**

Create `src/jobpilot/evals/compare.py`:

```python
"""Diff two ScoredBatch runs to surface regressions + improvements."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from jobpilot.evals.metrics import (
    BatchConfig,
    EvalRecord,
    ScoredBatch,
    aggregate,
)


class DeltaRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    metric: str
    baseline: str
    candidate: str
    delta: str


class RegressionRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    stem: str
    baseline_decision: str
    candidate_decision: str
    baseline_score: int
    candidate_score: int


class ImprovementRow(RegressionRow):
    pass


class ComparisonReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    delta_table: list[DeltaRow] = Field(default_factory=list)
    regressions: list[RegressionRow] = Field(default_factory=list)
    improvements: list[ImprovementRow] = Field(default_factory=list)
    added_cases: list[str] = Field(default_factory=list)
    removed_cases: list[str] = Field(default_factory=list)


def _passes(rec: EvalRecord) -> bool:
    if rec.error is not None or rec.metrics is None:
        return False
    m = rec.metrics
    return not any(
        v is False for v in (
            m.decision_correct, m.score_in_band, m.citation_evidence_ok,
            m.required_risk_flags_present, m.disallowed_risk_flags_absent,
            m.cited_chunks_retrieved,
        )
    )


def _fmt_pct(rate: float | None) -> str:
    return f"{rate * 100:.1f}%" if rate is not None else "n/a"


def _fmt_pp_delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{(b - a) * 100:+.1f}pp"


def _fmt_usd(x: float | None) -> str:
    return f"${x:.3f}" if x is not None else "n/a"


def _fmt_usd_delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{b - a:+.3f}"


def diff_runs(baseline: ScoredBatch, candidate: ScoredBatch) -> ComparisonReport:
    base_by_stem = {r.stem: r for r in baseline.records}
    cand_by_stem = {r.stem: r for r in candidate.records}
    common = set(base_by_stem) & set(cand_by_stem)

    regressions: list[RegressionRow] = []
    improvements: list[ImprovementRow] = []
    for stem in sorted(common):
        b = base_by_stem[stem]
        c = cand_by_stem[stem]
        if b.actual is None or c.actual is None:
            continue
        b_pass = _passes(b)
        c_pass = _passes(c)
        row_kwargs = dict(
            stem=stem,
            baseline_decision=b.actual.decision,
            candidate_decision=c.actual.decision,
            baseline_score=b.actual.score,
            candidate_score=c.actual.score,
        )
        if b_pass and not c_pass:
            regressions.append(RegressionRow(**row_kwargs))  # type: ignore[arg-type]
        elif not b_pass and c_pass:
            improvements.append(ImprovementRow(**row_kwargs))  # type: ignore[arg-type]

    ba = baseline.aggregates
    ca = candidate.aggregates
    delta_table = [
        DeltaRow(metric="decision_accuracy",
                 baseline=f"{ba.decision_correct_count}/{ba.decision_total}",
                 candidate=f"{ca.decision_correct_count}/{ca.decision_total}",
                 delta=_fmt_pp_delta(ba.decision_accuracy, ca.decision_accuracy)),
        DeltaRow(metric="score_mae",
                 baseline=f"{ba.score_mae:.1f}" if ba.score_mae is not None else "n/a",
                 candidate=f"{ca.score_mae:.1f}" if ca.score_mae is not None else "n/a",
                 delta=(f"{ca.score_mae - ba.score_mae:+.1f}"
                        if ba.score_mae is not None and ca.score_mae is not None else "n/a")),
        DeltaRow(metric="total_usd",
                 baseline=_fmt_usd(ba.total_usd),
                 candidate=_fmt_usd(ca.total_usd),
                 delta=_fmt_usd_delta(ba.total_usd, ca.total_usd)),
    ]

    added = sorted(set(cand_by_stem) - set(base_by_stem))
    removed = sorted(set(base_by_stem) - set(cand_by_stem))

    return ComparisonReport(
        delta_table=delta_table,
        regressions=regressions,
        improvements=improvements,
        added_cases=added,
        removed_cases=removed,
    )


def load_scored_batch(jsonl_path: Path) -> ScoredBatch:
    """Reconstruct a ScoredBatch from `results.jsonl` + sibling `meta.yaml`."""
    jsonl_path = Path(jsonl_path)
    meta_path = jsonl_path.parent / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    config = BatchConfig(
        run_id=meta.get("run_id", jsonl_path.parent.name),
        ts=meta.get("ts", ""),
        model=meta.get("model", "unknown"),
        prompt_version=meta.get("prompt_version", "unknown"),
        settings=meta.get("settings", {}) or {},
        git_sha=meta.get("git_sha"),
    )
    records: list[EvalRecord] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(EvalRecord.model_validate(json.loads(line)))
    return ScoredBatch(config=config, records=records, aggregates=aggregate(records))


def render_comparison_section(
    baseline: ScoredBatch, candidate: ScoredBatch, cmp: ComparisonReport
) -> str:
    """Markdown block injected into report.md when --baseline is set."""
    bcfg = baseline.config
    lines = [
        f"## Delta vs baseline ({bcfg.ts}, model=`{bcfg.model}`, prompt=`{bcfg.prompt_version}`)",
        "",
        "| metric | baseline | candidate | Δ |",
        "| --- | --- | --- | --- |",
    ]
    for row in cmp.delta_table:
        lines.append(f"| {row.metric} | {row.baseline} | {row.candidate} | {row.delta} |")
    lines.append("")
    lines.append(f"## Regressions ({len(cmp.regressions)})  *(passed in baseline, fail in candidate)*")
    for r in cmp.regressions:
        lines.append(f"### {r.stem}")
        lines.append(f"- baseline: decision={r.baseline_decision}, score={r.baseline_score} ✓")
        lines.append(f"- candidate: decision={r.candidate_decision}, score={r.candidate_score} ✗")
        lines.append(
            f"- diff: decision {r.baseline_decision}→{r.candidate_decision}, "
            f"score {r.baseline_score}→{r.candidate_score}"
        )
        lines.append("")
    if cmp.improvements:
        lines.append(f"## Improvements ({len(cmp.improvements)})")
        for r in cmp.improvements:
            lines.append(f"### {r.stem}")
            lines.append(
                f"- baseline ✗ → candidate ✓ "
                f"({r.baseline_decision}/{r.baseline_score} → {r.candidate_decision}/{r.candidate_score})"
            )
            lines.append("")
    if cmp.added_cases:
        lines.append("## Added cases")
        lines.extend(f"- {s}" for s in cmp.added_cases)
        lines.append("")
    if cmp.removed_cases:
        lines.append("## Removed cases")
        lines.extend(f"- {s}" for s in cmp.removed_cases)
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_evals_compare.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/jobpilot/evals/compare.py tests/unit/test_evals_compare.py
git commit -m "feat(evals): compare module (diff_runs, load_scored_batch, render_comparison_section)"
```

---

## Task 9: Wire `--baseline` into the Markdown report

**Files:**
- Test: `tests/unit/test_evals_report.py` (extend)

`write_report_md` already accepts `baseline` + `comparison` params (Task 7 Step 3) and lazily imports `render_comparison_section` from `compare.py`. This task adds the test that exercises that path and a small fixture builder for it.

- [ ] **Step 1: Add a baseline test to `test_evals_report.py`**

Append to `tests/unit/test_evals_report.py`:

```python
def test_report_md_with_baseline_includes_delta_and_regressions(tmp_path: Path) -> None:
    from jobpilot.evals.compare import diff_runs

    baseline = _scored([_record("x", decision="apply", score=80)])    # pass
    candidate = _scored([_record("x", decision="maybe", score=60)])   # fail
    cmp = diff_runs(baseline, candidate)

    out = tmp_path / "report.md"
    write_report_md(candidate, out, baseline=baseline, comparison=cmp)
    body = out.read_text(encoding="utf-8")
    assert "Delta vs baseline" in body
    assert "Regressions (1)" in body
    assert "decision apply→maybe" in body
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_evals_report.py -v`
Expected: 7 tests pass (the new one + the 6 from Task 7).

- [ ] **Step 3: Lint + types**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_evals_report.py
git commit -m "test(evals): baseline-aware report Markdown rendering"
```

---

## Task 10: Runner module (`evals/runner.py`)

**Files:**
- Create: `src/jobpilot/evals/runner.py`
- Test: `tests/unit/test_evals_runner.py`

The runner invokes the existing production graph per case, captures telemetry via `LLMClient.record()`, and assembles `EvalRecord`s. It does NOT touch CLI or filesystem — that's Task 11.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_evals_runner.py`:

```python
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jobpilot.config import Settings
from jobpilot.evals.fixtures import ExpectedOutcome
from jobpilot.evals.metrics import EvalRecord
from jobpilot.evals.runner import run_batch
from jobpilot.models.schemas import EvaluationResult, JobDescription, ProfileChunk
from jobpilot.models.state import AgentState
from tests.fixtures.evals import make_eval_case


class _StubGraph:
    """Mimics the LangGraph .ainvoke() contract used by orchestrator.build_graph()."""

    def __init__(self, response_by_stem: dict[str, AgentState | Exception]) -> None:
        self._response = response_by_stem

    async def ainvoke(self, state: AgentState) -> AgentState:
        stem = state["job"].source.replace(".txt", "")
        res = self._response[stem]
        if isinstance(res, Exception):
            raise res
        return res


def _state_for(stem: str, *, decision="apply", score=80) -> AgentState:
    chunk = ProfileChunk(id="c1", source="cv.docx", heading_path=["Work"], text="RAG pipeline")
    return AgentState(
        job=JobDescription(source=f"{stem}.txt", body="jd"),
        retrieved=[chunk],
        evaluation=EvaluationResult(score=score, decision=decision, reasoning="r",
                                    cited_chunk_ids=["c1"], risk_flags=[]),
        tailored=None,
        output_paths={},
    )


@pytest.mark.asyncio
async def test_run_batch_assembles_eval_records(settings: Settings) -> None:
    cases = [
        make_eval_case("a", decision="apply", score_band=(70, 90)),
        make_eval_case("b", decision="apply", score_band=(70, 90)),
    ]
    graph = _StubGraph({
        "a": _state_for("a", decision="apply", score=80),
        "b": _state_for("b", decision="maybe", score=60),
    })
    llm = MagicMock()
    llm.record.return_value.__enter__.return_value = []  # no calls recorded
    llm.record.return_value.__exit__.return_value = None

    records = await run_batch(
        cases=cases, graph=graph, llm=llm,  # type: ignore[arg-type]
        prompt_version="v1", model="m",
    )

    assert isinstance(records, list) and all(isinstance(r, EvalRecord) for r in records)
    assert [r.stem for r in records] == ["a", "b"]
    assert records[0].actual is not None and records[0].actual.decision == "apply"
    assert records[1].actual is not None and records[1].actual.score == 60
    assert all(r.error is None for r in records)


@pytest.mark.asyncio
async def test_run_batch_captures_error_and_continues(settings: Settings) -> None:
    cases = [make_eval_case("a"), make_eval_case("b")]
    graph = _StubGraph({
        "a": _state_for("a"),
        "b": RuntimeError("graph blew up"),
    })
    llm = MagicMock()
    llm.record.return_value.__enter__.return_value = []
    llm.record.return_value.__exit__.return_value = None

    records = await run_batch(
        cases=cases, graph=graph, llm=llm,  # type: ignore[arg-type]
        prompt_version="v1", model="m",
    )

    assert records[0].error is None
    assert records[1].error is not None
    assert records[1].error.type == "RuntimeError"
    assert "graph blew up" in records[1].error.message
    assert records[1].actual is None
    assert records[1].metrics is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_evals_runner.py -v`
Expected: ModuleNotFoundError on `jobpilot.evals.runner`.

- [ ] **Step 3: Implement `runner.py`**

Create `src/jobpilot/evals/runner.py`:

```python
"""Batch runner: invoke the production graph per case + capture telemetry."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Protocol

from jobpilot.evals.fixtures import EvalCase
from jobpilot.evals.metrics import (
    ActualOutcome,
    CallTelemetryDTO,
    CaseError,
    CaseTelemetry,
    EvalRecord,
    score_case,
)
from jobpilot.evals.pricing import PriceTable, cost
from jobpilot.llm.client import LLMClient
from jobpilot.logging_setup import get_logger
from jobpilot.models.state import AgentState

log = get_logger(__name__)


class _GraphLike(Protocol):
    async def ainvoke(self, state: AgentState) -> AgentState: ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def run_batch(
    *,
    cases: list[EvalCase],
    graph: _GraphLike,
    llm: LLMClient,
    prompt_version: str,
    model: str,
    prices: PriceTable | None = None,
) -> list[EvalRecord]:
    """Run each case through `graph` and assemble EvalRecord rows."""
    records: list[EvalRecord] = []
    for case in cases:
        ts = _now_iso()
        t0 = time.monotonic()
        try:
            with llm.record() as calls:
                state: AgentState = {"job": case.jd}
                out = await graph.ainvoke(state)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            evaluation = out["evaluation"]
            retrieved = out.get("retrieved") or []
            retrieved_ids = [c.id for c in retrieved]
            cited_texts = {c.id: c.text for c in retrieved if c.id in evaluation.cited_chunk_ids}
            actual = ActualOutcome(
                decision=evaluation.decision,
                score=evaluation.score,
                reasoning=evaluation.reasoning,
                cited_chunk_ids=list(evaluation.cited_chunk_ids),
                risk_flags=list(evaluation.risk_flags),
            )
            metrics = score_case(
                expected=case.expected, actual=actual,
                retrieved_chunk_ids=retrieved_ids, cited_chunk_texts=cited_texts,
            )
            call_dtos = [CallTelemetryDTO(model=c.model, input_tokens=c.input_tokens,
                                          output_tokens=c.output_tokens, latency_ms=c.latency_ms)
                         for c in calls]
            tot_in = sum(c.input_tokens for c in call_dtos)
            tot_out = sum(c.output_tokens for c in call_dtos)
            usd = cost(model, tot_in, tot_out, prices=prices) if prices is not None else None
            telemetry = CaseTelemetry(
                latency_ms=elapsed_ms, calls=call_dtos,
                input_tokens=tot_in, output_tokens=tot_out, estimated_usd=usd,
            )
            records.append(EvalRecord(
                stem=case.stem, ts=ts, prompt_version=prompt_version, model=model,
                expected=case.expected, actual=actual, retrieved_chunk_ids=retrieved_ids,
                metrics=metrics, telemetry=telemetry, error=None,
            ))
            log.info("eval.case_done", stem=case.stem, decision=actual.decision,
                     score=actual.score, latency_ms=elapsed_ms)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            records.append(EvalRecord(
                stem=case.stem, ts=ts, prompt_version=prompt_version, model=model,
                expected=case.expected, actual=None, retrieved_chunk_ids=[], metrics=None,
                telemetry=CaseTelemetry(latency_ms=elapsed_ms, calls=[],
                                        input_tokens=0, output_tokens=0, estimated_usd=None),
                error=CaseError(type=type(exc).__name__, message=str(exc)),
            ))
            log.warning("eval.case_failed", stem=case.stem, error=str(exc))
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_evals_runner.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check src tests && uv run mypy src`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/jobpilot/evals/runner.py tests/unit/test_evals_runner.py
git commit -m "feat(evals): run_batch runner with telemetry capture + error isolation"
```

---

## Task 11: CLI `eval-batch` command + integration test

**Files:**
- Modify: `src/jobpilot/cli.py`
- Test: `tests/integration/test_eval_batch.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_eval_batch.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from jobpilot.cli import app
from jobpilot.config import Settings
from jobpilot.models.schemas import EvaluationResult


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def eval_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Lay out a self-contained evals/ directory + minimal Chroma/profile state."""
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "evals/jobs/a.txt", "JD A — RAG and LangGraph")
    _write(tmp_path / "evals/jobs/b.txt", "JD B — Java backend")
    _write(tmp_path / "evals/labels/a.yaml",
           "expected_decision: apply\nexpected_score_band: [70, 90]\n")
    _write(tmp_path / "evals/labels/b.yaml",
           "expected_decision: apply\nexpected_score_band: [70, 90]\n")
    _write(tmp_path / "evals/model_prices.yaml",
           '"test-model":\n  input: 0.001\n  output: 0.005\n')
    return tmp_path


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, responses: dict[str, EvaluationResult]) -> None:
    """Replace build_graph + LLMClient + RagStore with stubs."""
    from jobpilot import cli
    from jobpilot.models.schemas import ProfileChunk
    from jobpilot.models.state import AgentState

    chunk = ProfileChunk(id="c1", source="cv.docx", heading_path=["Work"],
                         text="Built RAG pipeline using LangGraph")

    class _StubStore:
        def query(self, text: str, *, k: int) -> list[ProfileChunk]:
            return [chunk]

    class _StubGraph:
        async def ainvoke(self, state: AgentState) -> AgentState:
            stem = state["job"].source.replace(".txt", "")
            return AgentState(
                job=state["job"], retrieved=[chunk],
                evaluation=responses[stem], tailored=None, output_paths={},
            )

    def _build_graph_stub(*args: Any, **kwargs: Any) -> _StubGraph:
        return _StubGraph()

    monkeypatch.setattr(cli, "build_graph", _build_graph_stub)
    monkeypatch.setattr(cli, "_build_store", lambda: _StubStore())

    # LLMClient needs record() to work as a context manager.
    real_record = MagicMock()
    real_record.__enter__ = MagicMock(return_value=[])
    real_record.__exit__ = MagicMock(return_value=None)
    llm_stub = MagicMock()
    llm_stub.record.return_value = real_record
    monkeypatch.setattr(cli, "LLMClient", lambda *a, **kw: llm_stub)


def test_eval_batch_writes_jsonl_and_report(eval_workspace: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, {
        "a": EvaluationResult(score=80, decision="apply", reasoning="r",
                              cited_chunk_ids=["c1"], risk_flags=[]),
        "b": EvaluationResult(score=60, decision="maybe", reasoning="r",
                              cited_chunk_ids=["c1"], risk_flags=[]),
    })
    monkeypatch.setenv("LITELLM_BASE_URL", "http://x"); monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    runner = CliRunner()
    result = runner.invoke(app, ["eval-batch"])
    assert result.exit_code == 0, result.output

    run_dirs = list((eval_workspace / "evals/runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    rows = [json.loads(line) for line in
            (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {r["stem"] for r in rows} == {"a", "b"}
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "decision_accuracy" in report
    assert "1/2" in report  # one correct
    meta = (run_dir / "meta.yaml").read_text(encoding="utf-8")
    assert "model: test-model" in meta


def test_eval_batch_fail_under_returns_nonzero(eval_workspace: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, {
        "a": EvaluationResult(score=80, decision="apply", reasoning="r",
                              cited_chunk_ids=["c1"], risk_flags=[]),
        "b": EvaluationResult(score=60, decision="maybe", reasoning="r",
                              cited_chunk_ids=["c1"], risk_flags=[]),
    })
    monkeypatch.setenv("LITELLM_BASE_URL", "http://x"); monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    runner = CliRunner()
    bad = runner.invoke(app, ["eval-batch", "--fail-under", "1.0"])
    good = runner.invoke(app, ["eval-batch", "--fail-under", "0.0"])
    assert bad.exit_code == 1
    assert good.exit_code == 0


def test_eval_batch_with_baseline(eval_workspace: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, {
        "a": EvaluationResult(score=80, decision="apply", reasoning="r",
                              cited_chunk_ids=["c1"], risk_flags=[]),
        "b": EvaluationResult(score=80, decision="apply", reasoning="r",
                              cited_chunk_ids=["c1"], risk_flags=[]),
    })
    monkeypatch.setenv("LITELLM_BASE_URL", "http://x"); monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    runner = CliRunner()
    first = runner.invoke(app, ["eval-batch"])
    assert first.exit_code == 0
    baseline_jsonl = next((eval_workspace / "evals/runs").iterdir()) / "results.jsonl"

    # Second run: candidate where 'b' regresses.
    _patch_pipeline(monkeypatch, {
        "a": EvaluationResult(score=80, decision="apply", reasoning="r",
                              cited_chunk_ids=["c1"], risk_flags=[]),
        "b": EvaluationResult(score=60, decision="maybe", reasoning="r",
                              cited_chunk_ids=["c1"], risk_flags=[]),
    })
    second = runner.invoke(app, ["eval-batch", "--baseline", str(baseline_jsonl)])
    assert second.exit_code == 0
    # Find the *new* run dir
    new_dirs = sorted((eval_workspace / "evals/runs").iterdir())
    report = (new_dirs[-1] / "report.md").read_text(encoding="utf-8")
    assert "Delta vs baseline" in report
    assert "Regressions" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_eval_batch.py -v`
Expected: `No such command 'eval-batch'` — the CLI command doesn't exist yet.

- [ ] **Step 3: Implement the CLI command**

Append to `src/jobpilot/cli.py` (after the existing `run_cmd`):

```python
# ---- eval-batch ----------------------------------------------------------------


@app.command(name="eval-batch")
def eval_batch_cmd(
    jobs_glob: str = typer.Argument("evals/jobs/*.txt"),
    labels_dir: Path = typer.Option(Path("evals/labels"), "--labels-dir"),
    prompt_version: str = typer.Option("v1", "--prompt-version"),
    model: str | None = typer.Option(None, "--model"),
    baseline: Path | None = typer.Option(None, "--baseline", exists=True, readable=True),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    with_tailor: bool = typer.Option(False, "--tailor/--no-tailor"),
    fail_under: float | None = typer.Option(None, "--fail-under"),
    prices_path: Path = typer.Option(Path("evals/model_prices.yaml"), "--prices"),
    concurrency: int = typer.Option(1, "--concurrency", min=1, max=8),
) -> None:
    """Run the evaluator (and optionally tailor) over a labelled fixture set.

    Writes evals/runs/<ts>/{results.jsonl, report.md, meta.yaml}. With --baseline,
    the report also includes a Delta + Regressions section.
    """
    import asyncio
    import subprocess
    from datetime import datetime, timezone

    from jobpilot.evals.compare import diff_runs, load_scored_batch
    from jobpilot.evals.fixtures import load_eval_set
    from jobpilot.evals.metrics import BatchConfig, ScoredBatch, aggregate
    from jobpilot.evals.pricing import load_prices
    from jobpilot.evals.report import (
        write_meta_yaml,
        write_report_md,
        write_results_jsonl,
    )
    from jobpilot.evals.runner import run_batch

    settings = get_settings()
    configure_logging(settings.log_format)
    if model:
        settings = settings.model_copy(update={"llm_model": model})

    cases = load_eval_set(jobs_glob=jobs_glob, labels_dir=labels_dir)
    if not cases:
        typer.echo("No labelled cases found.")
        raise typer.Exit(code=1)

    prices = load_prices(prices_path) if prices_path.exists() else None

    store = _build_store()
    llm = LLMClient(settings=settings)
    evaluator = EvaluatorAgent(settings=settings, llm=llm, rag=store,
                               prompt_version=prompt_version)
    if with_tailor:
        pool = load_bullet_pool(settings.bullet_pool_path)
        tailor = TailorAgent(settings=settings, llm=llm, rag=store, pool=pool)
        graph = build_graph(settings=settings, evaluator=evaluator, tailor=tailor)
    else:
        graph = build_graph(settings=settings, evaluator=evaluator, tailor=None)  # type: ignore[arg-type]

    records = asyncio.run(run_batch(
        cases=cases, graph=graph, llm=llm,
        prompt_version=prompt_version, model=settings.llm_model, prices=prices,
    ))

    ts = datetime.now(timezone.utc)
    run_id = ts.strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = run_dir or (Path("evals/runs") / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        git_sha = None

    scored = ScoredBatch(
        config=BatchConfig(
            run_id=run_id, ts=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            model=settings.llm_model, prompt_version=prompt_version,
            settings={"retrieval_k": settings.retrieval_k,
                      "score_threshold": settings.score_threshold},
            git_sha=git_sha,
        ),
        records=records,
        aggregates=aggregate(records),
    )

    comparison = None
    base_scored = None
    if baseline is not None:
        base_scored = load_scored_batch(baseline)
        comparison = diff_runs(base_scored, scored)

    write_results_jsonl(scored, out_dir / "results.jsonl")
    write_meta_yaml(scored, out_dir / "meta.yaml")
    write_report_md(scored, out_dir / "report.md",
                    baseline=base_scored, comparison=comparison)

    # Terse stdout
    a = scored.aggregates
    typer.echo(f"n_cases={a.n_cases}  errors={a.n_errors}  "
               f"decision_accuracy={a.decision_accuracy:.2f}  "
               f"score_mae={a.score_mae if a.score_mae is None else round(a.score_mae, 1)}  "
               f"total_usd={a.total_usd}")
    typer.echo(f"report: {out_dir / 'report.md'}")

    if fail_under is not None and scored.aggregates.decision_accuracy < fail_under:
        raise typer.Exit(code=1)
```

The `build_graph(...tailor=None)` branch requires the orchestrator to support a `None` tailor (skip the tailor edge entirely). Check `src/jobpilot/agents/orchestrator.py` — if it currently requires a tailor, add a guard so that when `tailor is None`, the graph compiles with only the `evaluate` node and an unconditional edge to END. (Small change inside `build_graph`; do not refactor the orchestrator's public surface.)

- [ ] **Step 4: Inspect and (if needed) patch `orchestrator.build_graph` to accept `tailor=None`**

Run: `cat src/jobpilot/agents/orchestrator.py` and confirm whether `tailor` parameter is optional. If not, modify `build_graph` so that when `tailor is None`, the conditional edge is replaced by `add_edge("evaluate", END)`. Add a unit test in `tests/unit/test_orchestrator.py` for the new path:

```python
def test_build_graph_without_tailor_routes_directly_to_end(settings: Settings) -> None:
    from jobpilot.agents.orchestrator import build_graph
    graph = build_graph(settings=settings, evaluator=MagicMock(), tailor=None)
    # Smoke check: graph has only the evaluator node + END
    assert graph is not None
```

(If `tailor` is already optional, skip this step.)

- [ ] **Step 5: Run the integration test**

Run: `uv run pytest tests/integration/test_eval_batch.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Run the full suite + types + lint**

Run: `uv run pytest -q && uv run mypy src && uv run ruff check .`
Expected: all green, coverage still ≥70%.

- [ ] **Step 7: Commit**

```bash
git add src/jobpilot/cli.py src/jobpilot/agents/orchestrator.py tests/integration/test_eval_batch.py tests/unit/test_orchestrator.py
git commit -m "feat(cli): jobpilot eval-batch — batch runner, JSONL + Markdown + baseline diff"
```

---

## Task 12: First real eval-batch run + bootstrap labels

**Files:**
- Modify: `evals/labels/*.yaml` (fill in `expected_score_band` from real output)

This is a manual bootstrap step, not a code change. It's the first real exercise of the harness against a live LLM + Chroma — exactly the workflow `evals/README.md` documents.

- [ ] **Step 1: Confirm `.chroma/` is populated**

Run: `uv run jobpilot ingest --profile data/profile/`
Expected: prints chunk count. (Skip if already ingested in the working environment.)

- [ ] **Step 2: First batch run**

Run: `uv run jobpilot eval-batch`
Expected: stdout shows `n_cases=3 errors=0 decision_accuracy=... score_mae=None total_usd=...`, plus a `report: evals/runs/<ts>/report.md` line.

- [ ] **Step 3: Inspect the report**

Run: `open evals/runs/<ts>/report.md` (or `cat`).

For each JD where the model's score "looks right":
- Note the actual score.
- Open `evals/labels/<stem>.yaml` and add `expected_score_band: [score-10, score+10]`.

For each JD where the model cites garbage:
- Open the chunk text via `results.jsonl` (`jq .actual.cited_chunk_ids evals/runs/<ts>/results.jsonl`).
- Add `key_evidence_chunk_substrings: ["expected anchor", ...]` to the label.

For each JD where the model misses or hallucinates a risk flag, populate the relevant list.

- [ ] **Step 4: Re-run and confirm metrics now non-N/A**

Run: `uv run jobpilot eval-batch`
Expected: `score_mae` and `score_in_band_rate` are now numeric in the summary table (no longer `n/a`).

- [ ] **Step 5: Freeze the baseline**

```bash
LATEST=$(ls -1d evals/runs/2026-* | tail -1)
cp -r "$LATEST" evals/runs/_baseline_v1
git add evals/labels/ evals/runs/_baseline_v1
git commit -m "evals: bootstrap labels + freeze _baseline_v1"
```

(Confirm `evals/runs/_baseline_v1/` is staged despite the `evals/runs/` gitignore — the `!evals/runs/_baseline_*/` negation should allow it. If `git add` says "ignored", run `git check-ignore -v evals/runs/_baseline_v1/` to debug.)

- [ ] **Step 6: Smoke-test the comparison path**

Run: `uv run jobpilot eval-batch --baseline evals/runs/_baseline_v1/results.jsonl`
Expected: stdout shows accuracy/cost; report contains a "Delta vs baseline" section (likely all-zero deltas since this is the same model + prompt as baseline).

---

## Acceptance check

After Task 12 completes, manually verify:

- [ ] `uv run jobpilot eval-batch` produces a summary table + failure list in `evals/runs/<ts>/report.md`.
- [ ] `--baseline` adds a Regressions section that names cases that flipped from pass to fail.
- [ ] `--fail-under <high-number>` exits non-zero.
- [ ] `--no-tailor` is the default; `--tailor` runs the full graph (no docx written) and increases total_usd.
- [ ] `uv run pytest --cov=src/jobpilot --cov-report=term-missing` shows ≥70% coverage with the new module landing ≥85%.
- [ ] `uv run mypy src` and `uv run ruff check .` are green.
- [ ] No changes to `src/jobpilot/agents/evaluator.py` or `src/jobpilot/agents/tailor.py` apart from any optional tailor-skip wiring inside `build_graph`.

This satisfies the spec's stated acceptance check (§12): *"one command produces a summary table plus failure cases that explain what changed and why it matters."*
