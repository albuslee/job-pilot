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
