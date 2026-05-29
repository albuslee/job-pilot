# JobPilot

Multi-agent job-hunting assistant. Evaluates a job description against your CV,
optionally tailors a CV to match, and lets you rehearse interview answers with
CV-grounded feedback. All personal data stays local.

## Prerequisites

- **Python 3.12** (managed by `uv`)
- **`uv`** — install via `brew install uv` (macOS) or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **An OpenAI-compatible LLM endpoint** — any service that speaks the OpenAI Chat Completions
  protocol with `tools` + `tool_choice` works. The default config points at a local
  **[LiteLLM](https://docs.litellm.ai/)** gateway (`http://localhost:4000/v1`), which routes to
  Anthropic / OpenAI / Bedrock / etc. with one consistent interface. Alternatives:
  - OpenAI direct (`https://api.openai.com/v1`)
  - Anthropic via their OpenAI-compat endpoint
  - Azure OpenAI / Ollama / any other gateway speaking the same protocol
- **An embedding provider** — one of:
  - **Ollama** (default, free, local) — must be running at `http://localhost:11434` with the
    `nomic-embed-text` model pulled (`ollama pull nomic-embed-text`)
  - Voyage (`VOYAGE_API_KEY`)
  - OpenAI (`OPENAI_API_KEY`)

## How it works

```
jobpilot ingest   →   docx → chunk → embed → ChromaDB

jobpilot eval     →   JD ─┐
                          ├─▶ evaluator → EvaluationResult (score 0–100, reasoning)
                  Profile ┘

jobpilot run      →   JD + Profile ─▶ evaluator ─▶ (score ≥ SCORE_THRESHOLD?)
                                                       │
                                                       ├── no → END (no .docx)
                                                       └── yes → tailor ─▶ docx_writer
                                                                                │
                                                                                ▼
                                                            output/<OWNER>_CV_<company>.docx

jobpilot interview →  question ─▶ record (mic) ─▶ Whisper transcript + pitch
                                       │
                                       ▼
                      delivery metrics (WPM, fillers, pauses, pitch/monotone)
                      + RAG(CV chunks) ─▶ coach ─▶ CV-grounded feedback
                                                          │
                                                          ▼
                                          output/interview_<ts>.md
```

The **tailor** never invents bullet text for your current role — it picks IDs from a fixed
`data/bullet_pool.yaml` you maintain. See [Hallucination guardrails](#hallucination-guardrails)
below.

## Quick start

```bash
make setup                          # install deps + git hooks
cp .env.example .env                # fill in LITELLM_*, OWNER_NAME, CURRENT_ROLE_ANCHOR

# Drop your CV(s) into data/profile/ (gitignored), then copy whichever one
# you want as the tailor's structural template:
cp data/profile/<your_cv>.docx data/cv_template.docx

# Seed your bullet pool (gitignored). Two options:
#   (a) Bootstrap from the bullets already in your CV template:
make generate-pool                                  # → data/bullet_pool.yaml
#   (b) Or start from a blank template:
# cp data/bullet_pool.example.yaml data/bullet_pool.yaml
$EDITOR data/bullet_pool.yaml      # rename IDs, add alternative framings

make ingest                         # embed profile into ChromaDB
make eval JD=evals/jobs/canva_fullstack.txt
make run  JD=evals/jobs/canva_fullstack.txt   # evaluate + tailor → output/
```

> `data/profile/`, `data/cv_template.docx`, and `data/bullet_pool.yaml` are
> all gitignored — your CV and work history stay local. Tests build their own
> synthetic templates and pools at runtime and never read these files.

## Configuration

All personal info lives in `.env` (gitignored). Key variables:

| Env var                                              | Default                                              | Purpose                                                                                                                                                                                  |
| ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OWNER_NAME`                                         | `Owner`                                              | Used to build the output filename: `<owner_slug>_CV_<company>.docx`.                                                                                                                     |
| `CURRENT_ROLE_ANCHOR`                                | (empty)                                              | Exact text in your CV that identifies your current role. The tailor uses this to find the bullet block to rewrite. Empty = skip role-bullet rewrite; only summary + skills are tailored. |
| `CV_TEMPLATE_PATH` / `BULLET_POOL_PATH`              | `data/cv_template.docx` / `data/bullet_pool.yaml`    | Override file locations.                                                                                                                                                                 |
| `LITELLM_BASE_URL` / `LITELLM_API_KEY` / `LLM_MODEL` | `http://localhost:4000/v1` / — / `claude-sonnet-4-6` | Any OpenAI-compatible endpoint. Point at LiteLLM, OpenAI, Anthropic-compat, Azure OpenAI, Ollama `/v1`, etc.                                                                             |
| `EMBEDDING_PROVIDER`                                 | `ollama`                                             | `ollama` \| `voyage` \| `openai`.                                                                                                                                                        |
| `OLLAMA_BASE_URL` / `OLLAMA_EMBEDDING_MODEL`         | `http://localhost:11434` / `nomic-embed-text`        | Used when `EMBEDDING_PROVIDER=ollama`.                                                                                                                                                   |
| `SCORE_THRESHOLD`                                    | `70`                                                 | `jobpilot run` only writes a tailored CV when `evaluation.score ≥ this`.                                                                                                                 |
| `RETRIEVAL_K`                                        | `8`                                                  | Number of top-k profile chunks pulled from ChromaDB per query.                                                                                                                           |
| `SMART_DOCX_INGEST`                                  | `false`                                              | Enable opt-in LLM fallback for low-confidence DOCX section parsing. Equivalent to `jobpilot ingest --smart-docx`.                                                                        |
| `WHISPER_BACKEND` / `WHISPER_MODEL`                  | `local` / `base.en`                                  | `jobpilot interview` speech-to-text. `local` = on-device `faster-whisper`; `openai` reuses your OpenAI-compatible endpoint.                                                              |
| `INTERVIEW_FOLLOWUPS`                                | `2`                                                  | Max LLM-generated follow-up questions per seed question in `interview --mode mock`.                                                                                                      |
| `PAUSE_THRESHOLD_S`                                  | `1.5`                                                | Gap (seconds) between spoken words counted as a "long pause" in delivery metrics.                                                                                                        |
| `MONOTONE_STD_THRESHOLD_SEMITONES`                   | `1.5`                                                | Below this pitch variation (std-dev in semitones) an answer is flagged monotone. Speaker-normalized, so it holds across voices; tune against your own recordings.                        |
| `INTERVIEW_QUESTIONS_PATH`                           | `data/interview_questions.yaml`                      | Seed question bank; falls back to the committed `data/interview_questions.example.yaml` when absent.                                                                                     |

## Make targets

| Target                | Description                                                                                                                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `make setup`          | `uv sync` + install pre-commit hooks                                                                                                                                                         |
| `make test`           | Unit tests                                                                                                                                                                                   |
| `make test-all`       | Unit + integration tests                                                                                                                                                                     |
| `make lint`           | Ruff check                                                                                                                                                                                   |
| `make lint-fix`       | Ruff check with auto-fix                                                                                                                                                                     |
| `make typecheck`      | mypy                                                                                                                                                                                         |
| `make ingest`         | Ingest `data/profile/` into ChromaDB                                                                                                                                                         |
| `make eval JD=<path>` | Run evaluator against a job description                                                                                                                                                      |
| `make eval-batch`     | Run all JDs in `evals/jobs/` against labels, write report. Pass `ARGS="--model X --baseline Y"` for overrides.                                                                               |
| `make run JD=<path>`  | Evaluate + tailor (writes `output/<OWNER>_CV_<company>.docx` when score ≥ threshold)                                                                                                         |
| `make generate-pool`  | **Bootstrap-only.** Seed `data/bullet_pool.yaml` from the bullets in `data/cv_template.docx` under `CURRENT_ROLE_ANCHOR`. Refuses to overwrite an existing pool; pass `FORCE=1` to override. |

Or call the CLI directly: `uv run jobpilot {version,ingest,eval,run,interview} ...`.

## Interview practice ("Coach mode")

Rehearse spoken answers and get feedback grounded in your own ingested CV:

```bash
uv run jobpilot interview                       # mic, drill mode, 3 questions
uv run jobpilot interview --mode mock --num 2   # seed Qs + LLM follow-ups, summary at end
uv run jobpilot interview --text                # type answers instead of using a mic
uv run jobpilot interview --category behavioral --num 5 --no-save
```

- **drill** — one question at a time, feedback printed per answer. **mock** — each seed question
  gets up to `INTERVIEW_FOLLOWUPS` LLM-generated follow-ups, with a session summary at the end.
- **Delivery metrics** (computed locally, passed to the LLM as facts — never invented): words-per-minute,
  filler-word counts, long pauses, and **pitch/monotone** measured in semitones (parselmouth f0).
- **Content feedback is CV-grounded:** the coach cites the profile chunk IDs it relied on, flags
  claims your CV doesn't support, and surfaces real experiences you forgot to mention — it never
  writes a "model answer." Same anti-hallucination stance as the tailor (below).
- Requires an ingested CV (`jobpilot ingest`). Audio mode needs a working microphone; `--text` is the
  mic-free fallback. Reports are written to `output/interview_<ts>.md` (gitignored) unless `--no-save`.
- Seed questions come from `data/interview_questions.yaml` (gitignored override) or the committed
  `data/interview_questions.example.yaml`.

> **Note on `generate-pool`:** this is a _one-time_ helper. After bootstrap, edit
> `data/bullet_pool.yaml` by hand — rename auto IDs to meaningful slugs and add
> alternative framings. The tailor cannot reject what's in the pool, so human
> review before each addition is the safety net.

## Hallucination guardrails

CV-tailoring tools commonly fabricate accomplishments. JobPilot's tailor cannot, by design — four
independent layers prevent it from inventing current-role bullet text:

1. **Schema-level.** `TailoredCV` exposes `current_role_bullet_ids: list[str]` only — there is
   no free-text bullet field. The LLM's structured tool call can't return invented bullets.
2. **Prompt-level.** The tailor prompt explicitly enumerates "MAY NOT invent / MAY NOT return
   IDs not in the pool / MAY NOT repeat IDs."
3. **Post-call validation.** `BulletPool.resolve_current_role(ids)` raises `InvalidBulletIdError`
   on any returned ID not in `data/bullet_pool.yaml`.
4. **Writer invariant.** `docx_writer` only rewrites the `List Paragraph`-styled rows between
   your `CURRENT_ROLE_ANCHOR` and the following `Tech:` line. Promoted notes, hackathon blocks,
   previous experience, and education are passed through verbatim.

Free-form rewriting is still allowed for the **summary** and **skills** (with prompt-level "use
only what's in the retrieved chunks" constraints), since those are lower-risk to paraphrase.

## Troubleshooting

| Error                                                            | Fix                                                                                                                                                                             |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SectionAnchorMissingError: Anchor (contains) not found: 'X'`    | Your `CURRENT_ROLE_ANCHOR` doesn't appear verbatim in `data/cv_template.docx`. Copy the line directly from your CV.                                                             |
| `SectionAnchorMissingError: Anchor not found: 'CAREER OVERVIEW'` | Your template doesn't have the all-caps section headers JobPilot expects. Add `CAREER OVERVIEW`, `SKILLS & EXPERTISE`, `WORK EXPERIENCE` as plain (`Normal`-styled) paragraphs. |
| `InvalidBulletIdError: Unknown bullet id 'X'`                    | The LLM returned a bullet ID not in your pool. Usually means the pool was edited mid-run; re-run `jobpilot run`.                                                                |
| `Bullet pool ... must have a top-level current_role list.`       | Your `data/bullet_pool.yaml` is missing or malformed. Start from `data/bullet_pool.example.yaml`.                                                                               |
| Ollama connection refused                                        | `ollama serve` not running, or `nomic-embed-text` not pulled.                                                                                                                   |

## Status

- **Day 1 (complete)**
  - Pydantic schemas, settings, structured logging
  - LiteLLM gateway (OpenAI-compatible) + structured output via tool call
  - RAG over .docx profile via ChromaDB + Ollama embeddings (Voyage/OpenAI fallback)
  - Evaluator agent (`jobpilot eval`)
- **Day 2 (complete)**
  - Tailor agent + `python-docx` writer
  - Role bullet pool — schema-enforced, post-validated, hallucination-proof
  - LangGraph orchestrator: `evaluate → (score ≥ threshold) → tailor → END`
  - `jobpilot run <jd.txt>` produces `output/<OWNER>_CV_<company>.docx`
- **v3 item 1 (complete)**
  - Eval harness: `jobpilot eval-batch` with JSONL + Markdown reports
  - Baseline diffing, cost tracking, per-JD regression detection
  - 7 labelled JDs, 5-model benchmark (see `evals/README.md`)
  - LinkedIn JD scraper (`scripts/scrape_linkedin_jd.py`)
  - 114 tests, 92% coverage
- **Interview practice (complete)**
  - `jobpilot interview` — drill + mock modes, `--text` fallback
  - Local Whisper transcription (`faster-whisper`) with word timestamps; optional `openai` backend
  - Delivery metrics: WPM, fillers, long pauses, pitch/monotone (semitone-normalized via parselmouth)
  - CV-grounded `interview_coach` agent + mock-mode `interviewer` agent; markdown session reports
- **Backlog (v2/v3)**
  - Tracing + observability, LangGraph HITL checkpointing, hybrid retrieval
  - Discord bot wrapper, Monitor / Outreach / Prep / Tracker agents
  - FastAPI + LangSmith tracing
