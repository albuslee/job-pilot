# JobPilot

Multi-agent job-hunting assistant. Evaluates a job description against your CV,
optionally tailors a CV to match. All personal data stays local.

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
make eval JD=data/sample_jobs/canva_fullstack.txt
make run  JD=data/sample_jobs/canva_fullstack.txt   # evaluate + tailor → output/
```

> `data/profile/`, `data/cv_template.docx`, and `data/bullet_pool.yaml` are
> all gitignored — your CV and work history stay local. Tests build their own
> synthetic templates and pools at runtime and never read these files.

## Configuration

All personal info lives in `.env` (gitignored). Key variables:

| Env var | Default | Purpose |
|---|---|---|
| `OWNER_NAME` | `Owner` | Used to build the output filename: `<owner_slug>_CV_<company>.docx`. |
| `CURRENT_ROLE_ANCHOR` | (empty) | Exact text in your CV that identifies your current role. The tailor uses this to find the bullet block to rewrite. Empty = skip role-bullet rewrite; only summary + skills are tailored. |
| `CV_TEMPLATE_PATH` / `BULLET_POOL_PATH` | `data/cv_template.docx` / `data/bullet_pool.yaml` | Override file locations. |
| `LITELLM_BASE_URL` / `LITELLM_API_KEY` / `LLM_MODEL` | `http://localhost:4000/v1` / — / `claude-sonnet-4-6` | Any OpenAI-compatible endpoint. Point at LiteLLM, OpenAI, Anthropic-compat, Azure OpenAI, Ollama `/v1`, etc. |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama` \| `voyage` \| `openai`. |
| `OLLAMA_BASE_URL` / `OLLAMA_EMBEDDING_MODEL` | `http://localhost:11434` / `nomic-embed-text` | Used when `EMBEDDING_PROVIDER=ollama`. |
| `SCORE_THRESHOLD` | `70` | `jobpilot run` only writes a tailored CV when `evaluation.score ≥ this`. |
| `RETRIEVAL_K` | `8` | Number of top-k profile chunks pulled from ChromaDB per query. |

## Make targets

| Target | Description |
|---|---|
| `make setup` | `uv sync` + install pre-commit hooks |
| `make test` | Unit tests |
| `make test-all` | Unit + integration tests |
| `make lint` | Ruff check |
| `make lint-fix` | Ruff check with auto-fix |
| `make typecheck` | mypy |
| `make ingest` | Ingest `data/profile/` into ChromaDB |
| `make eval JD=<path>` | Run evaluator against a job description |
| `make run JD=<path>` | Evaluate + tailor (writes `output/<OWNER>_CV_<company>.docx` when score ≥ threshold) |
| `make generate-pool` | **Bootstrap-only.** Seed `data/bullet_pool.yaml` from the bullets in `data/cv_template.docx` under `CURRENT_ROLE_ANCHOR`. Refuses to overwrite an existing pool; pass `FORCE=1` to override. |

Or call the CLI directly: `uv run jobpilot {version,ingest,eval,run} ...`.

> **Note on `generate-pool`:** this is a *one-time* helper. After bootstrap, edit
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

| Error | Fix |
|---|---|
| `SectionAnchorMissingError: Anchor (contains) not found: 'X'` | Your `CURRENT_ROLE_ANCHOR` doesn't appear verbatim in `data/cv_template.docx`. Copy the line directly from your CV. |
| `SectionAnchorMissingError: Anchor not found: 'CAREER OVERVIEW'` | Your template doesn't have the all-caps section headers JobPilot expects. Add `CAREER OVERVIEW`, `SKILLS & EXPERTISE`, `WORK EXPERIENCE` as plain (`Normal`-styled) paragraphs. |
| `InvalidBulletIdError: Unknown bullet id 'X'` | The LLM returned a bullet ID not in your pool. Usually means the pool was edited mid-run; re-run `jobpilot run`. |
| `Bullet pool ... must have a top-level current_role list.` | Your `data/bullet_pool.yaml` is missing or malformed. Start from `data/bullet_pool.example.yaml`. |
| Ollama connection refused | `ollama serve` not running, or `nomic-embed-text` not pulled. |

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
- **Backlog (v2)**
  - Discord bot wrapper, Monitor / Outreach / Prep / Tracker agents
  - FastAPI + LangSmith tracing + full eval harness
