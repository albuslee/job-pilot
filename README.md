# JobPilot

Multi-agent job-hunting assistant. Evaluates a job description against your CV,
optionally tailors a CV to match. All personal data stays local.

## Quick start

```bash
make setup                          # install deps + git hooks
cp .env.example .env                # fill in LITELLM_*, OWNER_NAME, CURRENT_ROLE_ANCHOR

# Drop your CV(s) into data/profile/ (gitignored), then copy whichever one
# you want as the tailor's structural template:
cp data/profile/<your_cv>.docx data/cv_template.docx

# Seed your bullet pool (gitignored). Start from the .example template:
cp data/bullet_pool.example.yaml data/bullet_pool.yaml
$EDITOR data/bullet_pool.yaml      # replace placeholders with your real bullets

make ingest                         # embed profile into ChromaDB
make eval JD=data/sample_jobs/canva_fullstack.txt
make run  JD=data/sample_jobs/canva_fullstack.txt   # evaluate + tailor → output/
```

> `data/profile/`, `data/cv_template.docx`, and `data/bullet_pool.yaml` are
> all gitignored — your CV and work history stay local. Tests build their own
> synthetic templates and pools at runtime and never read these files.

## Configuration

All personal info lives in `.env` (gitignored). The most important variables:

| Env var | Purpose |
|---|---|
| `OWNER_NAME` | Used to build the output filename: `<owner_slug>_CV_<company>.docx`. |
| `CURRENT_ROLE_ANCHOR` | Exact text in your CV that identifies your current role. The tailor uses this to find the bullet block to rewrite. Empty = skip role-bullet rewrite (only summary + skills are tailored). |
| `CV_TEMPLATE_PATH` / `BULLET_POOL_PATH` | Override file locations if you keep them outside `data/`. |
| `LITELLM_BASE_URL` / `LITELLM_API_KEY` / `LLM_MODEL` | Your local LiteLLM gateway + model. |

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
