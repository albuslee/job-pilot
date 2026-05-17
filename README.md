# JobPilot

Multi-agent job-hunting assistant. **Day 1 ships the evaluator + RAG over your profile.**

## Quick start

```bash
make setup                          # install deps + git hooks
cp .env.example .env                # fill in LITELLM_API_KEY, LLM_MODEL
make ingest                         # embed profile into ChromaDB
make eval JD=data/sample_jobs/canva_fullstack.txt
```

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

## Day 1 status

- [x] Pydantic schemas, settings, structured logging
- [x] LiteLLM gateway (OpenAI-compatible) + structured output via tool call
- [x] RAG over .docx profile via ChromaDB + Ollama embeddings (Voyage/OpenAI fallback)
- [x] Evaluator agent (`jobpilot eval`)

## Day 2 (deferred)

- CV tailor agent + .docx writer
- LangGraph orchestrator + conditional edge on threshold
- `jobpilot run <jd.txt>` end-to-end
