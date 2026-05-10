# JobPilot

Multi-agent job-hunting assistant. **Day 1 ships the evaluator + RAG over your profile.**

## Quick start

```bash
uv sync
cp .env.example .env  # then fill in ANTHROPIC_API_KEY + VOYAGE_API_KEY
uv run jobpilot ingest --profile data/profile/
uv run jobpilot eval data/sample_jobs/canva.txt
```

## Day 1 status

- [x] Pydantic schemas, settings, structured logging
- [x] Anthropic client with prompt caching + structured output
- [x] RAG over .docx profile via ChromaDB + Voyage embeddings (OpenAI fallback)
- [x] Evaluator agent (`jobpilot eval`)

## Day 2 (deferred)

- CV tailor agent + .docx writer
- LangGraph orchestrator + conditional edge on threshold
- `jobpilot run <jd.txt>` end-to-end
