.PHONY: setup test test-all lint lint-fix typecheck ingest eval run generate-pool

setup:
	uv sync
	uv run pre-commit install

test:
	uv run pytest tests/unit/

test-all:
	uv run pytest

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

typecheck:
	uv run mypy

ingest:
	uv run jobpilot ingest --profile data/profile/

eval:
	@if [ -z "$(JD)" ]; then echo "Usage: make eval JD=evals/jobs/canva_fullstack.txt"; exit 1; fi
	uv run jobpilot eval $(JD)

run:
	@if [ -z "$(JD)" ]; then echo "Usage: make run JD=evals/jobs/canva_fullstack.txt"; exit 1; fi
	uv run jobpilot run $(JD)

# Bootstrap data/bullet_pool.yaml from your CV template (one-time helper).
# Refuses to clobber an existing pool file unless FORCE=1.
generate-pool:
	uv run python scripts/generate_bullet_pool.py $(if $(FORCE),--force,)
