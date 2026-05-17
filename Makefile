.PHONY: setup test lint typecheck ingest eval

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
	@if [ -z "$(JD)" ]; then echo "Usage: make eval JD=data/sample_jobs/canva_fullstack.txt"; exit 1; fi
	uv run jobpilot eval $(JD)
