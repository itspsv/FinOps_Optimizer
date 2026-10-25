.PHONY: install db-init dev test lint format

install:
	uv sync

db-init:
	uv run python scripts/init_db.py

dev:
	uv run uvicorn finops.api.main:app --reload --port 8000

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests
