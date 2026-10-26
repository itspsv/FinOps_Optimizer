.PHONY: install db-init dev report test lint format \
	bench-data bench-calibrate bench-run bench-eval bench-report bench-compare

install:
	uv sync

db-init:
	uv run python scripts/init_db.py

dev:
	uv run uvicorn finops.api.main:app --reload --port 8000

report:
	uv run python scripts/report.py $(ARGS)

RUN ?= $(shell date +%Y%m%d-%H%M)

bench-data:        ## download dataset samples and build the workload
	uv run python -m benchmark.datasets
	uv run python -m benchmark.workload $(ARGS)

bench-calibrate:   ## pick the cache similarity threshold (offline, free)
	uv run python -m benchmark.calibrate

bench-run:         ## replay the workload through `make dev` (RUN=<id> to resume)
	uv run python -m benchmark.replay --run-id $(RUN) $(ARGS)

bench-eval:        ## judge answer quality for RUN=<id>
	uv run python -m benchmark.evaluate --run-id $(RUN) $(ARGS)

bench-report:      ## write benchmark/results/<RUN>/report.md
	uv run python -m benchmark.report --run-id $(RUN)

bench-compare:     ## side-by-side runs, e.g. RUNS="full-1 full-2"
	uv run python -m benchmark.compare $(RUNS)

test:
	uv run pytest -q

lint:
	uv run ruff check src tests benchmark scripts

format:
	uv run ruff format src tests benchmark scripts
	uv run ruff check --fix src tests benchmark scripts
