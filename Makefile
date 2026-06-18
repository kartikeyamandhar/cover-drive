# Cricket commentary voice engine - developer entrypoints.
# All Python tooling runs through `uv run` so the locked versions are used.

.DEFAULT_GOAL := help
.PHONY: help setup lint format type test check web-check pipeline serve serve-smoke clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the env from the lockfile and install pre-commit hooks
	uv sync --locked
	uv run pre-commit install

lint:  ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format:  ## Apply ruff autofixes and formatting
	uv run ruff check --fix .
	uv run ruff format .

type:  ## mypy strict
	uv run mypy .

test:  ## pytest with coverage
	uv run pytest

check: lint type test  ## Full Python gate: ruff + mypy + pytest

web-check:  ## Web gate: install + lint + build
	cd web && pnpm install && pnpm lint && pnpm build

pipeline:  ## Placeholder for the data/training pipeline (implemented from Phase 1)
	@echo "pipeline: no stages yet; implemented from Phase 1."

serve:  ## Serve the model locally (transformers + peft); add ARGS="--stub" for no model
	uv run python -m scripts.serve $(ARGS)

serve-smoke:  ## Headless: stream one bundled match to stdout (stub runtime, no GPU)
	uv run python -m scripts.serve --smoke

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build
	find . -type d -name __pycache__ -not -path './web/*' -prune -exec rm -rf {} +
