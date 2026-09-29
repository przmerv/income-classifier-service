.PHONY: setup test lint format train

setup:
	uv sync --frozen
	uv run pre-commit install

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests

format:
	uv run ruff check --fix .
	uv run ruff format .

train:
	uv run python -m income_classifier.train