.PHONY: setup test lint format train serve

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

serve:
	uv run uvicorn income_classifier.api:app --reload