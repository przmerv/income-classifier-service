# syntax=docker/dockerfile:1

FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app


COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

    
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable
    
RUN find /app/.venv -type d -name tests -prune -exec rm -rf {} +

# Train the model: writes /app/artifacts/model.joblib
RUN /app/.venv/bin/python -m income_classifier.train

FROM python:3.14-slim AS runtime

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/artifacts /app/artifacts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    APP_MODEL_PATH=/app/artifacts/model.joblib

ARG GIT_SHA=local
ENV APP_GIT_SHA=$GIT_SHA

USER appuser
EXPOSE 8000

CMD ["uvicorn", "income_classifier.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]