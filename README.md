# income-classifier-service

[![CI](https://github.com/przmerv/income-classifier-service/actions/workflows/ci.yml/badge.svg)](https://github.com/przmerv/income-classifier-service/actions/workflows/ci.yml)

A production-style ML service that predicts the probability a person earns more than $50K a year, based on the UCI Adult Income dataset. The model is deliberately simple (logistic regression); the focus is everything around it: validation, observability, testing, containerization, and automated delivery.

## Features

- **FastAPI service** with typed request and response schemas
- **Clear 4xx errors**, including a specific 422 for text values never seen in training
- **Single saved pipeline:** preprocessing and model are saved together, preventing training-serving skew
- **Model version stamped at training time**, reported with every prediction
- **Structured JSON logs** with a request ID on every line, returned in the `X-Request-ID` header
- **Streaming batch endpoint** (Server-Sent Events) with per-person partial success
- **Multi-stage Docker image**, non-root, under 500 MB
- **CI/CD:** lint, type check, and tests on every push; image built and pushed to `ghcr.io` on `main`

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness check |
| `GET` | `/version` | Model version and code commit |
| `POST` | `/predict` | Probability for one person |
| `POST` | `/predict/stream` | Streams one result per person in a batch (SSE) |

Interactive docs are available at `/docs` when the service is running.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
make setup     # install dependencies and pre-commit hooks
make train     # train the model → artifacts/model.joblib
make serve     # start the API on http://127.0.0.1:8000
```

Example request:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @test-api-requests/person.json
```

## Run with Docker

Pull the published image:

```bash
docker pull ghcr.io/przmerv/income-classifier-service:latest
docker run --rm -p 8000:8000 ghcr.io/przmerv/income-classifier-service:latest
```

Or build locally (the model is trained during the build):

```bash
make docker-build
make docker-run
```

## Configuration

Settings are read from environment variables with the `APP_` prefix:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_MODEL_PATH` | `artifacts/model.joblib` | Location of the trained model |
| `APP_LOG_LEVEL` | `INFO` | Log verbosity |
| `APP_GIT_SHA` | `local` | Code commit, set automatically in CI builds |

## Development

```bash
make format    # auto-format and fix lint issues
make lint      # ruff + mypy --strict
make test      # pytest
```

Tests use a tiny model trained from `test-api-requests/batch.json`, so they run without the real dataset or a trained model file.

## Project structure

```
src/income_classifier/
├── config.py           # settings from environment variables
├── features.py         # column schema and preprocessing pipeline
├── train.py            # trains and saves the model with its version
├── predictor.py        # loads the model once and serves predictions
├── schemas.py          # request and response models
├── logging_config.py   # JSON logs with request IDs
└── api.py              # FastAPI app, middleware, and endpoints
tests/                  # unit and API tests
test-api-requests/      # sample request bodies
```

## Architecture

_To be added with the Azure deployment._

## Performance

_p50 and p95 latency to be added after load testing._