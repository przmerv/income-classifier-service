import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from income_classifier.config import Settings
from income_classifier.logging_config import request_id_var, setup_logging
from income_classifier.predictor import Predictor
from income_classifier.schemas import BatchRequest, Person, PredictResponse, VersionResponse

logger = logging.getLogger(__name__)


def generate_prediction_events(
    raw_people: list[dict[str, Any]], predictor: Predictor, request_id: str
) -> Iterator[str]:
    """Check and score each person, streaming one message per person.

    Valid people get a prediction message; invalid ones get a rejected
    message with the reason, and the rest carry on. Ends with a done
    message counting successes and failures.
    """
    succeeded = 0
    failed = 0

    for index, raw_person in enumerate(raw_people):
        try:
            person = Person.model_validate(raw_person)
        except ValidationError as error:
            failed += 1
            payload: dict[str, Any] = {
                "index": index,
                "reason": "invalid_input",
                "fields": describe_validation_error(error),
            }
            yield f"event: rejected\ndata: {json.dumps(payload)}\n\n"
            continue

        row = person.model_dump(by_alias=True)
        unknown = predictor.find_unknown_categories(row)
        if unknown:
            failed += 1
            logger.warning(
                "values not in training data",
                extra={"fields": {"index": index, "unknown": unknown, "request_id": request_id}},
            )
            detail = build_unknown_category_detail(unknown, predictor)
            payload = {
                "index": index,
                "reason": "value_not_in_training_data",
                "fields": detail["fields"],
            }
            yield f"event: rejected\ndata: {json.dumps(payload)}\n\n"
            continue

        probability = predictor.predict([row])[0]
        succeeded += 1
        payload = {
            "index": index,
            "probability": probability,
            "model_version": predictor.model_version,
        }
        yield f"event: prediction\ndata: {json.dumps(payload)}\n\n"

    logger.info(
        "stream completed",
        extra={"fields": {"succeeded": succeeded, "failed": failed, "request_id": request_id}},
    )
    summary = {"count": len(raw_people), "succeeded": succeeded, "failed": failed}
    yield f"event: done\ndata: {json.dumps(summary)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load settings and the trained model once when the API starts.

    Stores both on app.state so every request reuses them instead of
    reloading the model file. If the model file is missing, the API
    fails at startup rather than on the first request.
    """
    settings = Settings()
    setup_logging(settings.log_level)
    app.state.settings = settings
    app.state.predictor = Predictor(settings.model_path)
    yield


app = FastAPI(
    title="income-classifier-service",
    description="Predicts the probability that a person earns more than $50K a year.",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id_and_log(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Tag each request with an ID and log one summary line when it ends.

    Reuses the caller's X-Request-ID header if sent, otherwise makes a new
    ID. The ID is attached to every log line during the request and sent
    back in the response header, so one request can be traced end to end.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request failed",
            extra={"fields": {"method": request.method, "path": request.url.path}},
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request completed",
            extra={
                "fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                }
            },
        )
        return response
    finally:
        request_id_var.reset(token)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report that the API is up and answering.

    Azure calls this regularly to check the container is alive and
    restarts it if this stops responding.
    """
    return {"status": "ok"}


def build_unknown_category_detail(unknown: dict[str, str], predictor: Predictor) -> dict[str, Any]:
    """Build the error body for values the model never saw in training.

    Lists each problem field with the value sent and the values the model
    does know, so the caller can see exactly what to fix.
    """
    fields: list[dict[str, Any]] = []
    for column, value in unknown.items():
        fields.append(
            {
                "field": column.replace("-", "_"),
                "value": value,
                "allowed": sorted(predictor.known_categories[column]),
            }
        )
    return {
        "error": "value_not_in_training_data",
        "message": "These values were not in the data the model was trained on, "
        "so it cannot make a reliable prediction.",
        "fields": fields,
    }


@app.post("/predict")
def predict(person: Person, request: Request) -> PredictResponse:
    """Return the probability that one person earns more than $50K a year.

    FastAPI checks the JSON body against Person before this runs, so bad
    or missing fields get a 422 error automatically. Uses the model that
    was loaded once at startup.
    """
    predictor: Predictor = request.app.state.predictor

    row = person.model_dump(by_alias=True)
    unknown = predictor.find_unknown_categories(row)
    if unknown:
        logger.warning(
            "values not in training data",
            extra={"fields": {"unknown": unknown}},
        )
        raise HTTPException(
            status_code=422,
            detail=build_unknown_category_detail(unknown, predictor),
        )
    unknown = predictor.find_unknown_categories(row)
    probability = predictor.predict([row])[0]
    logger.info(
        "prediction made",
        extra={"fields": {"probability": probability, "model_version": predictor.model_version}},
    )

    return PredictResponse(probability=probability, model_version=predictor.model_version)


@app.get("/version")
def version(request: Request) -> VersionResponse:
    """Report which model version and code commit this API is running.

    The model version is read from the loaded model file, set at training
    time. The commit comes from APP_GIT_SHA, which CI fills in at build
    time; on a laptop it defaults to "local".
    """
    predictor: Predictor = request.app.state.predictor
    settings: Settings = request.app.state.settings
    return VersionResponse(model_version=predictor.model_version, git_sha=settings.git_sha)


@app.post("/predict/stream")
def predict_stream(batch: BatchRequest, request: Request) -> StreamingResponse:
    """Score a batch of people and stream each result as it is ready.

    Each person is checked on their own: valid people get a prediction,
    invalid ones get a rejected message with the reason, and one bad entry
    never blocks the others. Ends with a summary of successes and failures.
    """
    predictor: Predictor = request.app.state.predictor
    return StreamingResponse(
        generate_prediction_events(batch.people, predictor, request_id_var.get()),
        media_type="text/event-stream",
    )


def describe_validation_error(error: ValidationError) -> list[dict[str, str]]:
    """Turn a pydantic validation error into a short list of problems.

    Each problem names the field and what was wrong with it, e.g. age
    being above the allowed maximum, so callers get a readable reason
    instead of pydantic's full technical output.
    """
    problems: list[dict[str, str]] = []
    for item in error.errors(include_url=False):
        parts: list[str] = []
        for part in item["loc"]:
            parts.append(str(part))
        problems.append({"field": ".".join(parts), "problem": item["msg"]})
    return problems
