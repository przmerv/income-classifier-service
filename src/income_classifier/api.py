from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from income_classifier.config import Settings
from income_classifier.predictor import Predictor
from income_classifier.schemas import Person, PredictResponse, VersionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load settings and the trained model once when the API starts.

    Stores both on app.state so every request reuses them instead of
    reloading the model file. If the model file is missing, the API
    fails at startup rather than on the first request.
    """
    settings = Settings()
    app.state.settings = settings
    app.state.predictor = Predictor(settings.model_path)
    yield


app = FastAPI(
    title="income-classifier-service",
    description="Predicts the probability that a person earns more than $50K a year.",
    lifespan=lifespan,
)


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
        raise HTTPException(
            status_code=422,
            detail=build_unknown_category_detail(unknown, predictor),
        )
    unknown = predictor.find_unknown_categories(row)
    probability = predictor.predict([row])[0]

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
