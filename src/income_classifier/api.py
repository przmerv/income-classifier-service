from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from income_classifier.config import Settings
from income_classifier.predictor import Predictor
from income_classifier.schemas import Person, PredictResponse


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


@app.post("/predict")
def predict(person: Person, request: Request) -> PredictResponse:
    """Return the probability that one person earns more than $50K a year.

    FastAPI checks the JSON body against Person before this runs, so bad
    or missing fields get a 422 error automatically. Uses the model that
    was loaded once at startup.
    """
    predictor: Predictor = request.app.state.predictor
    settings: Settings = request.app.state.settings

    row = person.model_dump(by_alias=True)
    probability = predictor.predict([row])[0]

    return PredictResponse(probability=probability, model_version=settings.model_version)
