from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from income_classifier.config import Settings
from income_classifier.predictor import Predictor


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
