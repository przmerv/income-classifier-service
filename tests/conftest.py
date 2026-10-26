"""Shared pytest fixtures: a tiny trained model and an API test client.

Lets the test suite run without the real dataset or a trained model file.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from income_classifier.api import app
from income_classifier.features import build_pipeline
from income_classifier.schemas import Person

REQUESTS_DIR = Path(__file__).parent.parent / "test-api-requests"


def load_request(name: str) -> dict[str, Any]:
    """Load a sample request from the test-api-requests/ folder as a dict.

    The same files are used for manual curl testing, which keeps a single
    set of example people in the repository.
    """

    data: dict[str, Any] = json.loads((REQUESTS_DIR / name).read_text())
    return data


def make_training_rows() -> list[dict[str, Any]]:
    """Build pipeline training rows from the people in batch.json.

    Each person is validated with the Person schema and converted to dataset
    column names (e.g. education-num), mirroring /predict. As a result, the
    tiny model knows every category the test requests use."""

    rows: list[dict[str, Any]] = []
    for raw_person in load_request("batch.json")["people"]:
        person = Person.model_validate(raw_person)
        rows.append(person.model_dump(by_alias=True))
    return rows


@pytest.fixture
def person_request() -> dict[str, Any]:
    """Return the sample person from person.json as a request body."""
    return load_request("person.json")


@pytest.fixture
def batch_request() -> dict[str, Any]:
    """Return the sample batch from batch.json as a request body."""
    return load_request("batch.json")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Train a tiny model on the sample people and save it to a temp folder.

    Uses the production pipeline recipe and artifact format (pipeline plus
    version label), so the API loads it exactly like a real model. Meant only
    for testing the service's plumbing, not for real predictions.
    """
    pipeline = build_pipeline()
    pipeline.fit(pd.DataFrame(make_training_rows()), [1, 0, 1])

    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipeline, "model_version": "test-model", "roc_auc": 0.0}, path)
    return path


@pytest.fixture
def client(model_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Start the API against the tiny model and return a test client.

    Points APP_MODEL_PATH at the tiny model, then enters the TestClient
    context, which runs the app's startup (lifespan) like a real server.
    Requests are handled in-process, without a running server.
    """
    monkeypatch.setenv("APP_MODEL_PATH", str(model_path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_rows() -> list[dict[str, Any]]:
    """Return the batch.json people as dataset-named rows, ready for Predictor."""
    return make_training_rows()
