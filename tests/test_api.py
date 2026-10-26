"""Tests for the API endpoints, using the tiny model from conftest.py."""

from typing import Any

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    """The health check answers 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_reports_model_version(client: TestClient) -> None:
    """The version endpoint reports the version stored in the loaded model file.

    Confirms the label stamped at training time reaches callers unchanged,
    so a running service can be traced to the exact model it serves.
    """
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json()["model_version"] == "test-model"


def test_predict_returns_probability_and_version(
    client: TestClient, person_request: dict[str, Any]
) -> None:
    """A valid person gets a 200 with a probability and the model version.

    The probability must be between 0 and 1, and the version must match the
    loaded model, so every answer can be traced to the model that made it.
    """
    response = client.post("/predict", json=person_request)
    body = response.json()
    assert response.status_code == 200
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "test-model"


def test_predict_rejects_out_of_range_age(
    client: TestClient, person_request: dict[str, Any]
) -> None:
    """An age above the allowed maximum is rejected with a 422.

    Impossible values must be stopped by the schema before reaching the
    model, instead of producing a confident but meaningless prediction.
    """
    age_out_of_range = person_request.copy()
    age_out_of_range["age"] = 500
    response = client.post("/predict", json=age_out_of_range)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client: TestClient, person_request: dict[str, Any]) -> None:
    """A request missing a required field is rejected with a 422.

    Every field is required, so a missing one must be reported to the caller
    rather than silently filled in by the model's preparation steps.
    """
    person_without_age = person_request.copy()
    del person_without_age["age"]
    response = client.post("/predict", json=person_without_age)
    assert response.status_code == 422


def test_predict_rejects_value_not_in_training_data(
    client: TestClient, person_request: dict[str, Any]
) -> None:
    """A text value never seen in training is rejected with a specific 422.

    The error must say the value was not in the training data and name the
    field, so the caller knows exactly what to fix.
    """
    unknown_occupation = person_request.copy()
    unknown_occupation["occupation"] = "Astronaut"
    response = client.post("/predict", json=unknown_occupation)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error"] == "value_not_in_training_data"
    flagged_fields: list[str] = []
    for problem in body["detail"]["fields"]:
        flagged_fields.append(problem["field"])
    assert flagged_fields == ["occupation"]


def test_response_includes_request_id(client: TestClient, person_request: dict[str, Any]) -> None:
    """Every response carries an X-Request-ID header.

    The ID lets a caller report a specific request, and lets the service's
    logs be searched for everything that happened to it.
    """
    response = client.post("/predict", json=person_request)
    assert response.headers.get("x-request-id")


def test_caller_request_id_is_reused(client: TestClient, person_request: dict[str, Any]) -> None:
    """A request ID sent by the caller is reused instead of replaced.

    Reusing it lets one ID follow a request across several services, so all
    their logs can be searched together.
    """
    caller_id = "test-id-123"
    response = client.post("/predict", json=person_request, headers={"x-request-id": caller_id})
    assert response.headers.get("x-request-id") == caller_id


def test_stream_sends_one_prediction_per_person(
    client: TestClient, batch_request: dict[str, Any]
) -> None:
    """The stream sends one prediction per valid person, then a done message.

    Callers rely on the message count and the final summary to know every
    person in the batch was processed.
    """
    people_count = len(batch_request["people"])
    response = client.post(url="/predict/stream", json=batch_request)
    assert response.status_code == 200
    response_text = response.text
    assert response_text.count("event: prediction") == people_count
    assert response_text.count("event: done") == 1


def test_stream_rejects_bad_person_and_scores_the_rest(
    client: TestClient, batch_request: dict[str, Any]
) -> None:
    """One invalid person is rejected while the others are still scored.

    Partial success means a single bad entry never blocks the rest of the
    batch, and the summary must count the failure.
    """
    people = list(batch_request["people"])
    bad_person = people[0].copy()
    bad_person["age"] = 500
    people[0] = bad_person
    bad_batch = {"people": people}
    people_count = len(people)

    response = client.post("/predict/stream", json=bad_batch)
    assert response.status_code == 200

    text = response.text
    assert text.count("event: rejected") == 1
    assert text.count("event: prediction") == people_count - 1
    assert text.count("event: done") == 1
    assert '"failed": 1' in text


def test_stream_rejects_empty_batch(client: TestClient) -> None:
    """A batch with no people is rejected with a 422.

    A batch must hold at least one person; an empty one is a caller mistake
    and is reported before any streaming starts.
    """
    no_people: dict[str, list[Any]] = {"people": []}
    response = client.post("/predict/stream", json=no_people)
    assert response.status_code == 422
