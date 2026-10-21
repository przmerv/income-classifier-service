"""Tests for the API endpoints, using the tiny model from conftest.py."""

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    """The health check answers 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
