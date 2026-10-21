"""Unit tests for Predictor, using the tiny model from conftest.py.

Each test checks one behavior of Predictor on its own, without the API,
so a failure here points directly at model loading or prediction.
"""

from pathlib import Path
from typing import Any

import pytest

from income_classifier.features import MISSING_VALUE
from income_classifier.predictor import Predictor


def test_predict_returns_one_probability_per_row(
    model_path: Path, sample_rows: list[dict[str, Any]]
) -> None:
    """Predict returns exactly one probability for each input row.

    Callers match results to inputs by position, so a missing or extra
    result would attach predictions to the wrong people.
    """
    predictor = Predictor(model_path)

    predictions = predictor.predict(sample_rows)
    assert len(predictions) == len(sample_rows)


def test_predictions_are_between_0_and_1(
    model_path: Path, sample_rows: list[dict[str, Any]]
) -> None:
    """Every prediction is a valid probability between 0 and 1.

    A value outside this range means the model or its output handling is
    broken, and must never be sent to a caller.
    """
    predictor = Predictor(model_path)

    predictions = predictor.predict(sample_rows)

    for prediction in predictions:
        assert 0.0 <= prediction <= 1.0


def test_model_version_is_read_from_file(model_path: Path) -> None:
    """The model version is read from the saved model file.

    The version is stamped at training time, so the loaded predictor must
    report the label stored with the model, not a hand-typed value.
    """
    predictor = Predictor(model_path)
    assert predictor.model_version == "test-model"


def test_missing_model_file_raises(tmp_path: Path) -> None:
    """Loading from a path with no model file raises FileNotFoundError.

    A missing model must fail at startup with a clear error, instead of
    letting the service start and fail on its first request.
    """
    path = tmp_path / "nonexistent-model.joblib"
    with pytest.raises(FileNotFoundError) as excinfo:
        Predictor(path)

    assert "Model artifact not found" in str(excinfo.value)


def test_known_categories_exclude_missing_filler(model_path: Path) -> None:
    """The missing-value filler word is never listed as a known category.

    The filler is added during preparation to stand in for blanks, so it
    must not be offered to callers as a real value they can send.
    """
    predictor = Predictor(model_path)
    known_categories = predictor.known_categories
    known_categories_set: set[str] = set()

    for categories in known_categories.values():
        known_categories_set.update(categories)

    assert MISSING_VALUE not in known_categories_set


def test_known_row_has_no_unknown_categories(
    model_path: Path, sample_rows: list[dict[str, Any]]
) -> None:
    """A row built from training data passes the unseen-category check.

    Every text value in the row was seen during training, so no value
    should be reported as unknown.
    """
    predictor = Predictor(model_path)

    for row in sample_rows:
        assert predictor.find_unknown_categories(row) == {}


def test_unseen_category_is_flagged(model_path: Path, sample_rows: list[dict[str, Any]]) -> None:
    """An occupation never seen in training is reported as unknown.

    The check must name the exact column and value, so the API can tell
    the caller precisely what to fix.
    """
    predictor = Predictor(model_path)
    astronaut_row = sample_rows[0].copy()
    astronaut_row["occupation"] = "Astronaut"  # never seen in training
    assert predictor.find_unknown_categories(astronaut_row) == {"occupation": "Astronaut"}
