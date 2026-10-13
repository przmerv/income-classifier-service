from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from income_classifier.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, MISSING_VALUE


class Predictor:
    """Loads the trained pipeline once and makes predictions with it.

    Wraps the saved model file behind a simple interface: give it people as
    dicts, get back probabilities. Also knows which text values the model
    learned, so unfamiliar input can be caught before predicting.
    """

    def __init__(self, model_path: Path) -> None:
        """Load the trained pipeline from disk and read its learned categories.

        Fails immediately with FileNotFoundError if the model file is
        missing, so a broken deployment is caught at startup.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        self.model_path = model_path
        self._pipeline: Any = joblib.load(model_path)
        self.known_categories: dict[str, set[str]] = self._load_known_categories()

    def predict(self, rows: list[dict[str, Any]]) -> list[float]:
        """Return the probability of earning over $50K for each person.

        Each row is one person as a dict keyed by dataset column names.
        Rows are lined up to the training column order before the pipeline
        prepares them and the model predicts.
        """
        frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        probabilities = self._pipeline.predict_proba(frame)[:, 1]

        results: list[float] = []
        for value in probabilities:
            results.append(float(value))
        return results

    def find_unknown_categories(self, row: dict[str, Any]) -> dict[str, str]:
        """Return text values the model never saw during training.

        Checks one person's row against the learned categories. The result
        maps column name to the unknown value, and is empty if every value
        is known.
        """
        unknown: dict[str, str] = {}
        for column in CATEGORICAL_COLUMNS:
            value = row[column]
            if value not in self.known_categories[column]:
                unknown[column] = value
        return unknown

    def _load_known_categories(self) -> dict[str, set[str]]:
        """Read every category the model learned, per text column.

        Pulls the list from the trained encoder inside the pipeline, so it
        always matches the model exactly. Leaves out the "missing" filler,
        since callers shouldn't send that as a real value.
        """
        preprocess = self._pipeline.named_steps["preprocess"]
        encoder = preprocess.named_transformers_["cat"].named_steps["encode"]

        known: dict[str, set[str]] = {}
        for column, categories in zip(CATEGORICAL_COLUMNS, encoder.categories_, strict=True):
            values: set[str] = set()
            for value in categories:
                if value != MISSING_VALUE:
                    values.add(str(value))
            known[column] = values
        return known
