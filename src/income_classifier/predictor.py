from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from income_classifier.features import FEATURE_COLUMNS


class Predictor:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        self.model_path = model_path
        self._pipeline: Any = joblib.load(model_path)

    def predict(self, rows: list[dict[str, Any]]) -> list[float]:
        frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        probabilities = self._pipeline.predict_proba(frame)[:, 1]

        results: list[float] = []
        for value in probabilities:
            results.append(float(value))
        return results
