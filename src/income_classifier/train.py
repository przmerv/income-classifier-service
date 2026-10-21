"""Train the income model and save it, with its version, to one file.

Run with `make train`. Only needed when something changes what the model
learns: the columns, preparation steps, model settings, or data.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import joblib
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from income_classifier.config import Settings
from income_classifier.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    POSITIVE_LABEL,
    TARGET_COLUMN,
    build_pipeline,
)

logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.Series[int]]:
    """Download the Adult Income data and split it into inputs and answers.

    Keeps only the recipe's columns and turns text columns into plain
    strings, matching what the API will receive. The answer becomes 1 for
    earning over $50K and 0 otherwise.
    """
    dataset = fetch_openml("adult", version=2, as_frame=True)
    frame: pd.DataFrame = dataset.frame

    features = frame[FEATURE_COLUMNS].copy()
    # OpenML gives "category" dtype; the API will send plain strings.
    # Convert so training sees the same kind of input serving will.
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].astype("object")

    target = (frame[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
    return features, target


def train(settings: Settings) -> float:
    """Train the pipeline, score it, and save it with a version label.

    Holds back part of the data to measure ROC-AUC, then saves the trained
    pipeline, its version (the training time in UTC), and its score together
    in one file so they can never get separated. Returns the ROC-AUC.
    """
    features, target = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=target,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))

    model_version = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    artifact = {
        "pipeline": pipeline,
        "model_version": model_version,
        "roc_auc": auc,
    }

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, settings.model_path)
    return auc


def main() -> None:
    """Run training from the command line: read settings, train, report.

    Sets up logging at the configured level, trains and saves the model,
    then logs the ROC-AUC and where the file was saved.
    """
    settings = Settings()
    logging.basicConfig(level=settings.log_level)
    auc = train(settings)
    logger.info("ROC-AUC %.4f, model saved to %s", auc, settings.model_path)


if __name__ == "__main__":
    main()
