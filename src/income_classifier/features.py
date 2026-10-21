"""The model's columns and data preparation steps, defined in one place.

Both training and serving import from here, so they always use the same
columns and the same preparation, which prevents training-serving skew.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "class"  # the answer column in the dataset
POSITIVE_LABEL = ">50K"  # the answer that counts as 1
MISSING_VALUE = "missing"  # used to fill in missing values in categorical columns

# fnlwgt is left out on purpose: it's a census sampling weight, not a trait of the person.
NUMERIC_COLUMNS: list[str] = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_COLUMNS: list[str] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
FEATURE_COLUMNS: list[str] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def build_pipeline() -> Pipeline:
    """Build a fresh, untrained pipeline: data preparation plus the model.

    Number columns get blanks filled with the median, then scaled. Text
    columns get blanks filled with "missing", then turned into yes/no
    columns. Training fits it; saving it keeps every step with the model.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_VALUE)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_COLUMNS),
            ("cat", categorical, CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
