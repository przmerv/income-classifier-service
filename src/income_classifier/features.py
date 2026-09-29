from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "class"
POSITIVE_LABEL = ">50K"

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
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
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