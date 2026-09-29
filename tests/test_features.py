from income_classifier.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_pipeline,
)


def test_numeric_and_categorical_do_not_overlap() -> None:
    overlap = set(NUMERIC_COLUMNS) & set(CATEGORICAL_COLUMNS)
    assert overlap == set()


def test_feature_columns_have_no_duplicates() -> None:
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_pipeline_ends_with_model_step() -> None:
    pipeline = build_pipeline()
    last_step_name = pipeline.steps[-1][0]
    assert last_step_name == "model"
