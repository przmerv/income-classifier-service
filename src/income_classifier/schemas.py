from typing import Any

from pydantic import BaseModel, Field


class Person(BaseModel):
    """One person's details, as sent to the /predict endpoint.

    Every field is required. Numbers are checked against realistic ranges,
    so a bad value is rejected with a 422 error before it reaches the model.
    Aliases map the friendly names back to the dataset's column names.
    """

    age: int = Field(ge=17, le=100)
    education_num: int = Field(ge=1, le=16, serialization_alias="education-num")
    capital_gain: int = Field(ge=0, serialization_alias="capital-gain")
    capital_loss: int = Field(ge=0, serialization_alias="capital-loss")
    hours_per_week: int = Field(ge=1, le=99, serialization_alias="hours-per-week")
    workclass: str
    education: str
    marital_status: str = Field(serialization_alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(serialization_alias="native-country")


class PredictResponse(BaseModel):
    """What the /predict endpoint sends back.

    The probability (between 0 and 1) that the person earns more than $50K
    a year, plus the model version that produced it, so every answer can be
    traced back to the exact model used.
    """

    probability: float = Field(ge=0.0, le=1.0)
    model_version: str


class VersionResponse(BaseModel):
    """What the /version endpoint returns.

    Tells a caller which model is loaded and which commit of the code is
    running, so problems in production can be traced to an exact build.
    """

    model_version: str
    git_sha: str


class BatchRequest(BaseModel):
    """A list of people to score in one request, for /predict/stream.

    Holds between 1 and 100 people as raw JSON objects. Each person is
    checked individually while streaming, so one bad entry only fails
    that person instead of rejecting the whole batch.
    """

    people: list[dict[str, Any]] = Field(min_length=1, max_length=100)
