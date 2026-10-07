from pydantic import BaseModel, Field


class Person(BaseModel):
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
    probability: float = Field(ge=0.0, le=1.0)
    model_version: str
