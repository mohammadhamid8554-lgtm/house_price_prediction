from datetime import date

from pydantic import BaseModel, ConfigDict


class HouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: date
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    street: str
    city: str
    statezip: str
    country: str


class PredictionResponse(BaseModel):
    predicted_price: float


class HealthResponse(BaseModel):
    status: str
    model_artifact_available: bool
    preprocessor_artifact_available: bool
