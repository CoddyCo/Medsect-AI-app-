"""
Pydantic schemas for request / response validation.
"""
from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    text: str
    model_id: str = "model_0"


class SentencePrediction(BaseModel):
    sentence: str
    label: str
    confidence: float
    line_number: int


class PredictResponse(BaseModel):
    model_id: str
    model_name: str
    predictions: list[SentencePrediction]
    is_mock: bool


class CompareRequest(BaseModel):
    text: str


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    architecture: str
    accuracy: Optional[float]
    is_mock: bool
