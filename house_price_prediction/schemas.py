"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class HousingInput(BaseModel):
    """
    Request model for housing prediction input
    """
    longitude: float = Field(..., description="Longitude coordinate")
    latitude: float = Field(..., description="Latitude coordinate")
    housing_median_age: float = Field(..., description="Median age of houses")
    total_rooms: float = Field(..., description="Total number of rooms")
    total_bedrooms: float = Field(..., description="Total number of bedrooms")
    population: float = Field(..., description="Population in the area")
    households: float = Field(..., description="Number of households")
    median_income: float = Field(..., description="Median income")
    ocean_proximity: Optional[str] = Field(
        "<1H OCEAN", 
        description="Distance to ocean: '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'"
    )

    class Config:
        schema_extra = {
            "example": {
                "longitude": -121.89,
                "latitude": 37.29,
                "housing_median_age": 30.0,
                "total_rooms": 5099.0,
                "total_bedrooms": 1111.0,
                "population": 1490.0,
                "households": 1133.0,
                "median_income": 7.2574,
                "ocean_proximity": "<1H OCEAN"
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for a single prediction
    """
    id: int = Field(..., description="Database record ID")
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: Optional[str]
    model_name: str = Field(..., description="Name of the model used for prediction")
    prediction: float = Field(..., description="Predicted house price")
    timestamp: datetime = Field(..., description="Timestamp of the prediction")

    class Config:
        from_attributes = True


class AllModelsResponse(BaseModel):
    """
    Response model for predictions from all models
    """
    id: int
    input_data: HousingInput
    model_name: str = "all"
    predictions: dict = Field(..., description="Dictionary with predictions from different models")
    timestamp: datetime

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "input_data": {
                    "longitude": -121.89,
                    "latitude": 37.29,
                    "housing_median_age": 30.0,
                    "total_rooms": 5099.0,
                    "total_bedrooms": 1111.0,
                    "population": 1490.0,
                    "households": 1133.0,
                    "median_income": 7.2574,
                    "ocean_proximity": "<1H OCEAN"
                },
                "model_name": "all",
                "predictions": {
                    "linear_regression": 453751.71,
                    "svm_regression": 439957.22
                },
                "timestamp": "2024-04-22T10:30:45.123456"
            }
        }


class InferenceHistoryResponse(BaseModel):
    """
    Response model for historical inference records
    """
    total_records: int = Field(..., description="Total number of records")
    model_name: str = Field(..., description="Model name filter applied")
    records: List[PredictionResponse] = Field(..., description="List of prediction records")

    class Config:
        schema_extra = {
            "example": {
                "total_records": 5,
                "model_name": "linear",
                "records": [
                    {
                        "id": 1,
                        "longitude": -121.89,
                        "latitude": 37.29,
                        "housing_median_age": 30.0,
                        "total_rooms": 5099.0,
                        "total_bedrooms": 1111.0,
                        "population": 1490.0,
                        "households": 1133.0,
                        "median_income": 7.2574,
                        "ocean_proximity": "<1H OCEAN",
                        "model_name": "linear",
                        "prediction": 453751.71,
                        "timestamp": "2024-04-22T10:30:45.123456"
                    }
                ]
            }
        }
