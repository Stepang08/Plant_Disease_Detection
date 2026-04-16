"""Pydantic models for the FastAPI request/response schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    disease: str = Field(..., description="Predicted disease name")
    confidence: float = Field(..., ge=0, le=1, description="Softmax probability")


class PredictionResponse(BaseModel):
    disease: str = Field(..., description="Top predicted disease")
    confidence: float = Field(..., ge=0, le=1, description="Confidence of top prediction")
    top_k: list[PredictionResult] = Field(..., description="Top-k predictions with confidence")


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool


class ClassesResponse(BaseModel):
    num_classes: int
    classes: list[str]
