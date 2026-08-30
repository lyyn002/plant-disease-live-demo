"""Pydantic request/response schemas."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    num_classes: int


class PredictionItem(BaseModel):
    """Single class prediction with confidence."""

    label: str = Field(..., description="Human-readable disease label")
    crop: str = Field(..., description="Crop species")
    disease: str = Field(..., description="Disease or healthy status")
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Prediction response for an uploaded leaf image."""

    primary: PredictionItem
    top_k: list[PredictionItem]
    inference_ms: float = Field(..., description="Server-side inference latency in milliseconds")


class ErrorResponse(BaseModel):
    """Structured API error."""

    detail: str
    error_code: str
