# =============================================================================
# api/schemas.py — Pydantic v2 models for API request/response validation
# =============================================================================

from typing import Optional
from pydantic import BaseModel, Field


class Top5Item(BaseModel):
    """A single entry in the top-5 predictions list."""
    class_raw: str = Field(..., description="Raw class folder name")
    disease_name: str = Field(..., description="Human-readable disease name")
    confidence_pct: float = Field(..., ge=0, le=100,
                                   description="Confidence percentage")


class PredictionResponse(BaseModel):
    """Full prediction result for a single image."""
    predicted_class_raw: str = Field(...,
        description="Raw class folder name (e.g. Tomato___Early_blight)")
    disease_name: str = Field(...,
        description="Human-readable disease name")
    confidence_pct: float = Field(..., ge=0, le=100,
        description="Top-1 confidence percentage")
    is_healthy: bool = Field(...,
        description="Whether the plant is predicted as healthy")
    plant: str = Field(...,
        description="Plant species name")
    description: str = Field("",
        description="Disease description from CSV knowledge base")
    prevention: list[str] = Field(default_factory=list,
        description="List of prevention tips")
    cure: list[str] = Field(default_factory=list,
        description="List of treatment methods")
    top5: list[Top5Item] = Field(default_factory=list,
        description="Top-5 predictions with confidence")
    inference_time_ms: float = Field(0.0,
        description="Model inference time in milliseconds")
    warning: Optional[str] = Field(None,
        description="Warning message if confidence is low or image is unusual")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    results: list[PredictionResponse] = Field(...,
        description="List of prediction results")
    total_inference_time_ms: float = Field(0.0,
        description="Total inference time for all images")


class DiseaseRecord(BaseModel):
    """A single disease entry from the knowledge base."""
    disease_name: str
    description: str = ""
    prevention: list[str] = Field(default_factory=list)
    cure: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "ok"
    model: str = ""
    classes: int = 0
    uptime_seconds: float = 0.0
