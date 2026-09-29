"""
API response/request schemas.

These define the JSON contract the frontend (Person C) will build against,
and mirror the data the engine (Person A) needs to hand back.
"""

from pydantic import BaseModel, Field
from typing import Literal

ModelName = Literal["resnet50", "mobilenetv3", "vit"]


class Prediction(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ActivationPoint(BaseModel):
    """One point in the normalized activation/attention map, 0-1 coordinate space."""
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    intensity: float = Field(..., ge=0.0, le=1.0)


class CloakResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    cloaked_image_base64: str  # PNG, base64-encoded, no data-URI prefix
    model_used: ModelName
    method_used: str  # "fgsm" or "pgd"
    protection_mode: str
    strength: float
    epsilon: float
    misclassified: bool
    confidence_drop: float
    original_class_confidence_after: float

    original_predictions: list[Prediction]
    cloaked_predictions: list[Prediction]

    activation_map: list[ActivationPoint]

    ssim_score: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    engine_mode: str  # "mock" or "live" — tells you at a glance what's wired in


class ModelInfo(BaseModel):
    name: ModelName
    display_name: str
    supports_face_mode: bool
    implemented: bool  # False = listed for the UI but not callable yet, ask Person A


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
