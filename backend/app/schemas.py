from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str
    name: str
    runtime: str
    size: str
    inputSize: int
    available: bool
    recommended: bool = False
    unavailableReason: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    selectedModelId: str


class SelectModelRequest(BaseModel):
    modelId: str


class HealthResponse(BaseModel):
    status: str
    selectedModelId: str
    selectedModelAvailable: bool
    runtime: str


class Box(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(ge=0)
    height: float = Field(ge=0)


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    box: Box


class DetectFrameRequest(BaseModel):
    frameId: int
    imageData: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    modelId: str | None = None
    confidenceThreshold: float = Field(default=0.45, ge=0, le=1)


class DetectFrameResponse(BaseModel):
    frameId: int
    modelId: str
    inferenceMs: float
    detections: list[Detection]
