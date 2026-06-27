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
    confidenceThreshold: float = Field(default=0.55, ge=0, le=1)


class DetectFrameResponse(BaseModel):
    frameId: int
    modelId: str
    inferenceMs: float
    detections: list[Detection]


class VisionModelsRequest(BaseModel):
    baseUrl: str = Field(min_length=1)


class VisionModelInfo(BaseModel):
    id: str
    object: str | None = None


class VisionModelsResponse(BaseModel):
    models: list[VisionModelInfo]


class VisionAnalyzeRequest(BaseModel):
    baseUrl: str = Field(min_length=1)
    modelId: str = Field(min_length=1)
    prompt: str | None = None
    imageData: str = Field(min_length=1)
    eventType: str = Field(pattern="^(new_person|person_moved)$")
    frameId: int
    detections: list[Detection] = Field(default_factory=list)


class VisionAnalyzeResponse(BaseModel):
    message: str
    createdAt: str
    modelId: str
