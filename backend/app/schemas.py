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


class VisionTriggerSettings(BaseModel):
    cooldownSeconds: float = Field(default=8, ge=1, le=60)
    stableConfirmFrames: int = Field(default=2, ge=1, le=10)
    missToleranceFrames: int = Field(default=2, ge=0, le=10)
    trackIouThreshold: float = Field(default=0.35, ge=0.1, le=0.9)
    movementDistancePercent: float = Field(default=8, ge=1, le=50)
    movementIouThreshold: float = Field(default=0.55, ge=0.1, le=0.95)


class AppSettings(BaseModel):
    selectedModelId: str | None = None
    selectedCameraId: str | None = None
    language: str = Field(default="en", pattern="^(en|zh)$")
    appearance: str = Field(default="system", pattern="^(system|dark|light)$")
    lmStudioUrl: str = Field(default="http://192.168.4.181:1234/v1", min_length=1)
    lmStudioModelId: str = Field(default="qwen/qwen3-v1-4b")
    lmStudioPrompt: str = Field(default="请用中文简洁描述截图中可见的人、动作、位置变化和明显风险。")
    visionTriggerSettings: VisionTriggerSettings = Field(default_factory=VisionTriggerSettings)
    historyRetentionDays: int = Field(default=1, ge=1, le=365)
    confidenceThreshold: float = Field(default=0.55, ge=0, le=1)


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
    retentionDays: int = Field(default=1, ge=1, le=365)
    sessionId: str | None = None
    trackId: int | None = None
    detections: list[Detection] = Field(default_factory=list)


class VisionAnalyzeResponse(BaseModel):
    message: str
    createdAt: str
    modelId: str
    eventId: str | None = None
    duplicateCount: int = 0
    deduplicated: bool = False


class VisionEventRecord(BaseModel):
    id: str
    sessionId: str | None = None
    trackId: int | None = None
    eventType: str
    modelId: str
    frameId: int
    message: str
    summary: str
    detections: list[dict]
    primaryBox: dict | None = None
    duplicateCount: int
    firstSeenAt: str
    lastSeenAt: str
    createdAt: str
    expiresAt: str
    hasScreenshot: bool
    screenshotMimeType: str | None = None
    screenshotSizeBytes: int
    screenshotWidth: int
    screenshotHeight: int


class VisionEventsResponse(BaseModel):
    events: list[VisionEventRecord]
