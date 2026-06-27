from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx

from app.model_registry import ModelRegistry
from app.schemas import (
    AppSettings,
    DetectFrameRequest,
    DetectFrameResponse,
    HealthResponse,
    ModelsResponse,
    SelectModelRequest,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
    VisionEventRecord,
    VisionEventsResponse,
    VisionModelInfo,
    VisionModelsRequest,
    VisionModelsResponse,
)
from app.settings_store import AppSettingsStore
from app.vision_store import VisionEventStore

app = FastAPI(title="Seeya API")
registry = ModelRegistry()
event_store = VisionEventStore()
settings_store = AppSettingsStore(event_store.data_dir)
LM_STUDIO_TIMEOUT_SECONDS = 45.0
DEFAULT_VISION_PROMPT = "请用中文简洁描述截图中可见的人、动作、位置变化和明显风险。"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    selected = registry.selected_model()
    return HealthResponse(
        status="ok",
        selectedModelId=selected.id,
        selectedModelAvailable=selected.available,
        runtime=selected.runtime,
    )


@app.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    return ModelsResponse(models=registry.list_models(), selectedModelId=registry.selected_model_id)


@app.post("/models/select", response_model=ModelsResponse)
async def select_model(payload: SelectModelRequest) -> ModelsResponse:
    try:
        registry.select_model(payload.modelId)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {payload.modelId}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ModelsResponse(models=registry.list_models(), selectedModelId=registry.selected_model_id)


@app.get("/settings", response_model=AppSettings)
async def get_settings() -> AppSettings:
    return settings_store.get_settings()


@app.put("/settings", response_model=AppSettings)
async def put_settings(payload: AppSettings) -> AppSettings:
    return settings_store.save_settings(payload)


@app.post("/vision/models", response_model=VisionModelsResponse)
async def vision_models(payload: VisionModelsRequest) -> VisionModelsResponse:
    base_url = normalize_lm_studio_base_url(payload.baseUrl)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/models")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Unable to reach LM Studio: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"LM Studio returned {response.status_code}: {response.text}")

    data = response.json()
    raw_models = data.get("data")
    if not isinstance(raw_models, list):
        raise HTTPException(status_code=502, detail="LM Studio /models response did not include a data list")

    models = [
        VisionModelInfo(id=item["id"], object=item.get("object"))
        for item in raw_models
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not models:
        raise HTTPException(status_code=502, detail="LM Studio did not return any usable models")
    return VisionModelsResponse(models=models)


@app.post("/vision/analyze", response_model=VisionAnalyzeResponse)
async def vision_analyze(payload: VisionAnalyzeRequest) -> VisionAnalyzeResponse:
    base_url = normalize_lm_studio_base_url(payload.baseUrl)
    created_at = datetime.now(UTC)
    try:
        duplicate = event_store.merge_duplicate_before_analysis(
            session_id=payload.sessionId,
            event_type=payload.eventType,
            detections=payload.detections,
            image_data=payload.imageData,
            retention_days=payload.retentionDays,
            now=created_at,
        )
    except Exception:
        duplicate = None
    if duplicate is not None:
        return VisionAnalyzeResponse(
            message=duplicate.message,
            createdAt=created_at.isoformat(),
            modelId=payload.modelId,
            eventId=duplicate.event_id,
            duplicateCount=duplicate.duplicate_count,
            deduplicated=True,
        )

    request_body = {
        "model": payload.modelId,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "你是本地视觉监控助手。只根据画面内容回答，语言简洁，避免臆测身份。",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_vision_prompt(payload),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": payload.imageData},
                    },
                ],
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=LM_STUDIO_TIMEOUT_SECONDS) as client:
            response = await client.post(f"{base_url}/chat/completions", json=request_body)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Unable to reach LM Studio: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"LM Studio returned {response.status_code}: {response.text}")

    data = response.json()
    message = extract_chat_message(data)
    if not message:
        raise HTTPException(status_code=502, detail="LM Studio response did not include message content")

    event_id: str | None = None
    duplicate_count = 0
    deduplicated = False
    try:
        persistence = event_store.save_or_merge_event(
            session_id=payload.sessionId,
            track_id=payload.trackId,
            event_type=payload.eventType,
            message=message,
            detections=payload.detections,
            image_data=payload.imageData,
            model_id=payload.modelId,
            frame_id=payload.frameId,
            retention_days=payload.retentionDays,
            now=created_at,
        )
        event_id = persistence.event_id
        duplicate_count = persistence.duplicate_count
        deduplicated = persistence.deduplicated
    except Exception:
        event_id = None

    return VisionAnalyzeResponse(
        message=message,
        createdAt=created_at.isoformat(),
        modelId=payload.modelId,
        eventId=event_id,
        duplicateCount=duplicate_count,
        deduplicated=deduplicated,
    )


@app.get("/vision/events", response_model=VisionEventsResponse)
async def vision_events(
    startAt: str | None = None,
    endAt: str | None = None,
    keyword: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> VisionEventsResponse:
    return VisionEventsResponse(
        events=[
            VisionEventRecord.model_validate(event)
            for event in event_store.list_events(
                start_at=parse_optional_datetime(startAt),
                end_at=parse_optional_datetime(endAt),
                keyword=keyword,
                limit=limit,
            )
        ]
    )


@app.get("/vision/events/{event_id}", response_model=VisionEventRecord)
async def vision_event(event_id: str) -> VisionEventRecord:
    event = event_store.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Vision event not found")
    return VisionEventRecord.model_validate(event)


@app.get("/vision/events/{event_id}/screenshot")
async def vision_event_screenshot(event_id: str) -> FileResponse:
    screenshot = event_store.screenshot_file(event_id)
    if screenshot is None:
        raise HTTPException(status_code=404, detail="Vision event screenshot not found")
    path, media_type = screenshot
    return FileResponse(path, media_type=media_type)


def normalize_lm_studio_base_url(raw_url: str) -> str:
    base_url = raw_url.strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=422, detail="LM Studio URL must start with http:// or https://")
    return base_url


def parse_optional_datetime(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    normalized = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid datetime: {raw_value}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def build_vision_prompt(payload: VisionAnalyzeRequest) -> str:
    event_text = "新的人出现在画面中" if payload.eventType == "new_person" else "画面中的人发生了移动"
    detection_lines = [
        (
            f"- {item.label} confidence={item.confidence:.2f} "
            f"box=({item.box.x:.0f},{item.box.y:.0f},{item.box.width:.0f},{item.box.height:.0f})"
        )
        for item in payload.detections[:10]
    ]
    detections_text = "\n".join(detection_lines) if detection_lines else "- none"
    instruction = payload.prompt.strip() if payload.prompt and payload.prompt.strip() else DEFAULT_VISION_PROMPT
    return (
        f"事件：{event_text}。\n"
        f"帧编号：{payload.frameId}。\n"
        f"检测框参考：\n{detections_text}\n"
        f"{instruction}"
    )


def extract_chat_message(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(part for part in parts if part).strip()
    return ""


@app.websocket("/ws/detect")
async def detect_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = DetectFrameRequest.model_validate(await websocket.receive_json())
            started = time.perf_counter()
            detector = registry.detector_for(payload.modelId)
            detections = await detector.detect(
                image_data=payload.imageData,
                width=payload.width,
                height=payload.height,
                confidence_threshold=payload.confidenceThreshold,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            response = DetectFrameResponse(
                frameId=payload.frameId,
                modelId=detector.model_id,
                inferenceMs=round(elapsed_ms, 2),
                detections=detections,
            )
            await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"error": str(exc)})
        await websocket.close(code=1011)
