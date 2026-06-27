from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.model_registry import ModelRegistry
from app.schemas import (
    DetectFrameRequest,
    DetectFrameResponse,
    HealthResponse,
    ModelsResponse,
    SelectModelRequest,
)

app = FastAPI(title="Seeya API")
registry = ModelRegistry()

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
