from __future__ import annotations

import base64
from datetime import datetime
import io
import sqlite3

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.settings_store import AppSettingsStore
from app.vision_store import VisionEventStore


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    models_response = FakeResponse(
        200,
        {"data": [{"id": "qwen/qwen3-v1-4b", "object": "model"}, {"id": "other-model"}]},
    )
    chat_response = FakeResponse(
        200,
        {"choices": [{"message": {"content": "画面中有一名人员正在移动。"}}]},
    )
    last_post_json: dict | None = None
    post_count = 0

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def get(self, url: str) -> FakeResponse:
        assert url == "http://127.0.0.1:1234/v1/models"
        return self.models_response

    async def post(self, url: str, json: dict) -> FakeResponse:
        assert url == "http://127.0.0.1:1234/v1/chat/completions"
        FakeAsyncClient.last_post_json = json
        FakeAsyncClient.post_count += 1
        return self.chat_response


def test_vision_models_lists_lm_studio_models(monkeypatch) -> None:
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)
    client = TestClient(main.app)

    response = client.post("/vision/models", json={"baseUrl": "http://127.0.0.1:1234/v1/"})

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {"id": "qwen/qwen3-v1-4b", "object": "model"},
            {"id": "other-model", "object": None},
        ]
    }


def test_app_settings_persist_to_database(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main, "settings_store", AppSettingsStore(tmp_path))
    client = TestClient(main.app)

    initial = client.get("/settings")
    assert initial.status_code == 200
    assert initial.json()["historyRetentionDays"] == 1

    payload = initial.json()
    payload.update(
        {
            "selectedModelId": "yolo11x-onnx",
            "selectedCameraId": "camera-1",
            "language": "zh",
            "appearance": "dark",
            "lmStudioUrl": "http://127.0.0.1:1234/v1",
            "lmStudioModelId": "vision-model",
            "lmStudioPrompt": "测试提示词",
            "historyRetentionDays": 7,
            "confidenceThreshold": 0.72,
        }
    )
    payload["visionTriggerSettings"]["cooldownSeconds"] = 12

    saved = client.put("/settings", json=payload)
    loaded = client.get("/settings")

    assert saved.status_code == 200
    assert loaded.status_code == 200
    assert loaded.json()["selectedModelId"] == "yolo11x-onnx"
    assert loaded.json()["selectedCameraId"] == "camera-1"
    assert loaded.json()["language"] == "zh"
    assert loaded.json()["appearance"] == "dark"
    assert loaded.json()["lmStudioModelId"] == "vision-model"
    assert loaded.json()["lmStudioPrompt"] == "测试提示词"
    assert loaded.json()["historyRetentionDays"] == 7
    assert loaded.json()["confidenceThreshold"] == 0.72
    assert loaded.json()["visionTriggerSettings"]["cooldownSeconds"] == 12


def test_vision_analyze_returns_chat_message(monkeypatch) -> None:
    FakeAsyncClient.last_post_json = None
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)
    client = TestClient(main.app)

    response = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "prompt": "只输出安全风险摘要。",
            "imageData": "data:image/jpeg;base64,abc",
            "eventType": "person_moved",
            "frameId": 12,
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.9,
                    "box": {"x": 10, "y": 20, "width": 30, "height": 40},
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["message"] == "画面中有一名人员正在移动。"
    assert response.json()["modelId"] == "qwen/qwen3-v1-4b"
    assert FakeAsyncClient.last_post_json is not None
    assert FakeAsyncClient.last_post_json["model"] == "qwen/qwen3-v1-4b"
    prompt = FakeAsyncClient.last_post_json["messages"][1]["content"][0]["text"]
    assert "画面中的人发生了移动" in prompt
    assert "只输出安全风险摘要。" in prompt


def test_vision_models_rejects_invalid_url() -> None:
    client = TestClient(main.app)

    response = client.post("/vision/models", json={"baseUrl": "file:///tmp/model"})

    assert response.status_code == 422
    assert "http://" in response.json()["detail"]


def test_vision_models_reports_lm_studio_error(monkeypatch) -> None:
    class ErrorClient(FakeAsyncClient):
        async def get(self, url: str) -> FakeResponse:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(main.httpx, "AsyncClient", ErrorClient)
    client = TestClient(main.app)

    response = client.post("/vision/models", json={"baseUrl": "http://127.0.0.1:1234/v1"})

    assert response.status_code == 502
    assert "Unable to reach LM Studio" in response.json()["detail"]


def test_vision_analyze_requires_message_content(monkeypatch) -> None:
    class EmptyChatClient(FakeAsyncClient):
        chat_response = FakeResponse(200, {"choices": [{"message": {}}]})

    monkeypatch.setattr(main.httpx, "AsyncClient", EmptyChatClient)
    client = TestClient(main.app)

    response = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": "data:image/jpeg;base64,abc",
            "eventType": "new_person",
            "frameId": 1,
        },
    )

    assert response.status_code == 502
    assert "message content" in response.json()["detail"]


def test_vision_analyze_deduplicates_recent_same_event_type(monkeypatch, tmp_path) -> None:
    FakeAsyncClient.post_count = 0
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(main, "event_store", VisionEventStore(tmp_path))
    client = TestClient(main.app)
    image_data = make_image_data()

    first = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": image_data,
            "eventType": "new_person",
            "frameId": 1,
            "sessionId": "session-a",
            "trackId": 1,
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.9,
                    "box": {"x": 10, "y": 20, "width": 30, "height": 40},
                }
            ],
        },
    )
    second = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": image_data,
            "eventType": "new_person",
            "frameId": 2,
            "sessionId": "session-a",
            "trackId": 99,
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.9,
                    "box": {"x": 12, "y": 22, "width": 30, "height": 40},
                }
            ],
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["eventId"] == second.json()["eventId"]
    assert second.json()["deduplicated"] is True
    assert second.json()["duplicateCount"] == 1
    assert FakeAsyncClient.post_count == 1

    events_response = client.get("/vision/events")
    assert events_response.status_code == 200
    events = events_response.json()["events"]
    assert len(events) == 1
    assert events[0]["duplicateCount"] == 1


def test_vision_analyze_does_not_deduplicate_when_person_count_increases(monkeypatch, tmp_path) -> None:
    FakeAsyncClient.post_count = 0
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(main, "event_store", VisionEventStore(tmp_path))
    client = TestClient(main.app)
    image_data = make_image_data()

    first = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": image_data,
            "eventType": "new_person",
            "frameId": 1,
            "sessionId": "session-a",
            "detections": [
                {"label": "person", "confidence": 0.9, "box": {"x": 10, "y": 20, "width": 30, "height": 40}},
            ],
        },
    )
    second = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": image_data,
            "eventType": "new_person",
            "frameId": 2,
            "sessionId": "session-a",
            "detections": [
                {"label": "person", "confidence": 0.9, "box": {"x": 10, "y": 20, "width": 30, "height": 40}},
                {"label": "person", "confidence": 0.88, "box": {"x": 60, "y": 20, "width": 28, "height": 38}},
            ],
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["eventId"] != second.json()["eventId"]
    assert second.json()["deduplicated"] is False
    assert FakeAsyncClient.post_count == 2

    events_response = client.get("/vision/events")
    assert len(events_response.json()["events"]) == 2


def test_vision_analyze_final_text_check_merges_same_clothing_and_action(monkeypatch, tmp_path) -> None:
    class SimilarTextClient(FakeAsyncClient):
        messages = ["门口有一名穿黑色上衣的人员站立，没有风险。", "门口有一名穿黑色上衣的人员停留，没有风险。"]
        post_count = 0

        async def post(self, url: str, json: dict) -> FakeResponse:
            self.__class__.post_count += 1
            return FakeResponse(200, {"choices": [{"message": {"content": self.messages.pop(0)}}]})

    monkeypatch.setattr(main.httpx, "AsyncClient", SimilarTextClient)
    monkeypatch.setattr(main, "event_store", VisionEventStore(tmp_path))
    client = TestClient(main.app)

    first = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": make_image_data("split"),
            "eventType": "new_person",
            "frameId": 1,
            "sessionId": "session-a",
            "detections": [
                {"label": "person", "confidence": 0.9, "box": {"x": 10, "y": 20, "width": 30, "height": 40}},
            ],
        },
    )
    second = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": make_image_data("inverse-split"),
            "eventType": "new_person",
            "frameId": 2,
            "sessionId": "session-a",
            "detections": [
                {"label": "person", "confidence": 0.9, "box": {"x": 12, "y": 21, "width": 30, "height": 40}},
            ],
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["eventId"] == second.json()["eventId"]
    assert second.json()["deduplicated"] is True
    assert SimilarTextClient.post_count == 2
    assert len(client.get("/vision/events").json()["events"]) == 1


def test_vision_analyze_final_text_check_keeps_risk_change(monkeypatch, tmp_path) -> None:
    class RiskChangeClient(FakeAsyncClient):
        messages = ["门口有一名穿黑色上衣的人员站立，没有风险。", "门口有一名穿黑色上衣的人员摔倒，有风险。"]

        async def post(self, url: str, json: dict) -> FakeResponse:
            return FakeResponse(200, {"choices": [{"message": {"content": self.messages.pop(0)}}]})

    monkeypatch.setattr(main.httpx, "AsyncClient", RiskChangeClient)
    monkeypatch.setattr(main, "event_store", VisionEventStore(tmp_path))
    client = TestClient(main.app)

    first = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": make_image_data("split"),
            "eventType": "new_person",
            "frameId": 1,
            "sessionId": "session-a",
            "detections": [
                {"label": "person", "confidence": 0.9, "box": {"x": 10, "y": 20, "width": 30, "height": 40}},
            ],
        },
    )
    second = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": make_image_data("inverse-split"),
            "eventType": "new_person",
            "frameId": 2,
            "sessionId": "session-a",
            "detections": [
                {"label": "person", "confidence": 0.9, "box": {"x": 12, "y": 21, "width": 30, "height": 40}},
            ],
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["eventId"] != second.json()["eventId"]
    assert second.json()["deduplicated"] is False
    assert len(client.get("/vision/events").json()["events"]) == 2


def test_vision_event_store_migrates_deduplication_columns(tmp_path) -> None:
    db_path = tmp_path / "seeya.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE vision_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                track_id INTEGER,
                event_type TEXT NOT NULL,
                model_id TEXT NOT NULL,
                frame_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                summary TEXT NOT NULL,
                detections_json TEXT NOT NULL,
                primary_box_json TEXT,
                message_fingerprint TEXT NOT NULL,
                image_fingerprint TEXT NOT NULL,
                duplicate_count INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                screenshot_path TEXT,
                screenshot_mime_type TEXT,
                screenshot_size_bytes INTEGER NOT NULL DEFAULT 0,
                screenshot_width INTEGER NOT NULL DEFAULT 0,
                screenshot_height INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    store = VisionEventStore(tmp_path)
    with store.connect() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(vision_events)").fetchall()}

    assert {"person_count", "person_boxes_json", "message_features_json"} <= columns


def test_vision_event_screenshot_returns_saved_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(main, "event_store", VisionEventStore(tmp_path))
    client = TestClient(main.app)

    response = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": make_image_data(),
            "eventType": "person_moved",
            "frameId": 3,
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.92,
                    "box": {"x": 40, "y": 40, "width": 40, "height": 60},
                }
            ],
        },
    )

    event_id = response.json()["eventId"]
    screenshot = client.get(f"/vision/events/{event_id}/screenshot")

    assert screenshot.status_code == 200
    assert screenshot.headers["content-type"] in {"image/webp", "image/jpeg"}
    assert len(screenshot.content) > 0


def test_vision_analyze_uses_retention_days_for_event_expiry(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(main, "event_store", VisionEventStore(tmp_path))
    client = TestClient(main.app)

    response = client.post(
        "/vision/analyze",
        json={
            "baseUrl": "http://127.0.0.1:1234/v1",
            "modelId": "qwen/qwen3-v1-4b",
            "imageData": make_image_data(),
            "eventType": "person_moved",
            "frameId": 9,
            "retentionDays": 3,
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.92,
                    "box": {"x": 40, "y": 40, "width": 40, "height": 60},
                }
            ],
        },
    )

    assert response.status_code == 200
    event_id = response.json()["eventId"]
    event = client.get(f"/vision/events/{event_id}").json()
    created_at = datetime.fromisoformat(event["createdAt"])
    expires_at = datetime.fromisoformat(event["expiresAt"])
    assert round((expires_at - created_at).total_seconds()) == 3 * 24 * 60 * 60


def test_vision_events_filters_by_keyword(monkeypatch, tmp_path) -> None:
    class KeywordClient(FakeAsyncClient):
        messages = ["门口有一名人员停留。", "走廊无人异常。"]

        async def post(self, url: str, json: dict) -> FakeResponse:
            message = self.messages.pop(0)
            return FakeResponse(200, {"choices": [{"message": {"content": message}}]})

    monkeypatch.setattr(main.httpx, "AsyncClient", KeywordClient)
    monkeypatch.setattr(main, "event_store", VisionEventStore(tmp_path))
    client = TestClient(main.app)
    image_data = make_image_data()

    for frame_id in (1, 2):
        response = client.post(
            "/vision/analyze",
            json={
                "baseUrl": "http://127.0.0.1:1234/v1",
                "modelId": "qwen/qwen3-v1-4b",
                "imageData": image_data,
                "eventType": "new_person",
                "frameId": frame_id,
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.9,
                        "box": {"x": 10 * frame_id, "y": 20, "width": 30, "height": 40},
                    }
                ],
            },
        )
        assert response.status_code == 200

    events_response = client.get("/vision/events", params={"keyword": "门口"})

    assert events_response.status_code == 200
    events = events_response.json()["events"]
    assert len(events) == 1
    assert "门口" in events[0]["message"]


def make_image_data(variant: str = "solid") -> str:
    image = Image.new("RGB", (96, 64), color=(32, 64, 96))
    if variant == "split":
        image.paste((240, 240, 240), (0, 0, 48, 64))
        image.paste((16, 16, 16), (48, 0, 96, 64))
    if variant == "inverse-split":
        image.paste((16, 16, 16), (0, 0, 48, 64))
        image.paste((240, 240, 240), (48, 0, 96, 64))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
