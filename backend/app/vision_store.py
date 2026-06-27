from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid

from PIL import Image, ImageOps

from app.schemas import Box, Detection


RETENTION_HOURS = int(os.environ.get("SEEYA_VISION_RETENTION_HOURS", "48"))
MAX_IMAGE_SIDE = 960
IMAGE_HASH_SIMILAR_DISTANCE = 6
TEXT_SIMILARITY_THRESHOLD = 0.85
BOX_CENTER_DISTANCE_RATIO = 0.15
BOX_IOU_THRESHOLD = 0.35
DEDUPLICATION_CANDIDATE_LIMIT = 10
RISK_TERMS = ("风险", "危险", "摔倒", "跌倒", "火", "烟", "入侵", "异常", "求助", "受伤", "警告")


@dataclass(frozen=True)
class PreparedScreenshot:
    content: bytes
    mime_type: str
    extension: str
    width: int
    height: int
    image_fingerprint: str


@dataclass(frozen=True)
class VisionPersistenceResult:
    event_id: str | None
    duplicate_count: int
    deduplicated: bool


class VisionEventStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        default_data_dir = Path(__file__).resolve().parents[1] / "data"
        self.data_dir = data_dir or Path(os.environ.get("SEEYA_DATA_DIR", default_data_dir))
        self.db_path = self.data_dir / "seeya.db"
        self.screenshot_dir = self.data_dir / "screenshots"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.init_db()
        self.cleanup_expired()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vision_events (
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
            self._ensure_columns(connection)
            connection.execute("CREATE INDEX IF NOT EXISTS idx_vision_events_type_created ON vision_events(event_type, created_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_vision_events_expires ON vision_events(expires_at)")

    def _ensure_columns(self, connection: sqlite3.Connection) -> None:
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(vision_events)").fetchall()}
        migrations = {
            "model_id": "ALTER TABLE vision_events ADD COLUMN model_id TEXT NOT NULL DEFAULT ''",
            "frame_id": "ALTER TABLE vision_events ADD COLUMN frame_id INTEGER NOT NULL DEFAULT 0",
            "screenshot_mime_type": "ALTER TABLE vision_events ADD COLUMN screenshot_mime_type TEXT",
            "screenshot_width": "ALTER TABLE vision_events ADD COLUMN screenshot_width INTEGER NOT NULL DEFAULT 0",
            "screenshot_height": "ALTER TABLE vision_events ADD COLUMN screenshot_height INTEGER NOT NULL DEFAULT 0",
        }
        for column, statement in migrations.items():
            if column not in existing:
                connection.execute(statement)

    def cleanup_expired(self, now: datetime | None = None) -> int:
        current = now or datetime.now(UTC)
        current_iso = current.isoformat()
        deleted = 0
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, screenshot_path FROM vision_events WHERE expires_at <= ?",
                (current_iso,),
            ).fetchall()
            for row in rows:
                if row["screenshot_path"]:
                    self._delete_screenshot(row["screenshot_path"])
            connection.execute("DELETE FROM vision_events WHERE expires_at <= ?", (current_iso,))
            deleted = len(rows)
        return deleted

    def save_or_merge_event(
        self,
        *,
        session_id: str | None,
        track_id: int | None,
        event_type: str,
        message: str,
        detections: list[Detection],
        image_data: str,
        model_id: str,
        frame_id: int,
        now: datetime | None = None,
    ) -> VisionPersistenceResult:
        current = now or datetime.now(UTC)
        self.cleanup_expired(current)
        screenshot = prepare_screenshot(image_data)
        primary_box = select_primary_box(detections)
        detections_json = json.dumps([item.model_dump() for item in detections], ensure_ascii=False)
        primary_box_json = json.dumps(primary_box.model_dump(), ensure_ascii=False) if primary_box else None
        message_fingerprint = normalize_message(message)
        summary = summarize_message(message)

        with self.connect() as connection:
            candidates = connection.execute(
                """
                SELECT * FROM vision_events
                WHERE event_type = ? AND expires_at > ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (event_type, current.isoformat(), DEDUPLICATION_CANDIDATE_LIMIT),
            ).fetchall()
            matching = find_matching_event(
                candidates=candidates,
                event_type=event_type,
                message=message,
                message_fingerprint=message_fingerprint,
                image_fingerprint=screenshot.image_fingerprint,
                primary_box=primary_box,
                frame_width=screenshot.width,
                frame_height=screenshot.height,
            )
            if matching is not None:
                duplicate_count = int(matching["duplicate_count"]) + 1
                connection.execute(
                    """
                    UPDATE vision_events
                    SET duplicate_count = ?, last_seen_at = ?
                    WHERE id = ?
                    """,
                    (duplicate_count, current.isoformat(), matching["id"]),
                )
                return VisionPersistenceResult(
                    event_id=matching["id"],
                    duplicate_count=duplicate_count,
                    deduplicated=True,
                )

            event_id = str(uuid.uuid4())
            screenshot_path = self._write_screenshot(event_id, current, screenshot)
            connection.execute(
                """
                INSERT INTO vision_events (
                    id, session_id, track_id, event_type, model_id, frame_id, message, summary, detections_json,
                    primary_box_json, message_fingerprint, image_fingerprint, duplicate_count,
                    first_seen_at, last_seen_at, created_at, expires_at, screenshot_path,
                    screenshot_mime_type, screenshot_size_bytes, screenshot_width, screenshot_height
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    track_id,
                    event_type,
                    model_id,
                    frame_id,
                    message,
                    summary,
                    detections_json,
                    primary_box_json,
                    message_fingerprint,
                    screenshot.image_fingerprint,
                    0,
                    current.isoformat(),
                    current.isoformat(),
                    current.isoformat(),
                    (current + timedelta(hours=RETENTION_HOURS)).isoformat(),
                    str(screenshot_path.relative_to(self.data_dir)),
                    screenshot.mime_type,
                    len(screenshot.content),
                    screenshot.width,
                    screenshot.height,
                ),
            )
            return VisionPersistenceResult(event_id=event_id, duplicate_count=0, deduplicated=False)

    def list_events(self, start_at: datetime | None, end_at: datetime | None, limit: int) -> list[dict]:
        self.cleanup_expired()
        limited = max(1, min(limit, 500))
        clauses = ["expires_at > ?"]
        params: list[object] = [datetime.now(UTC).isoformat()]
        if start_at is not None:
            clauses.append("created_at >= ?")
            params.append(start_at.isoformat())
        if end_at is not None:
            clauses.append("created_at <= ?")
            params.append(end_at.isoformat())
        params.append(limited)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM vision_events
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [event_row_to_dict(row) for row in rows]

    def get_event(self, event_id: str) -> dict | None:
        self.cleanup_expired()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM vision_events WHERE id = ? AND expires_at > ?",
                (event_id, datetime.now(UTC).isoformat()),
            ).fetchone()
        return event_row_to_dict(row) if row else None

    def screenshot_file(self, event_id: str) -> tuple[Path, str] | None:
        event = self.get_event(event_id)
        if not event or not event["screenshotPath"]:
            return None
        path = self.data_dir / event["screenshotPath"]
        if not path.is_file():
            return None
        return path, event["screenshotMimeType"] or "application/octet-stream"

    def _write_screenshot(self, event_id: str, created_at: datetime, screenshot: PreparedScreenshot) -> Path:
        day_dir = self.screenshot_dir / created_at.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{event_id}.{screenshot.extension}"
        path.write_bytes(screenshot.content)
        return path

    def _delete_screenshot(self, screenshot_path: str) -> None:
        try:
            (self.data_dir / screenshot_path).unlink(missing_ok=True)
        except OSError:
            return


def prepare_screenshot(image_data: str) -> PreparedScreenshot:
    payload = image_data.split(",", 1)[1] if "," in image_data else image_data
    raw = base64.b64decode(payload, validate=False)
    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        width, height = image.size
        image_fingerprint = average_hash(image)

        webp_buffer = io.BytesIO()
        try:
            image.save(webp_buffer, format="WEBP", quality=70, method=4)
            return PreparedScreenshot(
                content=webp_buffer.getvalue(),
                mime_type="image/webp",
                extension="webp",
                width=width,
                height=height,
                image_fingerprint=image_fingerprint,
            )
        except Exception:
            jpeg_buffer = io.BytesIO()
            image.save(jpeg_buffer, format="JPEG", quality=75, optimize=True)
            return PreparedScreenshot(
                content=jpeg_buffer.getvalue(),
                mime_type="image/jpeg",
                extension="jpg",
                width=width,
                height=height,
                image_fingerprint=image_fingerprint,
            )


def average_hash(image: Image.Image) -> str:
    small = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    average = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= average else "0" for pixel in pixels)
    return f"{int(bits, 2):016x}"


def hamming_distance(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 64


def normalize_message(message: str) -> str:
    normalized = message.lower()
    normalized = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?", "", normalized)
    normalized = re.sub(r"frame\s*\d+|帧编号[:：]?\s*\d+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
    return normalized


def summarize_message(message: str) -> str:
    normalized = re.sub(r"\s+", " ", message).strip()
    return f"{normalized[:56]}..." if len(normalized) > 56 else normalized


def text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def select_primary_box(detections: list[Detection]) -> Box | None:
    person_detections = [item for item in detections if item.label.lower() == "person"]
    candidates = person_detections or detections
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.confidence).box


def boxes_are_close(left: Box | None, right: Box | None, frame_width: int, frame_height: int) -> bool:
    if left is None or right is None:
        return False
    frame_diagonal = max(1.0, (frame_width**2 + frame_height**2) ** 0.5)
    return box_center_distance(left, right) / frame_diagonal <= BOX_CENTER_DISTANCE_RATIO or box_iou(left, right) >= BOX_IOU_THRESHOLD


def box_center_distance(left: Box, right: Box) -> float:
    return ((left.x + left.width / 2 - right.x - right.width / 2) ** 2 + (left.y + left.height / 2 - right.y - right.height / 2) ** 2) ** 0.5


def box_iou(left: Box, right: Box) -> float:
    left_x2 = left.x + left.width
    left_y2 = left.y + left.height
    right_x2 = right.x + right.width
    right_y2 = right.y + right.height
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left.x, right.x))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left.y, right.y))
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def find_matching_event(
    *,
    candidates: list[sqlite3.Row],
    event_type: str,
    message: str,
    message_fingerprint: str,
    image_fingerprint: str,
    primary_box: Box | None,
    frame_width: int,
    frame_height: int,
) -> sqlite3.Row | None:
    for candidate in candidates:
        if has_significant_text_change(message, candidate["message"]):
            continue
        image_similar = hamming_distance(image_fingerprint, candidate["image_fingerprint"]) <= IMAGE_HASH_SIMILAR_DISTANCE
        text_similar = text_similarity(message_fingerprint, candidate["message_fingerprint"]) >= TEXT_SIMILARITY_THRESHOLD
        candidate_box = parse_box(candidate["primary_box_json"])
        box_close = boxes_are_close(primary_box, candidate_box, frame_width, frame_height)
        if event_type == "new_person" and image_similar and text_similar:
            return candidate
        if event_type == "person_moved" and image_similar and text_similar and box_close:
            return candidate
    return None


def has_significant_text_change(current: str, previous: str) -> bool:
    if risk_polarity(current) != risk_polarity(previous):
        return True
    current_terms = {term for term in RISK_TERMS if term in current}
    previous_terms = {term for term in RISK_TERMS if term in previous}
    if current_terms != previous_terms:
        return True
    current_people = extract_people_count(current)
    previous_people = extract_people_count(previous)
    return current_people is not None and previous_people is not None and current_people != previous_people


def risk_polarity(message: str) -> str:
    if re.search(r"(无|未发现|没有|暂无).{0,4}(风险|危险|异常)", message):
        return "negative"
    if any(term in message for term in RISK_TERMS):
        return "positive"
    return "neutral"


def extract_people_count(message: str) -> int | None:
    match = re.search(r"(\d+)\s*(?:个|名)?\s*(?:人|人员)", message)
    if match:
        return int(match.group(1))
    chinese_counts = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}
    match = re.search(r"([一二两三四五])\s*(?:个|名)?\s*(?:人|人员)", message)
    if match:
        return chinese_counts[match.group(1)]
    return None


def parse_box(raw_box: str | None) -> Box | None:
    if not raw_box:
        return None
    try:
        return Box.model_validate(json.loads(raw_box))
    except Exception:
        return None


def event_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "sessionId": row["session_id"],
        "trackId": row["track_id"],
        "eventType": row["event_type"],
        "modelId": row["model_id"],
        "frameId": row["frame_id"],
        "message": row["message"],
        "summary": row["summary"],
        "detections": json.loads(row["detections_json"]),
        "primaryBox": json.loads(row["primary_box_json"]) if row["primary_box_json"] else None,
        "duplicateCount": row["duplicate_count"],
        "firstSeenAt": row["first_seen_at"],
        "lastSeenAt": row["last_seen_at"],
        "createdAt": row["created_at"],
        "expiresAt": row["expires_at"],
        "hasScreenshot": bool(row["screenshot_path"]),
        "screenshotPath": row["screenshot_path"],
        "screenshotMimeType": row["screenshot_mime_type"],
        "screenshotSizeBytes": row["screenshot_size_bytes"],
        "screenshotWidth": row["screenshot_width"],
        "screenshotHeight": row["screenshot_height"],
    }
