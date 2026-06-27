from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sqlite3

from app.schemas import AppSettings


SETTINGS_KEY = "frontend"


class AppSettingsStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        default_data_dir = Path(__file__).resolve().parents[1] / "data"
        self.data_dir = data_dir or Path(os.environ.get("SEEYA_DATA_DIR", default_data_dir))
        self.db_path = self.data_dir / "seeya.db"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def get_settings(self) -> AppSettings:
        with self.connect() as connection:
            row = connection.execute("SELECT value_json FROM app_settings WHERE key = ?", (SETTINGS_KEY,)).fetchone()
        if row is None:
            return AppSettings()
        try:
            return AppSettings.model_validate(json.loads(row["value_json"]))
        except Exception:
            return AppSettings()

    def save_settings(self, settings: AppSettings) -> AppSettings:
        normalized = AppSettings.model_validate(settings.model_dump())
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO app_settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (
                    SETTINGS_KEY,
                    normalized.model_dump_json(),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return normalized
