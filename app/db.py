import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    upload_path TEXT,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT
                )
                """
            )

    def create(
        self,
        job_id: str,
        idempotency_key: str,
        request_hash: str,
        request_data: dict[str, Any],
        upload_path: str | None,
        correlation_id: str,
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, idempotency_key, request_hash, request_json, upload_path,
                    status, progress, created_at, updated_at, correlation_id
                ) VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?, ?)
                """,
                (
                    job_id,
                    idempotency_key,
                    request_hash,
                    json.dumps(request_data, ensure_ascii=False),
                    upload_path,
                    now,
                    now,
                    correlation_id,
                ),
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE idempotency_key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def requeue_interrupted(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'queued', updated_at = ? WHERE status = 'running'",
                (utc_now(),),
            )
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        changes: list[str] = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        for column, value in (
            ("status", status),
            ("progress", progress),
            ("result_json", json.dumps(result) if result is not None else None),
            ("error_json", json.dumps(error) if error is not None else None),
        ):
            if value is not None:
                changes.append(f"{column} = ?")
                values.append(value)
        values.append(job_id)
        with self.connect() as connection:
            connection.execute(f"UPDATE jobs SET {', '.join(changes)} WHERE id = ?", values)
