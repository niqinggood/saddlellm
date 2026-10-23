"""Durable, model-scoped chat history for the built-in serving page."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


class ChatHistoryStore:
    """Persist browser conversations beneath the serve process's startup cwd."""

    def __init__(self, startup_dir: Path):
        self.path = Path(startup_dir).resolve() / ".saddlellm" / "chat_history.sqlite3"

    @staticmethod
    def _session_id(value: str) -> str:
        try:
            canonical = str(uuid.UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("session_id must be a canonical UUID") from exc
        if canonical != value:
            raise ValueError("session_id must be a canonical UUID")
        return canonical

    def _connect(self, *, create: bool = False) -> sqlite3.Connection | None:
        if create:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        elif not self.path.is_file():
            return None
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        if create:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS conversations ("
                "id TEXT PRIMARY KEY, model TEXT NOT NULL, title TEXT NOT NULL, "
                "messages_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS conversations_model_updated "
                "ON conversations(model, updated_at DESC)"
            )
            connection.commit()
        return connection

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, str]:
        return {key: row[key] for key in ("id", "model", "title", "created_at", "updated_at")}

    def list_sessions(self, model: str) -> list[dict[str, str]]:
        connection = self._connect()
        if connection is None:
            return []
        try:
            rows = connection.execute(
                "SELECT id, model, title, created_at, updated_at FROM conversations "
                "WHERE model=? ORDER BY updated_at DESC, id DESC", (model,)
            ).fetchall()
            return [self._summary(row) for row in rows]
        finally:
            connection.close()

    def get_session(self, session_id: str, model: str) -> dict[str, Any] | None:
        session_id = self._session_id(session_id)
        connection = self._connect()
        if connection is None:
            return None
        try:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id=? AND model=?", (session_id, model)
            ).fetchone()
            return None if row is None else {
                **self._summary(row), "messages": json.loads(row["messages_json"])
            }
        finally:
            connection.close()

    def save_session(
        self, session_id: str, model: str, messages: Sequence[Mapping[str, Any]]
    ) -> dict[str, str]:
        session_id = self._session_id(session_id)
        if not isinstance(model, str) or not model:
            raise ValueError("model must not be empty")
        if not 2 <= len(messages) <= 200:
            raise ValueError("history must contain 2 to 200 messages")
        cleaned = []
        for message in messages:
            role, content = message.get("role"), message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
                raise ValueError("history messages require non-empty user/assistant text")
            cleaned.append({"role": role, "content": content})
        if cleaned[0]["role"] != "user" or cleaned[-1]["role"] != "assistant":
            raise ValueError("history must start with a user and end with an assistant")
        encoded = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 2_000_000:
            raise ValueError("history exceeds the 2 MB session limit")
        title = " ".join(cleaned[0]["content"].split())[:80] or "新对话"
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        connection = self._connect(create=True)
        assert connection is not None
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT model, created_at FROM conversations WHERE id=?", (session_id,)
                ).fetchone()
                if existing is not None and existing["model"] != model:
                    raise ValueError("session_id belongs to another model")
                created_at = existing["created_at"] if existing else now
                connection.execute(
                    "INSERT INTO conversations(id, model, title, messages_json, created_at, updated_at) "
                    "VALUES(?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET title=excluded.title, "
                    "messages_json=excluded.messages_json, updated_at=excluded.updated_at",
                    (session_id, model, title, encoded, created_at, now),
                )
            return {"id": session_id, "model": model, "title": title,
                    "created_at": created_at, "updated_at": now}
        finally:
            connection.close()

    def delete_session(self, session_id: str, model: str) -> bool:
        session_id = self._session_id(session_id)
        connection = self._connect()
        if connection is None:
            return False
        try:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM conversations WHERE id=? AND model=?", (session_id, model)
                )
            return cursor.rowcount > 0
        finally:
            connection.close()
