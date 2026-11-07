"""Persistent conversations and their message history."""

import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any, Iterator
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "chroma" / "conversations.sqlite3"
DEFAULT_CONVERSATION_ID = "default"


class ConversationService:
    """Store conversation metadata and chat messages in a small SQLite DB."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id)
                        ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO conversations (id, title) VALUES (?, ?)",
                (DEFAULT_CONVERSATION_ID, "My first conversation"),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, title: str = "New conversation") -> dict[str, str]:
        clean_title = " ".join(title.split())[:120] or "New conversation"
        conversation_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (conversation_id, clean_title),
            )
        return self.get(conversation_id)  # type: ignore[return-value]

    def get(self, conversation_id: str) -> dict[str, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, created_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at FROM conversations "
                "ORDER BY created_at, id"
            ).fetchall()
        return [dict(row) for row in rows]

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages (conversation_id, role, content, sources_json) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, json.dumps(sources or [])),
            )
            row = connection.execute(
                "SELECT id, conversation_id, role, content, sources_json, created_at "
                "FROM messages WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return self._message_dict(row)

    def messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, conversation_id, role, content, sources_json, created_at "
                "FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [self._message_dict(row) for row in rows]

    def delete(self, conversation_id: str) -> bool:
        if conversation_id == DEFAULT_CONVERSATION_ID:
            raise ValueError("The default conversation cannot be deleted")
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        return cursor.rowcount > 0

    @staticmethod
    def _message_dict(row: sqlite3.Row) -> dict[str, Any]:
        message = dict(row)
        message["sources"] = json.loads(message.pop("sources_json"))
        return message
