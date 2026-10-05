from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Any, Literal
import json
import sqlite3
import struct
import uuid

from agent.embeddings import EmbeddingProvider


ChatRoleFilter = Literal["user", "agent"] | None
EmbeddingView = Literal["all", "user", "agent"]
_CHUNK_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _pack_vector(vector: list[float]) -> bytes:
    if not vector:
        raise ValueError("embedding vector cannot be empty")
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(blob: bytes, dimensions: int) -> list[float]:
    if dimensions < 1:
        raise ValueError("embedding dimensions must be positive")
    expected = dimensions * 4
    if len(blob) != expected:
        raise ValueError("stored embedding has an invalid byte length")
    return list(struct.unpack(f"<{dimensions}f", blob))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding dimensions do not match")

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class StoredChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    active_task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "active_task_id": self.active_task_id,
        }


@dataclass(frozen=True)
class ChatChunk:
    session_id: str
    chunk_index: int
    start_sequence: int
    end_sequence: int
    all_text: str
    user_text: str
    agent_text: str

    def text_for_view(self, view: EmbeddingView) -> str:
        if view == "user":
            return self.user_text
        if view == "agent":
            return self.agent_text
        return self.all_text


@dataclass(frozen=True)
class _SearchableMessage:
    sequence: int
    role: Literal["user", "assistant"]
    content: str

    @property
    def estimated_tokens(self) -> int:
        # Avoid binding persistence to a specific model tokenizer. Four UTF-8-ish
        # characters per token is conservative enough for chunk sizing and the
        # actual embedding backend remains authoritative about its context limit.
        return max(1, (len(self.content) + 3) // 4) + 2


class ChatSessionAccess:
    """Ephemeral capability set controlling cross-session reads/searches."""

    def __init__(self):
        self.allow_all = False
        self.allowed_ids: set[str] = set()
        self.denied_ids: set[str] = set()

    def reset(self) -> None:
        self.allow_all = False
        self.allowed_ids.clear()
        self.denied_ids.clear()

    def allow_session(self, session_id: str) -> None:
        self.denied_ids.discard(session_id)
        if not self.allow_all:
            self.allowed_ids.add(session_id)

    def deny_session(self, session_id: str) -> None:
        if self.allow_all:
            self.denied_ids.add(session_id)
        else:
            self.allowed_ids.discard(session_id)

    def allow_all_sessions(self) -> None:
        self.allow_all = True
        self.allowed_ids.clear()
        self.denied_ids.clear()

    def deny_all_sessions(self) -> None:
        self.reset()

    def forget_session(self, session_id: str) -> None:
        self.allowed_ids.discard(session_id)
        self.denied_ids.discard(session_id)

    def can_read(self, session_id: str) -> bool:
        if self.allow_all:
            return session_id not in self.denied_ids
        return session_id in self.allowed_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "all" if self.allow_all else "selected",
            "allowed_session_ids": sorted(self.allowed_ids),
            "denied_session_ids": sorted(self.denied_ids),
        }


class ChatSessionStore:
    """SQLite-backed persistent chat sessions and model-visible message history."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0
                        CHECK (message_count >= 0),
                    active_task_id TEXT
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK (sequence >= 1),
                    created_at TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence),
                    FOREIGN KEY (session_id)
                        REFERENCES chat_sessions(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_chunks (
                    session_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                    start_sequence INTEGER NOT NULL,
                    end_sequence INTEGER NOT NULL,
                    all_text TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    agent_text TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    PRIMARY KEY (session_id, chunk_index),
                    FOREIGN KEY (session_id)
                        REFERENCES chat_sessions(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_chunk_embeddings (
                    session_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    view TEXT NOT NULL CHECK (view IN ('all', 'user', 'agent')),
                    text_hash TEXT NOT NULL,
                    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (session_id, chunk_index, view),
                    FOREIGN KEY (session_id, chunk_index)
                        REFERENCES chat_chunks(session_id, chunk_index)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_index_state (
                    session_id TEXT PRIMARY KEY,
                    chunked_message_count INTEGER NOT NULL
                        CHECK (chunked_message_count >= 0),
                    chunk_version INTEGER NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES chat_sessions(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
                    ON chat_sessions(updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                    ON chat_messages(session_id, sequence);

                CREATE INDEX IF NOT EXISTS idx_chat_chunks_session
                    ON chat_chunks(session_id, chunk_index);

                CREATE INDEX IF NOT EXISTS idx_chat_embeddings_view
                    ON chat_chunk_embeddings(view, session_id, chunk_index);
                """
            )

    @staticmethod
    def _validate_title(title: str) -> str:
        title = " ".join(title.split())
        if not title:
            raise ValueError("session title cannot be empty")
        if len(title) > 120:
            raise ValueError("session title cannot exceed 120 characters")
        return title

    @staticmethod
    def _validate_message(message: dict[str, Any]) -> tuple[str, str]:
        if not isinstance(message, dict):
            raise TypeError("chat message must be a dictionary")

        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("chat message role must be a non-empty string")

        try:
            encoded = json.dumps(
                message,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "chat message must be JSON serializable"
            ) from exc

        return role, encoded

    def create_session(
        self,
        title: str,
        *,
        active_task_id: str | None = None,
    ) -> StoredChatSession:
        title = self._validate_title(title)
        session_id = f"session_{uuid.uuid4().hex}"
        now = _utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (
                    id,
                    title,
                    created_at,
                    updated_at,
                    message_count,
                    active_task_id
                ) VALUES (?, ?, ?, ?, 0, ?)
                """,
                (
                    session_id,
                    title,
                    now,
                    now,
                    active_task_id,
                ),
            )

        return self.get_session(session_id)

    def get_session(self, session_id: str) -> StoredChatSession:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"unknown chat session: {session_id}")

        return self._session_from_row(row)

    def resolve_session_id(self, reference: str) -> str:
        reference = reference.strip()
        if not reference:
            raise ValueError("session reference cannot be empty")

        allowed = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-"
        )
        if any(character not in allowed for character in reference):
            raise ValueError("session reference contains invalid characters")

        with self._connect() as connection:
            exact = connection.execute(
                "SELECT id FROM chat_sessions WHERE id = ?",
                (reference,),
            ).fetchone()

            if exact is not None:
                return str(exact["id"])

            if len(reference) < 8:
                raise ValueError(
                    "session prefix must contain at least 8 characters"
                )

            rows = connection.execute(
                """
                SELECT id
                FROM chat_sessions
                WHERE id LIKE ?
                ORDER BY updated_at DESC
                LIMIT 2
                """,
                (f"{reference}%",),
            ).fetchall()

        if not rows:
            raise KeyError(f"unknown chat session: {reference}")
        if len(rows) > 1:
            raise ValueError(
                f"ambiguous chat session prefix: {reference}"
            )

        return str(rows[0]["id"])

    def list_sessions(
        self,
        *,
        limit: int = 10,
        before: str | None = None,
        query: str | None = None,
    ) -> list[StoredChatSession]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")

        clauses: list[str] = []
        parameters: list[Any] = []
        if before is not None:
            clauses.append("updated_at < ?")
            parameters.append(before)
        if query is not None:
            query = query.strip()
            if not query:
                raise ValueError("query cannot be empty")
            clauses.append("title LIKE ? COLLATE NOCASE")
            parameters.append(f"%{query}%")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM chat_sessions
                {where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        return [self._session_from_row(row) for row in rows]

    def all_session_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM chat_sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        self.get_session(session_id)

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT message_json
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()

        messages: list[dict[str, Any]] = []
        for row in rows:
            message = json.loads(row["message_json"])
            if not isinstance(message, dict):
                raise ValueError(
                    "stored chat message is not a JSON object"
                )
            messages.append(message)

        return messages

    def read_messages(
        self,
        session_id: str,
        *,
        limit: int = 20,
        before_sequence: int | None = None,
        around_sequence: int | None = None,
        role: ChatRoleFilter = None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if before_sequence is not None and before_sequence < 1:
            raise ValueError("before_sequence must be at least 1")
        if around_sequence is not None and around_sequence < 1:
            raise ValueError("around_sequence must be at least 1")
        if before_sequence is not None and around_sequence is not None:
            raise ValueError(
                "before_sequence and around_sequence are mutually exclusive"
            )
        if role not in {None, "user", "agent"}:
            raise ValueError("role must be 'user', 'agent', or null")

        role_clause = "role IN ('user', 'assistant')"
        if role == "user":
            role_clause = "role = 'user'"
        elif role == "agent":
            role_clause = "role = 'assistant'"

        clauses = ["session_id = ?", role_clause]
        parameters: list[Any] = [session_id]

        # Tool-call assistant messages can have content=null. Scan a bounded
        # multiple of the requested page so those rows do not consume the user's
        # useful message budget, while still keeping database reads bounded.
        scan_limit = min(limit * 8, 800)

        if around_sequence is not None:
            query = f"""
                SELECT sequence, role, message_json
                FROM chat_messages
                WHERE {' AND '.join(clauses)}
                ORDER BY ABS(sequence - ?) ASC, sequence ASC
                LIMIT ?
            """
            parameters.extend([around_sequence, scan_limit])
        else:
            if before_sequence is not None:
                clauses.append("sequence < ?")
                parameters.append(before_sequence)
            query = f"""
                SELECT sequence, role, message_json
                FROM chat_messages
                WHERE {' AND '.join(clauses)}
                ORDER BY sequence DESC
                LIMIT ?
            """
            parameters.append(scan_limit)

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        decoded: list[dict[str, Any]] = []
        for row in rows:
            message = json.loads(row["message_json"])
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            decoded.append(
                {
                    "sequence": int(row["sequence"]),
                    "role": "agent" if row["role"] == "assistant" else "user",
                    "content": content,
                }
            )

        if around_sequence is not None:
            decoded.sort(
                key=lambda item: (
                    abs(item["sequence"] - around_sequence),
                    item["sequence"],
                )
            )
            decoded = decoded[:limit]

        else:
            # SQL returned newest-first; keep only the bounded useful page before
            # presenting it chronologically to the model.
            decoded = decoded[:limit]

        decoded.sort(key=lambda item: item["sequence"])

        next_before_sequence = None
        if around_sequence is None and decoded:
            oldest = decoded[0]["sequence"]
            with self._connect() as connection:
                older = connection.execute(
                    f"""
                    SELECT 1
                    FROM chat_messages
                    WHERE session_id = ?
                      AND {role_clause}
                      AND sequence < ?
                    LIMIT 1
                    """,
                    (session_id, oldest),
                ).fetchone()
            if older is not None:
                next_before_sequence = oldest

        return {
            "session": session.to_dict(),
            "messages": decoded,
            "next_before_sequence": next_before_sequence,
        }

    def append_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        expected_count: int,
    ) -> StoredChatSession:
        if expected_count < 0:
            raise ValueError("expected_count cannot be negative")
        if not messages:
            return self.get_session(session_id)

        encoded_messages = [
            self._validate_message(message)
            for message in messages
        ]

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT message_count
                    FROM chat_sessions
                    WHERE id = ?
                    """,
                    (session_id,),
                ).fetchone()

                if row is None:
                    raise KeyError(
                        f"unknown chat session: {session_id}"
                    )

                current_count = int(row["message_count"])
                if current_count != expected_count:
                    raise RuntimeError(
                        "chat session changed concurrently: "
                        f"expected {expected_count} messages, "
                        f"found {current_count}"
                    )

                now = _utc_now()
                for offset, (message_role, encoded) in enumerate(
                    encoded_messages,
                    start=1,
                ):
                    connection.execute(
                        """
                        INSERT INTO chat_messages (
                            session_id,
                            sequence,
                            created_at,
                            role,
                            message_json
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            current_count + offset,
                            now,
                            message_role,
                            encoded,
                        ),
                    )

                new_count = current_count + len(encoded_messages)
                connection.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = ?, message_count = ?
                    WHERE id = ?
                    """,
                    (now, new_count, session_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        return self.get_session(session_id)

    def set_active_task(
        self,
        session_id: str,
        task_id: str | None,
    ) -> StoredChatSession:
        now = _utc_now()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE chat_sessions
                SET active_task_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (task_id, now, session_id),
            )

        if cursor.rowcount == 0:
            raise KeyError(f"unknown chat session: {session_id}")

        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM chat_sessions WHERE id = ?",
                (session_id,),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"unknown chat session: {session_id}")

    def delete_all_sessions(self) -> int:
        with self._connect() as connection:
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM chat_sessions"
                ).fetchone()[0]
            )
            connection.execute("DELETE FROM chat_sessions")
        return count

    def sync_chunks(
        self,
        session_id: str,
        *,
        target_tokens: int = 320,
        max_tokens: int = 420,
        overlap_tokens: int = 64,
    ) -> list[ChatChunk]:
        session = self.get_session(session_id)

        with self._connect() as connection:
            state = connection.execute(
                """
                SELECT chunked_message_count, chunk_version
                FROM chat_index_state
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if (
            state is not None
            and int(state["chunked_message_count"]) == session.message_count
            and int(state["chunk_version"]) == _CHUNK_VERSION
        ):
            return self._load_chunks(session_id)

        messages = self._load_searchable_messages(session_id)
        chunks = build_chat_chunks(
            session_id,
            messages,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = {
                    int(row["chunk_index"]): str(row["content_hash"])
                    for row in connection.execute(
                        """
                        SELECT chunk_index, content_hash
                        FROM chat_chunks
                        WHERE session_id = ?
                        """,
                        (session_id,),
                    ).fetchall()
                }

                for chunk in chunks:
                    content_hash = _text_hash(
                        "\0".join(
                            (
                                chunk.all_text,
                                chunk.user_text,
                                chunk.agent_text,
                            )
                        )
                    )
                    if existing.get(chunk.chunk_index) == content_hash:
                        continue
                    connection.execute(
                        """
                        INSERT INTO chat_chunks (
                            session_id, chunk_index, start_sequence,
                            end_sequence, all_text, user_text, agent_text,
                            content_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id, chunk_index) DO UPDATE SET
                            start_sequence = excluded.start_sequence,
                            end_sequence = excluded.end_sequence,
                            all_text = excluded.all_text,
                            user_text = excluded.user_text,
                            agent_text = excluded.agent_text,
                            content_hash = excluded.content_hash
                        """,
                        (
                            chunk.session_id,
                            chunk.chunk_index,
                            chunk.start_sequence,
                            chunk.end_sequence,
                            chunk.all_text,
                            chunk.user_text,
                            chunk.agent_text,
                            content_hash,
                        ),
                    )

                connection.execute(
                    """
                    DELETE FROM chat_chunks
                    WHERE session_id = ? AND chunk_index >= ?
                    """,
                    (session_id, len(chunks)),
                )
                connection.execute(
                    """
                    INSERT INTO chat_index_state (
                        session_id, chunked_message_count, chunk_version
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        chunked_message_count = excluded.chunked_message_count,
                        chunk_version = excluded.chunk_version
                    """,
                    (session_id, session.message_count, _CHUNK_VERSION),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        return self._load_chunks(session_id)

    def pending_embedding_rows(
        self,
        session_ids: list[str],
        view: EmbeddingView,
    ) -> list[dict[str, Any]]:
        if not session_ids:
            return []
        results: list[dict[str, Any]] = []
        for batch in _batches(session_ids, 500):
            placeholders = ",".join("?" for _ in batch)
            text_column = _view_column(view)
            with self._connect() as connection:
                rows = connection.execute(
                    f"""
                    SELECT
                        c.session_id,
                        c.chunk_index,
                        c.{text_column} AS text,
                        e.text_hash AS stored_hash
                    FROM chat_chunks AS c
                    LEFT JOIN chat_chunk_embeddings AS e
                      ON e.session_id = c.session_id
                     AND e.chunk_index = c.chunk_index
                     AND e.view = ?
                    WHERE c.session_id IN ({placeholders})
                      AND c.{text_column} != ''
                    ORDER BY c.session_id, c.chunk_index
                    """,
                    [view, *batch],
                ).fetchall()
            for row in rows:
                text = str(row["text"])
                current_hash = _text_hash(text)
                if row["stored_hash"] != current_hash:
                    results.append(
                        {
                            "session_id": str(row["session_id"]),
                            "chunk_index": int(row["chunk_index"]),
                            "text": text,
                            "text_hash": current_hash,
                        }
                    )
        return results

    def store_embedding(
        self,
        session_id: str,
        chunk_index: int,
        view: EmbeddingView,
        text_hash: str,
        vector: list[float],
    ) -> None:
        blob = _pack_vector(vector)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_chunk_embeddings (
                    session_id, chunk_index, view, text_hash,
                    dimensions, embedding
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, chunk_index, view) DO UPDATE SET
                    text_hash = excluded.text_hash,
                    dimensions = excluded.dimensions,
                    embedding = excluded.embedding
                """,
                (
                    session_id,
                    chunk_index,
                    view,
                    text_hash,
                    len(vector),
                    blob,
                ),
            )

    def embedding_candidates(
        self,
        session_ids: list[str],
        view: EmbeddingView,
    ) -> list[dict[str, Any]]:
        if not session_ids:
            return []
        results: list[dict[str, Any]] = []
        text_column = _view_column(view)
        for batch in _batches(session_ids, 500):
            placeholders = ",".join("?" for _ in batch)
            with self._connect() as connection:
                rows = connection.execute(
                    f"""
                    SELECT
                        c.session_id,
                        c.chunk_index,
                        c.start_sequence,
                        c.end_sequence,
                        c.{text_column} AS text,
                        s.title,
                        e.text_hash,
                        e.dimensions,
                        e.embedding
                    FROM chat_chunks AS c
                    JOIN chat_sessions AS s ON s.id = c.session_id
                    JOIN chat_chunk_embeddings AS e
                      ON e.session_id = c.session_id
                     AND e.chunk_index = c.chunk_index
                     AND e.view = ?
                    WHERE c.session_id IN ({placeholders})
                      AND c.{text_column} != ''
                    ORDER BY s.updated_at DESC, c.chunk_index
                    """,
                    [view, *batch],
                ).fetchall()
            for row in rows:
                text = str(row["text"])
                if row["text_hash"] != _text_hash(text):
                    continue
                results.append(
                    {
                        "session_id": str(row["session_id"]),
                        "session_title": str(row["title"]),
                        "chunk_index": int(row["chunk_index"]),
                        "start_sequence": int(row["start_sequence"]),
                        "end_sequence": int(row["end_sequence"]),
                        "text": text,
                        "vector": _unpack_vector(
                            row["embedding"],
                            int(row["dimensions"]),
                        ),
                    }
                )
        return results

    def _load_searchable_messages(
        self,
        session_id: str,
    ) -> list[_SearchableMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, role, message_json
                FROM chat_messages
                WHERE session_id = ?
                  AND role IN ('user', 'assistant')
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()

        messages: list[_SearchableMessage] = []
        for row in rows:
            message = json.loads(row["message_json"])
            content = message.get("content")
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue
            messages.append(
                _SearchableMessage(
                    sequence=int(row["sequence"]),
                    role=row["role"],
                    content=content,
                )
            )
        return messages

    def _load_chunks(self, session_id: str) -> list[ChatChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM chat_chunks
                WHERE session_id = ?
                ORDER BY chunk_index ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ChatChunk(
                session_id=str(row["session_id"]),
                chunk_index=int(row["chunk_index"]),
                start_sequence=int(row["start_sequence"]),
                end_sequence=int(row["end_sequence"]),
                all_text=str(row["all_text"]),
                user_text=str(row["user_text"]),
                agent_text=str(row["agent_text"]),
            )
            for row in rows
        ]

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> StoredChatSession:
        return StoredChatSession(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=int(row["message_count"]),
            active_task_id=row["active_task_id"],
        )


class ChatHistorySearch:
    """Chunked semantic retrieval over authorized persisted chat sessions."""

    def __init__(
        self,
        store: ChatSessionStore,
        embedding_provider: EmbeddingProvider,
        *,
        embedding_batch_size: int = 32,
        target_tokens: int = 320,
        max_tokens: int = 420,
        overlap_tokens: int = 64,
    ):
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        self.store = store
        self.embedding_provider = embedding_provider
        self.embedding_batch_size = embedding_batch_size
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def search(
        self,
        query: str,
        session_ids: list[str],
        *,
        role: ChatRoleFilter = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if role not in {None, "user", "agent"}:
            raise ValueError("role must be 'user', 'agent', or null")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        if not session_ids:
            return []

        view: EmbeddingView = "all" if role is None else role

        for session_id in session_ids:
            self.store.sync_chunks(
                session_id,
                target_tokens=self.target_tokens,
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens,
            )

        pending = self.store.pending_embedding_rows(session_ids, view)
        for batch in _batches(pending, self.embedding_batch_size):
            vectors = self.embedding_provider.embed_documents(
                [item["text"] for item in batch]
            )
            if len(vectors) != len(batch):
                raise RuntimeError(
                    "embedding provider returned an unexpected number of vectors"
                )
            for item, vector in zip(batch, vectors, strict=True):
                self.store.store_embedding(
                    item["session_id"],
                    item["chunk_index"],
                    view,
                    item["text_hash"],
                    vector,
                )

        query_vector = self.embedding_provider.embed_query(query)
        scored: list[dict[str, Any]] = []
        for candidate in self.store.embedding_candidates(session_ids, view):
            score = _cosine_similarity(
                query_vector,
                candidate.pop("vector"),
            )
            preview_text = candidate.pop("text")
            scored.append(
                {
                    **candidate,
                    "score": round(score, 6),
                    "preview": _preview(preview_text),
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]


def build_chat_chunks(
    session_id: str,
    messages: list[_SearchableMessage],
    *,
    target_tokens: int = 320,
    max_tokens: int = 420,
    overlap_tokens: int = 64,
) -> list[ChatChunk]:
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    if max_tokens < target_tokens:
        raise ValueError("max_tokens must be >= target_tokens")
    if not 0 <= overlap_tokens < target_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < target_tokens")
    if not messages:
        return []

    chunks: list[ChatChunk] = []
    start = 0

    while start < len(messages):
        end = start
        total = 0

        while end < len(messages):
            tokens = messages[end].estimated_tokens
            if end > start and total + tokens > max_tokens:
                break
            total += tokens
            end += 1
            if total >= target_tokens:
                break

        if end == start:
            end = start + 1

        selected = messages[start:end]
        chunks.append(
            _make_chunk(
                session_id,
                len(chunks),
                selected,
            )
        )

        if end >= len(messages):
            break

        back = end
        overlap = 0
        while back > start:
            tokens = messages[back - 1].estimated_tokens
            if overlap + tokens > overlap_tokens:
                break
            overlap += tokens
            back -= 1

        start = max(start + 1, back)

    return chunks


def _make_chunk(
    session_id: str,
    chunk_index: int,
    messages: list[_SearchableMessage],
) -> ChatChunk:
    all_parts: list[str] = []
    user_parts: list[str] = []
    agent_parts: list[str] = []

    for message in messages:
        if message.role == "user":
            rendered = f"User: {message.content}"
            user_parts.append(rendered)
        else:
            rendered = f"Agent: {message.content}"
            agent_parts.append(rendered)
        all_parts.append(rendered)

    return ChatChunk(
        session_id=session_id,
        chunk_index=chunk_index,
        start_sequence=messages[0].sequence,
        end_sequence=messages[-1].sequence,
        all_text="\n\n".join(all_parts),
        user_text="\n\n".join(user_parts),
        agent_text="\n\n".join(agent_parts),
    )


def _view_column(view: EmbeddingView) -> str:
    return {
        "all": "all_text",
        "user": "user_text",
        "agent": "agent_text",
    }[view]


def _preview(text: str, max_length: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def _batches(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]
