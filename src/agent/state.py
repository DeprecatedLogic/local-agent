from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import json
import sqlite3
import uuid

TaskStatus = Literal["in_progress", "completed", "blocked", "cancelled"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json_list(value: str) -> list[str]:
    data = json.loads(value)
    if not isinstance(data, list):
        raise ValueError("stored task-state value is not a list")
    return [str(item) for item in data]


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    goal: str
    status: TaskStatus
    created_at: str
    updated_at: str
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "goal": self.goal,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }


@dataclass(init=False)
class TaskState:
    task_id: str | None
    title: str
    goal: str
    status: str
    current: str
    completed: list[str]
    blocked: list[str]
    files: list[str]
    revision: int
    saved_at: str | None

    def __init__(
        self,
        task_id: str | None = None,
        title: str = "",
        goal: str = "",
        status: str = "idle",
        current: str = "",
        completed: list[str] | None = None,
        blocked: list[str] | None = None,
        files: list[str] | None = None,
        revision: int = 0,
        saved_at: str | None = None,
        *,
        task: str | None = None,
    ):
        # `task` is a compatibility alias for the old API. New code uses `goal`.
        if task is not None and not goal:
            goal = task

        self.task_id = task_id
        self.title = title
        self.goal = goal
        self.status = status
        self.current = current
        self.completed = list(completed or [])
        self.blocked = list(blocked or [])
        self.files = list(files or [])
        self.revision = revision
        self.saved_at = saved_at

    @property
    def task(self) -> str:
        """Deprecated compatibility alias for goal."""
        return self.goal

    @task.setter
    def task(self, value: str) -> None:
        self.goal = value

    @property
    def active(self) -> bool:
        return self.task_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "task_id": self.task_id,
            "title": self.title,
            "goal": self.goal,
            "status": self.status,
            "current": self.current,
            "completed": list(self.completed),
            "blocked": list(self.blocked),
            "files": list(self.files),
            "revision": self.revision,
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        return cls(
            task_id=data.get("task_id"),
            title=data.get("title", ""),
            goal=data.get("goal", data.get("task", "")),
            status=data.get("status", "idle"),
            current=data.get("current", ""),
            completed=data.get("completed", []),
            blocked=data.get("blocked", []),
            files=data.get("files", []),
            revision=data.get("revision", 0),
            saved_at=data.get("saved_at"),
        )


class TaskStore:
    """SQLite-backed persistent task metadata and versioned task state."""

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
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('in_progress', 'completed', 'blocked', 'cancelled')
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS task_state_revisions (
                    task_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    saved_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('in_progress', 'completed', 'blocked', 'cancelled')
                    ),
                    current TEXT NOT NULL,
                    completed_json TEXT NOT NULL,
                    blocked_json TEXT NOT NULL,
                    files_json TEXT NOT NULL,
                    PRIMARY KEY (task_id, revision),
                    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status_updated
                    ON tasks(status, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_tasks_created
                    ON tasks(created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_task_revisions_lookup
                    ON task_state_revisions(task_id, revision DESC);
                """
            )

    @staticmethod
    def _validate_text(value: str, *, name: str, max_length: int) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{name} cannot be empty")
        if len(value) > max_length:
            raise ValueError(f"{name} cannot exceed {max_length} characters")
        return value

    def create_task(self, title: str, goal: str) -> TaskState:
        title = self._validate_text(title, name="title", max_length=200)
        goal = self._validate_text(goal, name="goal", max_length=20_000)

        task_id = f"task_{uuid.uuid4().hex}"
        now = _utc_now()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO tasks (
                        id, title, goal, status, created_at, updated_at, completed_at
                    ) VALUES (?, ?, ?, 'in_progress', ?, ?, NULL)
                    """,
                    (task_id, title, goal, now, now),
                )
                connection.execute(
                    """
                    INSERT INTO task_state_revisions (
                        task_id, revision, saved_at, status, current,
                        completed_json, blocked_json, files_json
                    ) VALUES (?, 1, ?, 'in_progress', '', '[]', '[]', '[]')
                    """,
                    (task_id, now),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        return self.get_state(task_id)

    def get_task(self, task_id: str) -> Task:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"unknown task: {task_id}")

        return Task(
            id=row["id"],
            title=row["title"],
            goal=row["goal"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    def get_state(
        self,
        task_id: str,
        *,
        revision: int | None = None,
    ) -> TaskState:
        if revision is not None and revision < 1:
            raise ValueError("revision must be at least 1")

        operator = "= ?" if revision is not None else "IS NOT NULL"
        parameters: tuple[Any, ...]
        if revision is None:
            parameters = (task_id,)
        else:
            parameters = (task_id, revision)

        query = f"""
            SELECT
                t.id AS task_id,
                t.title,
                t.goal,
                r.status,
                r.revision,
                r.saved_at,
                r.current,
                r.completed_json,
                r.blocked_json,
                r.files_json
            FROM tasks AS t
            JOIN task_state_revisions AS r ON r.task_id = t.id
            WHERE t.id = ? AND r.revision {operator}
            ORDER BY r.revision DESC
            LIMIT 1
        """

        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()

        if row is None:
            if revision is None:
                raise KeyError(f"unknown task: {task_id}")
            raise KeyError(f"task {task_id} has no revision {revision}")

        return self._state_from_row(row)

    def save_state(self, state: TaskState) -> TaskState:
        if state.task_id is None:
            raise ValueError("cannot save task state without a task_id")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = connection.execute(
                    "SELECT id FROM tasks WHERE id = ?",
                    (state.task_id,),
                ).fetchone()
                if task is None:
                    raise KeyError(f"unknown task: {state.task_id}")

                latest = connection.execute(
                    """
                    SELECT
                        r.status, r.revision, r.saved_at, r.current,
                        r.completed_json, r.blocked_json, r.files_json
                    FROM tasks AS t
                    JOIN task_state_revisions AS r ON r.task_id = t.id
                    WHERE t.id = ?
                    ORDER BY r.revision DESC
                    LIMIT 1
                    """,
                    (state.task_id,),
                ).fetchone()

                if latest is not None and (
                    latest["status"] == state.status
                    and latest["current"] == state.current
                    and _json_list(latest["completed_json"]) == state.completed
                    and _json_list(latest["blocked_json"]) == state.blocked
                    and _json_list(latest["files_json"]) == state.files
                ):
                    connection.rollback()
                    return self.get_state(state.task_id)

                next_revision = int(latest["revision"]) + 1
                now = _utc_now()
                completed_at = (
                    now
                    if state.status in {"completed", "cancelled"}
                    else None
                )

                connection.execute(
                    """
                    UPDATE tasks
                    SET status = ?, updated_at = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (state.status, now, completed_at, state.task_id),
                )
                connection.execute(
                    """
                    INSERT INTO task_state_revisions (
                        task_id, revision, saved_at, status, current,
                        completed_json, blocked_json, files_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state.task_id,
                        next_revision,
                        now,
                        state.status,
                        state.current,
                        json.dumps(state.completed, ensure_ascii=False),
                        json.dumps(state.blocked, ensure_ascii=False),
                        json.dumps(state.files, ensure_ascii=False),
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        return self.get_state(state.task_id)

    def set_status(self, task_id: str, status: TaskStatus) -> TaskState:
        state = self.get_state(task_id)
        state.status = status
        return self.save_state(state)

    def list_tasks(
        self,
        *,
        limit: int = 10,
        status: TaskStatus | None = None,
        before: str | None = None,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")

        clauses: list[str] = []
        parameters: list[Any] = []

        if status is not None:
            clauses.append("t.status = ?")
            parameters.append(status)
        if before is not None:
            clauses.append("t.updated_at < ?")
            parameters.append(before)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)

        query = f"""
            SELECT
                t.*,
                r.revision,
                r.current,
                r.saved_at
            FROM tasks AS t
            JOIN task_state_revisions AS r
              ON r.task_id = t.id
             AND r.revision = (
                 SELECT MAX(r2.revision)
                 FROM task_state_revisions AS r2
                 WHERE r2.task_id = t.id
             )
            {where}
            ORDER BY t.updated_at DESC
            LIMIT ?
        """

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [self._summary_from_row(row) for row in rows]

    def search_tasks(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        query = self._validate_text(query, name="query", max_length=500)
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")

        pattern = f"%{query}%"
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    t.*,
                    r.revision,
                    r.current,
                    r.saved_at
                FROM tasks AS t
                JOIN task_state_revisions AS r
                  ON r.task_id = t.id
                 AND r.revision = (
                     SELECT MAX(r2.revision)
                     FROM task_state_revisions AS r2
                     WHERE r2.task_id = t.id
                 )
                WHERE t.title LIKE ? COLLATE NOCASE
                   OR t.goal LIKE ? COLLATE NOCASE
                   OR r.current LIKE ? COLLATE NOCASE
                ORDER BY t.updated_at DESC
                LIMIT ?
                """,
                (pattern, pattern, pattern, limit),
            ).fetchall()

        return [self._summary_from_row(row) for row in rows]

    def get_history(
        self,
        task_id: str,
        *,
        limit: int = 5,
        before_revision: int | None = None,
    ) -> list[TaskState]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if before_revision is not None and before_revision < 2:
            return []

        parameters: list[Any] = [task_id]
        revision_clause = ""
        if before_revision is not None:
            revision_clause = "AND r.revision < ?"
            parameters.append(before_revision)
        parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    t.id AS task_id,
                    t.title,
                    t.goal,
                    r.status,
                    r.revision,
                    r.saved_at,
                    r.current,
                    r.completed_json,
                    r.blocked_json,
                    r.files_json
                FROM tasks AS t
                JOIN task_state_revisions AS r ON r.task_id = t.id
                WHERE t.id = ? {revision_clause}
                ORDER BY r.revision DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        if not rows and not self._task_exists(task_id):
            raise KeyError(f"unknown task: {task_id}")

        return [self._state_from_row(row) for row in rows]

    def _task_exists(self, task_id: str) -> bool:
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone() is not None

    @staticmethod
    def _state_from_row(row: sqlite3.Row) -> TaskState:
        return TaskState(
            task_id=row["task_id"],
            title=row["title"],
            goal=row["goal"],
            status=row["status"],
            current=row["current"],
            completed=_json_list(row["completed_json"]),
            blocked=_json_list(row["blocked_json"]),
            files=_json_list(row["files_json"]),
            revision=row["revision"],
            saved_at=row["saved_at"],
        )

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["id"],
            "title": row["title"],
            "goal_preview": (
                row["goal"]
                if len(row["goal"]) <= 240
                else row["goal"][:237].rstrip() + "..."
            ),
            "status": row["status"],
            "current": row["current"],
            "revision": row["revision"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }


class ActiveTaskStateStore:
    """Compatibility facade over TaskStore for code that expects server.state."""

    def __init__(self, task_store: TaskStore, active_task_id_getter):
        self.task_store = task_store
        self.path = task_store.path
        self._active_task_id_getter = active_task_id_getter

    def load(self) -> TaskState:
        task_id = self._active_task_id_getter()
        if task_id is None:
            return TaskState()
        return self.task_store.get_state(task_id)

    def save(self, state: TaskState) -> TaskState:
        task_id = self._active_task_id_getter()
        if task_id is None:
            if not state.active:
                return state
            task_id = state.task_id

        if state.task_id is None:
            state.task_id = task_id
        if state.task_id != task_id:
            raise ValueError("task state does not belong to the active task")
        return self.task_store.save_state(state)
