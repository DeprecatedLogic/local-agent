from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class TaskState:
    task: str = ""
    status: str = "idle"
    current: str = ""
    completed: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "status": self.status,
            "current": self.current,
            "completed": self.completed,
            "blocked": self.blocked,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskState":
        return cls(
            task=data.get("task", ""),
            status=data.get("status", "idle"),
            current=data.get("current", ""),
            completed=data.get("completed", []),
            blocked=data.get("blocked", []),
            files=data.get("files", []),
        )


class StateStore:
    """Persistent structured state for the current agent task."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> TaskState:
        if not self.path.exists():
            return TaskState()

        with self.path.open("r", encoding="utf-8") as f:
            return TaskState.from_dict(json.load(f))

    def save(self, state: TaskState) -> None:
        temporary = self.path.with_suffix(".tmp")

        with temporary.open("w", encoding="utf-8") as f:
            json.dump(
                state.to_dict(),
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")

        temporary.replace(self.path)
