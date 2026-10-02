from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


AgentEventKind = Literal[
    "reasoning",
    "content",
    "tool_call",
    "tool_result",
]


@dataclass(frozen=True)
class AgentEvent:
    kind: AgentEventKind
    text: str = ""
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    is_error: bool = False


StreamHandler = Callable[[AgentEvent], None]
