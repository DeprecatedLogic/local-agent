from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolPolicy:
    """Capability filter applied by a runtime before exposing/executing tools."""

    allowed: frozenset[str] | None = None
    denied: frozenset[str] = field(default_factory=frozenset)

    def allows(self, tool_name: str) -> bool:
        if tool_name in self.denied:
            return False
        if self.allowed is None:
            return True
        return tool_name in self.allowed
