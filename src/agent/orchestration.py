from __future__ import annotations

import asyncio
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from agent.capabilities import ToolPolicy


DelegationStatus = Literal[
    "completed",
    "blocked",
    "failed",
    "timeout",
]


DEFAULT_MAX_CONTEXT_TOKENS = 32768
DEFAULT_MAX_REASONING_TOKENS = 4096


DEFAULT_AGENTS_TOML = '''# Local Agent specialist definitions.
#
# This file is created automatically when missing. Existing values are never
# overwritten during updates. Add/remove/rename agents here; no Python changes
# are required. Tool groups are reusable shortcuts and may also be customized.

[defaults]
# Logical per-specialist token budgets. Individual agents may override either value.
max_context_tokens = 32768
max_reasoning_tokens = 4096

[tool_groups]
inspect = [
  "project_info",
  "get_agent_context",
  "read_file",
  "list_dir_contents",
  "list_tree",
  "search_in_file",
  "search_files",
  "git_status",
  "git_diff",
  "git_log",
  "git_show",
]
modify = [
  "write_file",
  "edit_file_lines",
  "create_file",
]
manage_files = [
  "create_folder",
  "move_item",
  "delete_item",
]
execute = ["run_command"]
task_history = [
  "get_task_state",
  "list_tasks",
  "search_tasks",
  "get_task_history",
]
chat_history = [
  "list_chat_sessions",
  "read_chat_session",
  "search_chat_history",
]

[agents.coder]
enabled = true
name = "Coder"
description = "Implements focused code changes and validates them with project tools."
instructions = """
Inspect the relevant implementation before changing it. Make only changes required by
this delegated task. Validate the result with the strongest practical checks and
report exactly what changed, what was verified, and any remaining uncertainty.
"""
backend = "primary"
tool_groups = ["inspect", "modify", "manage_files", "execute"]
tools = []

[agents.tester]
enabled = true
name = "Tester"
description = "Reproduces failures, runs targeted tests, and reports evidence."
instructions = """
Focus on reproduction and verification. Run the narrowest useful checks first, expand
only when justified, and distinguish observed failures from hypotheses. Do not edit
project files intentionally.
"""
backend = "primary"
tool_groups = ["inspect", "execute"]
tools = []

[agents.reviewer]
enabled = true
name = "Reviewer"
description = "Performs an independent review for correctness, regressions, edge cases, and maintainability."
instructions = """
Review independently. Prioritize concrete correctness issues, regressions, unsafe
assumptions, and missing validation. Cite specific files or evidence. Do not modify
files or execute shell commands.
"""
backend = "primary"
tool_groups = ["inspect"]
tools = []

[agents.docs]
enabled = true
name = "Documentation"
description = "Updates documentation and explanatory text after inspecting implementation."
instructions = """
Keep documentation aligned with observed behavior. Modify only documentation or
explanatory text required by the delegated task; do not change implementation logic.
"""
backend = "primary"
tool_groups = ["inspect", "modify"]
tools = []
'''


# Worker-level invariants are runtime-owned rather than configurable. Users may
# configure ordinary capabilities freely, but a depth-one worker must not recursively
# delegate, mutate the parent task lifecycle, or delete persistent chat history.
_ALWAYS_DENIED_WORKER_TOOLS = frozenset({
    "delegate_task",
    "list_agents",
    "start_task",
    "resume_task",
    "set_task_state",
    "finish_task",
    "delete_chat_session",
})


class WorkerModel(Protocol):
    async def generate(self, messages, tools, on_stream=None):
        ...


@dataclass(frozen=True)
class DelegationLimits:
    max_delegations_per_turn: int = 4
    max_concurrent_workers: int = 1
    worker_timeout_seconds: float = 180.0
    worker_max_iterations: int = 8
    worker_max_tool_calls: int = 24
    max_result_chars: int = 6000

    def __post_init__(self) -> None:
        if self.max_delegations_per_turn < 1:
            raise ValueError("max_delegations_per_turn must be at least 1")
        if self.max_concurrent_workers < 1:
            raise ValueError("max_concurrent_workers must be at least 1")
        if self.worker_timeout_seconds <= 0:
            raise ValueError("worker_timeout_seconds must be positive")
        if self.worker_max_iterations < 1:
            raise ValueError("worker_max_iterations must be at least 1")
        if self.worker_max_tool_calls < 1:
            raise ValueError("worker_max_tool_calls must be at least 1")
        if self.max_result_chars < 256:
            raise ValueError("max_result_chars must be at least 256")


@dataclass(frozen=True)
class AgentSpec:
    id: str
    name: str
    description: str
    instructions: str
    allowed_tools: frozenset[str]
    backend_id: str = "primary"
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_reasoning_tokens: int = DEFAULT_MAX_REASONING_TOKENS
    timeout_seconds: float | None = None
    max_iterations: int | None = None
    max_tool_calls: int | None = None
    max_result_chars: int | None = None

    def tool_policy(self) -> ToolPolicy:
        return ToolPolicy(
            allowed=self.allowed_tools,
            denied=_ALWAYS_DENIED_WORKER_TOOLS,
        )

    def effective_timeout(self, defaults: DelegationLimits) -> float:
        return self.timeout_seconds or defaults.worker_timeout_seconds

    def effective_iterations(self, defaults: DelegationLimits) -> int:
        return self.max_iterations or defaults.worker_max_iterations

    def effective_tool_calls(self, defaults: DelegationLimits) -> int:
        return self.max_tool_calls or defaults.worker_max_tool_calls

    def effective_result_chars(self, defaults: DelegationLimits) -> int:
        return self.max_result_chars or defaults.max_result_chars


@dataclass(frozen=True)
class AgentSession:
    id: str
    agent_id: str
    backend_id: str
    max_context_tokens: int
    max_reasoning_tokens: int
    started_at: str

    @classmethod
    def create(cls, spec: AgentSpec) -> "AgentSession":
        return cls(
            id=f"agent_session_{uuid4().hex}",
            agent_id=spec.id,
            backend_id=spec.backend_id,
            max_context_tokens=spec.max_context_tokens,
            max_reasoning_tokens=spec.max_reasoning_tokens,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )


@dataclass(frozen=True)
class DelegationRequest:
    id: str
    agent_id: str
    task: str
    context: str | None = None

    @classmethod
    def create(
        cls,
        agent_id: str,
        task: str,
        context: str | None = None,
    ) -> "DelegationRequest":
        task = task.strip()
        if not task:
            raise ValueError("delegated task cannot be empty")

        normalized_context = None
        if context is not None:
            normalized_context = context.strip() or None

        return cls(
            id=f"delegation_{uuid4().hex}",
            agent_id=agent_id.strip().lower(),
            task=task,
            context=normalized_context,
        )


@dataclass(frozen=True)
class DelegationResult:
    delegation_id: str
    session_id: str
    agent_id: str
    backend_id: str
    status: DelegationStatus
    summary: str
    modified_files: tuple[str, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "delegation_id": self.delegation_id,
            "session_id": self.session_id,
            "agent": self.agent_id,
            "backend_id": self.backend_id,
            "status": self.status,
            "summary": self.summary,
            "modified_files": list(self.modified_files),
            "usage": dict(self.usage),
            "tool_calls": self.tool_calls,
        }


class AgentRegistry:
    def __init__(self, specs: list[AgentSpec], *, source: Path | None = None):
        self._specs: dict[str, AgentSpec] = {}
        self.source = source
        for spec in specs:
            key = self._agent_id(spec.id)
            if key in self._specs:
                raise ValueError(f"duplicate agent id: {key}")
            self._specs[key] = spec

    @staticmethod
    def _agent_id(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("agent id must be a string")
        key = value.strip().lower()
        if not key:
            raise ValueError("agent id cannot be empty")
        if not key.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"invalid agent id: {value}")
        return key

    @staticmethod
    def _text(value: object, *, field_name: str, default: str | None = None) -> str:
        if value is None and default is not None:
            return default
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        value = value.strip()
        if not value and default is None:
            raise ValueError(f"{field_name} cannot be empty")
        return value or (default or "")

    @staticmethod
    def _string_list(value: object, *, field_name: str) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field_name} must be an array of strings")
        result: list[str] = []
        for item in value:
            normalized = item.strip()
            if not normalized:
                raise ValueError(f"{field_name} cannot contain empty values")
            result.append(normalized)
        return result

    @staticmethod
    def _positive_int(value: object, *, field_name: str) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field_name} must be an integer >= 1")
        return value

    @staticmethod
    def _positive_float(value: object, *, field_name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{field_name} must be a positive number")
        return float(value)

    @classmethod
    def from_config_dir(cls, config_dir: str | Path) -> "AgentRegistry":
        config_dir = Path(config_dir).expanduser().resolve()
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / "agents.toml"

        try:
            with path.open("x", encoding="utf-8") as file:
                file.write(DEFAULT_AGENTS_TOML)
        except FileExistsError:
            pass

        with path.open("rb") as file:
            data = tomllib.load(file)

        raw_defaults = data.get("defaults", {})
        if not isinstance(raw_defaults, dict):
            raise ValueError("agents.toml [defaults] must be a table")

        default_max_context_tokens = (
            cls._positive_int(
                raw_defaults.get("max_context_tokens"),
                field_name="defaults.max_context_tokens",
            )
            or DEFAULT_MAX_CONTEXT_TOKENS
        )
        default_max_reasoning_tokens = (
            cls._positive_int(
                raw_defaults.get("max_reasoning_tokens"),
                field_name="defaults.max_reasoning_tokens",
            )
            or DEFAULT_MAX_REASONING_TOKENS
        )
        if default_max_reasoning_tokens > default_max_context_tokens:
            raise ValueError(
                "defaults.max_reasoning_tokens cannot exceed "
                "defaults.max_context_tokens"
            )

        raw_groups = data.get("tool_groups", {})
        if not isinstance(raw_groups, dict):
            raise ValueError("agents.toml [tool_groups] must be a table")

        groups: dict[str, frozenset[str]] = {}
        for group_name, raw_tools in raw_groups.items():
            key = cls._agent_id(group_name)
            groups[key] = frozenset(
                cls._string_list(
                    raw_tools,
                    field_name=f"tool_groups.{group_name}",
                )
            )

        raw_agents = data.get("agents", {})
        if not isinstance(raw_agents, dict):
            raise ValueError("agents.toml [agents] must be a table")

        specs: list[AgentSpec] = []
        for raw_id, raw_spec in raw_agents.items():
            agent_id = cls._agent_id(raw_id)
            if not isinstance(raw_spec, dict):
                raise ValueError(f"agents.{agent_id} must be a table")

            enabled = raw_spec.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError(f"agents.{agent_id}.enabled must be boolean")
            if not enabled:
                continue

            group_names = cls._string_list(
                raw_spec.get("tool_groups"),
                field_name=f"agents.{agent_id}.tool_groups",
            )
            allowed: set[str] = set(
                cls._string_list(
                    raw_spec.get("tools"),
                    field_name=f"agents.{agent_id}.tools",
                )
            )
            for group_name in group_names:
                key = cls._agent_id(group_name)
                try:
                    allowed.update(groups[key])
                except KeyError as exc:
                    raise ValueError(
                        f"agents.{agent_id} references unknown tool group: {group_name}"
                    ) from exc

            max_context_tokens = (
                cls._positive_int(
                    raw_spec.get("max_context_tokens"),
                    field_name=f"agents.{agent_id}.max_context_tokens",
                )
                or default_max_context_tokens
            )

            legacy_reasoning_budget = raw_spec.get("reasoning_budget")
            if (
                raw_spec.get("max_reasoning_tokens") is not None
                and legacy_reasoning_budget is not None
            ):
                raise ValueError(
                    f"agents.{agent_id} cannot define both max_reasoning_tokens "
                    "and legacy reasoning_budget"
                )
            max_reasoning_tokens = (
                cls._positive_int(
                    raw_spec.get("max_reasoning_tokens", legacy_reasoning_budget),
                    field_name=f"agents.{agent_id}.max_reasoning_tokens",
                )
                or default_max_reasoning_tokens
            )
            if max_reasoning_tokens > max_context_tokens:
                raise ValueError(
                    f"agents.{agent_id}.max_reasoning_tokens cannot exceed "
                    f"agents.{agent_id}.max_context_tokens"
                )

            max_result_chars = cls._positive_int(
                raw_spec.get("max_result_chars"),
                field_name=f"agents.{agent_id}.max_result_chars",
            )
            if max_result_chars is not None and max_result_chars < 256:
                raise ValueError(
                    f"agents.{agent_id}.max_result_chars must be at least 256"
                )

            specs.append(
                AgentSpec(
                    id=agent_id,
                    name=cls._text(
                        raw_spec.get("name"),
                        field_name=f"agents.{agent_id}.name",
                        default=agent_id.replace("_", " ").replace("-", " ").title(),
                    ),
                    description=cls._text(
                        raw_spec.get("description"),
                        field_name=f"agents.{agent_id}.description",
                    ),
                    instructions=cls._text(
                        raw_spec.get("instructions"),
                        field_name=f"agents.{agent_id}.instructions",
                    ),
                    allowed_tools=frozenset(allowed),
                    backend_id=cls._text(
                        raw_spec.get("backend"),
                        field_name=f"agents.{agent_id}.backend",
                        default="primary",
                    ),
                    timeout_seconds=cls._positive_float(
                        raw_spec.get("timeout_seconds"),
                        field_name=f"agents.{agent_id}.timeout_seconds",
                    ),
                    max_iterations=cls._positive_int(
                        raw_spec.get("max_iterations"),
                        field_name=f"agents.{agent_id}.max_iterations",
                    ),
                    max_tool_calls=cls._positive_int(
                        raw_spec.get("max_tool_calls"),
                        field_name=f"agents.{agent_id}.max_tool_calls",
                    ),
                    max_context_tokens=max_context_tokens,
                    max_reasoning_tokens=max_reasoning_tokens,
                    max_result_chars=max_result_chars,
                )
            )

        return cls(specs, source=path)

    def get(self, agent_id: str) -> AgentSpec:
        key = self._agent_id(agent_id)
        try:
            return self._specs[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._specs)) or "none"
            raise KeyError(
                f"unknown specialist '{agent_id}'; available: {available}"
            ) from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def summaries(self) -> list[dict[str, Any]]:
        return [
            {
                "id": spec.id,
                "name": spec.name,
                "description": spec.description,
                "backend_id": spec.backend_id,
                "max_context_tokens": spec.max_context_tokens,
                "max_reasoning_tokens": spec.max_reasoning_tokens,
                "tools": sorted(spec.allowed_tools),
            }
            for spec in sorted(self._specs.values(), key=lambda item: item.id)
        ]


def build_orchestrator_system_suffix(registry: AgentRegistry) -> str:
    agents = registry.summaries()
    if agents:
        catalog = "\n".join(
            f"- {item['id']}: {item['description']}"
            for item in agents
        )
    else:
        catalog = "- No specialist agents are currently enabled."

    return f"""
You are also the coordinator for bounded depth-one specialist agents. Use delegate_task
when a self-contained subtask would materially benefit from a configured specialist.
Do not delegate trivial work merely to use the feature. Workers cannot delegate
further. Remain responsible for integrating their evidence, resolving conflicts,
validating the final result, and answering the user.

Configured specialists:
{catalog}

Use list_agents when you need the current machine-readable specialist catalog.
""".strip()


class DelegationManager:
    """Runs depth-one specialist agents under hard per-turn capability limits."""

    def __init__(
        self,
        server,
        backends: dict[str, WorkerModel],
        *,
        registry: AgentRegistry,
        limits: DelegationLimits | None = None,
    ):
        self.server = server
        self.backends = dict(backends)
        self.registry = registry
        self.limits = limits or DelegationLimits()
        self._delegations_this_turn = 0
        self._counter_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(
            self.limits.max_concurrent_workers
        )

        missing = sorted({
            item["backend_id"]
            for item in self.registry.summaries()
            if item["backend_id"] not in self.backends
        })
        if missing:
            raise ValueError(
                "missing model backend(s): " + ", ".join(missing)
            )

    def reset_turn(self) -> None:
        self._delegations_this_turn = 0

    def available_agents(self) -> list[dict[str, Any]]:
        return self.registry.summaries()

    async def delegate(
        self,
        agent: str,
        task: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        spec = self.registry.get(agent)
        await self._validate_tools(spec)
        request = DelegationRequest.create(spec.id, task, context)
        session = AgentSession.create(spec)

        async with self._counter_lock:
            if self._delegations_this_turn >= self.limits.max_delegations_per_turn:
                raise RuntimeError(
                    "Delegation limit reached for this user turn "
                    f"({self.limits.max_delegations_per_turn})."
                )
            self._delegations_this_turn += 1

        async with self._semaphore:
            result = await self._run_worker(spec, request, session)

        return result.to_dict()

    async def _validate_tools(self, spec: AgentSpec) -> None:
        registered = {tool.name for tool in await self.server.mcp.list_tools()}
        unknown = sorted(spec.allowed_tools - registered)
        if unknown:
            source = f" in {self.registry.source}" if self.registry.source else ""
            raise ValueError(
                f"specialist '{spec.id}' references unknown tool(s){source}: "
                + ", ".join(unknown)
            )

    async def _run_worker(
        self,
        spec: AgentSpec,
        request: DelegationRequest,
        session: AgentSession,
    ) -> DelegationResult:
        from agent.runtime import AgentRuntime

        model = self._worker_model(
            self.backends[spec.backend_id],
            session.max_reasoning_tokens,
        )
        runtime = AgentRuntime(
            server=self.server,
            model=model,
            max_iterations=spec.effective_iterations(self.limits),
            tool_policy=spec.tool_policy(),
            system_prompt_suffix=self._worker_system_prompt(
                spec,
                request,
                session,
            ),
            max_tool_calls_total=spec.effective_tool_calls(self.limits),
            affect_task_lifecycle=False,
        )

        try:
            async with asyncio.timeout(spec.effective_timeout(self.limits)):
                summary = await runtime.run_subtask(request.task)
        except TimeoutError:
            return DelegationResult(
                delegation_id=request.id,
                session_id=session.id,
                agent_id=spec.id,
                backend_id=spec.backend_id,
                status="timeout",
                summary=(
                    "The specialist exceeded its execution timeout before "
                    "returning a result."
                ),
                modified_files=tuple(sorted(runtime.modified_files)),
                usage=dict(runtime.usage_totals),
                tool_calls=runtime.total_tool_calls,
            )
        except Exception as exc:
            return DelegationResult(
                delegation_id=request.id,
                session_id=session.id,
                agent_id=spec.id,
                backend_id=spec.backend_id,
                status="failed",
                summary=f"Specialist failed: {exc}",
                modified_files=tuple(sorted(runtime.modified_files)),
                usage=dict(runtime.usage_totals),
                tool_calls=runtime.total_tool_calls,
            )

        status: DelegationStatus = (
            "blocked"
            if runtime.last_stop_reason in {
                "iteration_limit",
                "tool_budget",
            }
            else "completed"
        )

        return DelegationResult(
            delegation_id=request.id,
            session_id=session.id,
            agent_id=spec.id,
            backend_id=spec.backend_id,
            status=status,
            summary=self._bounded_summary(
                summary,
                spec.effective_result_chars(self.limits),
            ),
            modified_files=tuple(sorted(runtime.modified_files)),
            usage=dict(runtime.usage_totals),
            tool_calls=runtime.total_tool_calls,
        )

    @staticmethod
    def _worker_model(model: WorkerModel, reasoning_budget: int) -> WorkerModel:
        fork = getattr(model, "with_reasoning_budget", None)
        if callable(fork):
            return fork(reasoning_budget)
        return model

    @staticmethod
    def _bounded_summary(text: str, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        keep = max_chars - 40
        return text[:keep].rstrip() + "\n...[delegation result truncated]"

    @staticmethod
    def _worker_system_prompt(
        spec: AgentSpec,
        request: DelegationRequest,
        session: AgentSession,
    ) -> str:
        allowed = ", ".join(sorted(spec.allowed_tools)) or "none"
        context = request.context or "No additional context was supplied."
        return f"""
You are a depth-one specialist working for the primary local agent.

Specialist: {spec.name} ({spec.id})
Logical session: {session.id}
Backend: {session.backend_id}
Context token budget: {session.max_context_tokens}
Reasoning token budget: {session.max_reasoning_tokens}

Role:
{spec.description}

Operating instructions:
{spec.instructions}

Delegated context:
{context}

Configured tools:
{allowed}

Complete only the delegated subtask. You cannot delegate to another agent and you do
not own the parent persistent task lifecycle. Return a concise evidence-based result to
the primary agent, including relevant files, commands/checks performed, failures, and
remaining uncertainty.
""".strip()
