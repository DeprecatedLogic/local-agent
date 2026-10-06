from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from fastmcp.exceptions import ToolError

from agent.capabilities import ToolPolicy
from agent.context import build_system_prompt
from agent.events import AgentEvent, StreamHandler
from agent.mcp_.agent_server import AgentServer


class ModelClient(Protocol):
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_stream: StreamHandler | None = None,
    ) -> dict[str, Any]:
        """Generate the next model response."""
        ...


@dataclass
class AgentResponse:
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    error: str | None = None
    usage: dict[str, int] | None = None

    @property
    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)


class AgentRuntime:
    def __init__(
        self,
        server: AgentServer,
        model: ModelClient,
        max_iterations: int = 10,
        *,
        tool_policy: ToolPolicy | None = None,
        system_prompt_suffix: str | None = None,
        max_tool_calls_total: int | None = None,
        affect_task_lifecycle: bool = True,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if max_tool_calls_total is not None and max_tool_calls_total < 1:
            raise ValueError("max_tool_calls_total must be at least 1.")

        self.server = server
        self.model = model
        self.max_iterations = max_iterations
        self.tool_policy = tool_policy or ToolPolicy()
        self.system_prompt_suffix = (
            system_prompt_suffix.strip()
            if system_prompt_suffix
            else None
        )
        self.max_tool_calls_total = max_tool_calls_total
        self.affect_task_lifecycle = affect_task_lifecycle

        self._tool_call_count = 0
        self._max_tool_calls_per_iter = 5
        self._cooldown_interval = 0.1
        self._last_tool_call_time = 0.0
        self._last_heartbeat = {}
        self._tool_budget_exhausted = False
        self.stream_handler: StreamHandler | None = None
        self.delegation_manager = None

        self.total_tool_calls = 0
        self.modified_files: set[str] = set()
        self.usage_totals = self._empty_usage()
        self.last_stop_reason = "idle"

    _FILE_MUTATING_TOOLS = {
        "write_file",
        "create_file",
        "edit_file_lines",
        "move_item",
        "delete_item",
    }

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _begin_run(self, *, reset_delegations: bool) -> None:
        self._tool_call_count = 0
        self._tool_budget_exhausted = False
        self.total_tool_calls = 0
        self.modified_files.clear()
        self.usage_totals = self._empty_usage()
        self.last_stop_reason = "running"

        if reset_delegations and self.delegation_manager is not None:
            self.delegation_manager.reset_turn()

    def set_delegation_manager(self, manager) -> None:
        self.delegation_manager = manager

    def _update_file_state(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        if tool_name not in self._FILE_MUTATING_TOOLS:
            return

        if isinstance(result, dict) and result.get("is_error"):
            return

        if tool_name == "move_item":
            candidates = [arguments.get("old_path"), arguments.get("new_path")]
        else:
            candidates = [arguments.get("path")]

        paths = [path for path in candidates if isinstance(path, str)]
        if not paths:
            return

        self.modified_files.update(paths)

        state = self.server.get_active_task_state()
        if not state.active:
            return

        changed = False
        for path in paths:
            if path not in state.files:
                state.files.append(path)
                changed = True
        if changed:
            self.server.save_active_task_state(state)

    def set_stream_handler(
        self,
        handler: StreamHandler | None,
    ) -> None:
        self.stream_handler = handler

    def _emit_event(self, event: AgentEvent) -> None:
        if self.stream_handler is not None:
            self.stream_handler(event)

    def create_chat_history(self) -> list[dict[str, Any]]:
        prompt = build_system_prompt(
            self.server.context.identity
        )
        if self.system_prompt_suffix:
            prompt = f"{prompt}\n\n{self.system_prompt_suffix}"

        return [
            {
                "role": "system",
                "content": prompt,
            }
        ]

    async def run(self, task: str) -> str:
        task = task.strip()

        if not task:
            raise ValueError("task cannot be empty")

        self._begin_run(reset_delegations=True)
        self._initialize_task(task)

        messages = self.create_chat_history()
        messages.append(
            {
                "role": "user",
                "content": task,
            }
        )

        return await self._run_messages(
            messages,
            auto_complete_task=True,
        )

    async def run_chat_turn(
        self,
        messages: list[dict[str, Any]],
        user_message: str,
    ) -> str:
        user_message = user_message.strip()

        if not user_message:
            raise ValueError("user_message cannot be empty")

        self._begin_run(reset_delegations=True)

        if not messages:
            messages.extend(self.create_chat_history())

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return await self._run_messages(
            messages,
            auto_complete_task=False,
        )

    async def run_subtask(self, task: str) -> str:
        """Run an isolated one-shot worker without creating a persistent task."""
        task = task.strip()
        if not task:
            raise ValueError("task cannot be empty")

        self._begin_run(reset_delegations=False)
        messages = self.create_chat_history()
        messages.append({"role": "user", "content": task})
        return await self._run_messages(
            messages,
            auto_complete_task=False,
        )

    async def _run_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        auto_complete_task: bool,
    ) -> str:
        for _ in range(self.max_iterations):
            if self._tool_budget_exhausted:
                self.last_stop_reason = "tool_budget"
                return await self._generate_bounded_final(
                    messages,
                    reason=(
                        "The runtime tool-call budget for this agent was reached. "
                        "Do not call more tools. Summarize the useful evidence and "
                        "state what remains incomplete."
                    ),
                )

            self._tool_call_count = 0
            tools = await self._get_tools()

            response = await self.model.generate(
                messages=messages,
                tools=tools,
                on_stream=self.stream_handler,
            )

            agent_response = self._parse_response(response)
            self._record_usage(agent_response.usage)

            if agent_response.usage:
                messages.append(
                    {
                        "role": "system",
                        "content": self._usage_message(agent_response.usage),
                    }
                )

            if agent_response.error is not None:
                messages.append(
                    {
                        "role": "tool",
                        "content": {
                            "is_error": True,
                            "error": agent_response.error,
                        },
                    }
                )
                continue

            if not agent_response.is_tool_call:
                content = agent_response.content or ""
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

                state = self.server.get_active_task_state()
                if (
                    self.affect_task_lifecycle
                    and auto_complete_task
                    and state.active
                    and state.status == "in_progress"
                ):
                    state = self.server.finish_active_task("completed")

                self._last_heartbeat = {
                    "status": state.status if state.active else "idle",
                    "task_id": state.task_id,
                    "task": state.title if state.active else None,
                    "iteration": self._tool_call_count,
                    "last_tool": None,
                }
                self.last_stop_reason = "completed"
                return content

            tool_calls = agent_response.tool_calls or []
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call["id"],
                            "type": tool_call["type"],
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(
                                    tool_call["arguments"]
                                ),
                            },
                        }
                        for tool_call in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:
                self._emit_event(
                    AgentEvent(
                        kind="tool_call",
                        tool_name=tool_call["name"],
                        tool_arguments=tool_call.get("arguments", {}),
                    )
                )

                tool_result = await self._execute_tool(tool_call)
                content = (
                    tool_result.structured_content
                    if hasattr(tool_result, "structured_content")
                    else tool_result
                )

                is_error = (
                    isinstance(content, dict)
                    and content.get("is_error") is True
                )
                error_text = ""
                if is_error:
                    error_text = str(
                        content.get("error", "Tool execution failed.")
                    )

                self._emit_event(
                    AgentEvent(
                        kind="tool_result",
                        text=error_text,
                        tool_name=tool_call["name"],
                        is_error=is_error,
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(content),
                    }
                )

        self.last_stop_reason = "iteration_limit"
        return await self._generate_bounded_final(
            messages,
            reason=(
                "The agent runtime stopped further execution because the maximum "
                "iteration limit was reached. This was an automatic safety limit, "
                "not a user request. Summarize the current state, explain what "
                "prevented completion, and identify what should happen next."
            ),
            mark_task_blocked=True,
        )

    async def _generate_bounded_final(
        self,
        messages: list[dict[str, Any]],
        *,
        reason: str,
        mark_task_blocked: bool = False,
    ) -> str:
        messages.append({"role": "system", "content": reason})
        response = await self.model.generate(
            messages=messages,
            tools=[],
            on_stream=self.stream_handler,
        )
        agent_response = self._parse_response(response)
        self._record_usage(agent_response.usage)

        if agent_response.usage:
            messages.append(
                {
                    "role": "system",
                    "content": self._usage_message(agent_response.usage),
                }
            )

        if agent_response.error is not None:
            raise RuntimeError(
                "Failed to generate final response after runtime limit: "
                f"{agent_response.error}"
            )

        content = agent_response.content or ""
        messages.append({"role": "assistant", "content": content})

        if mark_task_blocked and self.affect_task_lifecycle:
            state = self.server.get_active_task_state()
            if state.active and state.status == "in_progress":
                blocker = "Maximum iteration limit reached before task completion."
                if blocker not in state.blocked:
                    state.blocked.append(blocker)
                state.status = "blocked"
                self.server.save_active_task_state(state)

        return content

    @staticmethod
    def _usage_message(usage: dict[str, int]) -> str:
        return (
            "\n[Session Stats: "
            f"Prompt={usage.get('prompt_tokens', 0)}, "
            f"Completion={usage.get('completion_tokens', 0)}, "
            f"Total={usage.get('total_tokens', 0)}]"
        )

    def _record_usage(self, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        for key in self.usage_totals:
            self.usage_totals[key] += int(usage.get(key, 0) or 0)

    def _initialize_task(self, task: str) -> None:
        self.server.start_task(
            title=self._task_title(task),
            goal=task,
        )

    @staticmethod
    def _task_title(task: str) -> str:
        compact = " ".join(task.split())
        if len(compact) <= 120:
            return compact
        return compact[:117].rstrip() + "..."

    async def _get_tools(self) -> list[dict[str, Any]]:
        tools = await self.server.mcp.list_tools()

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
            if self.tool_policy.allows(tool.name)
        ]

    @staticmethod
    def _parse_response(
        response: dict[str, Any],
    ) -> AgentResponse:
        choices = response.get("choices", [])
        usage = response.get("usage")

        if not choices:
            return AgentResponse(
                content=response.get("content", ""),
                usage=usage,
            )

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        reasoning = message.get("reasoning_content") or ""

        if tool_calls:
            parsed_tool_calls = []

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                arguments = function.get("arguments", {})

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        return AgentResponse(
                            error=f"Invalid tool arguments: {exc.msg}"
                        )

                parsed_tool_calls.append({
                    "id": tool_call.get("id"),
                    "type": tool_call.get("type", "function"),
                    "name": function["name"],
                    "arguments": arguments,
                })

            return AgentResponse(
                reasoning=reasoning,
                tool_calls=parsed_tool_calls,
                usage=usage,
            )

        return AgentResponse(
            content=message.get("content") or "",
            reasoning=reasoning,
            usage=usage,
        )

    async def _execute_tool(self, tool_call: dict[str, Any]) -> Any:
        name = tool_call["name"]
        arguments = tool_call.get("arguments", {})
        self._last_heartbeat["last_tool"] = name

        if not self.tool_policy.allows(name):
            return {
                "is_error": True,
                "error": f"Tool '{name}' is not permitted for this agent.",
            }

        self._tool_call_count += 1
        self.total_tool_calls += 1

        if self._tool_call_count > self._max_tool_calls_per_iter:
            return {
                "is_error": True,
                "error": (
                    "Tool call limit exceeded "
                    f"({self._max_tool_calls_per_iter}/iteration)."
                ),
            }

        if (
            self.max_tool_calls_total is not None
            and self.total_tool_calls > self.max_tool_calls_total
        ):
            self._tool_budget_exhausted = True
            return {
                "is_error": True,
                "error": (
                    "Total tool-call budget exceeded "
                    f"({self.max_tool_calls_total})."
                ),
            }

        now = time.monotonic()
        if now - self._last_tool_call_time < self._cooldown_interval:
            await asyncio.sleep(
                self._cooldown_interval - (now - self._last_tool_call_time)
            )
        self._last_tool_call_time = time.monotonic()

        try:
            result = await self.server.mcp.call_tool(name, arguments)
            content = getattr(result, "structured_content", result)
            self._update_file_state(name, arguments, content)
            self._last_heartbeat["last_tool"] = name
            return content
        except Exception as exc:
            error_msg = (
                str(exc)
                if isinstance(exc, ToolError)
                else f"Unexpected tool failure: {exc}"
            )
            return {"is_error": True, "error": error_msg}
