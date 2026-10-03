from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
from agent.mcp_.agent_server import AgentServer
from fastmcp.exceptions import ToolError
import json
import time
import asyncio
from agent.events import AgentEvent, StreamHandler
from agent.context import build_system_prompt



class ModelClient(Protocol):
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_stream: StreamHandler | None = None
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
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")

        self.server = server
        self.model = model
        self.max_iterations = max_iterations
        self._tool_call_count = 0
        self._max_tool_calls_per_iter = 5
        self._cooldown_interval = 0.1  # seconds between tool calls
        self._last_tool_call_time = 0.0
        self._last_heartbeat = {}
        self.stream_handler = None

    _FILE_MUTATING_TOOLS = {
        "write_file",
        "create_file",
        "edit_file_lines",
    }

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

        path = arguments.get("path")
        if not isinstance(path, str):
            return

        state = self.server.get_active_task_state()
        if not state.active:
            return

        if path not in state.files:
            state.files.append(path)
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
        return [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.server.context.identity
                ),
            }
        ]

    async def run(self, task: str) -> str:
        task = task.strip()

        if not task:
            raise ValueError(
                "task cannot be empty"
            )

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

    async def run_chat_turn(self, messages: list[dict[str, Any]], user_message: str) -> str:
        user_message = user_message.strip()

        if not user_message:
            raise ValueError(
                "user_message cannot be empty"
            )

        if not messages:
            messages.extend(
                self.create_chat_history()
            )

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

    async def _run_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        auto_complete_task: bool,
    ) -> str:
        for _ in range(self.max_iterations):
            self._tool_call_count = 0

            tools = await self._get_tools()

            response = await self.model.generate(
                messages=messages,
                tools=tools,
                on_stream=self.stream_handler
            )

            agent_response = (
                self._parse_response(response)
            )

            if agent_response.usage:
                stats = (
                    "\n[Session Stats: "
                    f"Prompt={agent_response.usage['prompt_tokens']}, "
                    f"Completion={agent_response.usage['completion_tokens']}, "
                    f"Total={agent_response.usage['total_tokens']}]"
                )

                messages.append(
                    {
                        "role": "system",
                        "content": stats,
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
                content = (
                    agent_response.content or ""
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

                state = self.server.get_active_task_state()
                if (
                    auto_complete_task
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

                return content

            tool_calls = (
                agent_response.tool_calls
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call["id"],
                            "type": (
                                tool_call["type"]
                            ),
                            "function": {
                                "name": (
                                    tool_call["name"]
                                ),
                                "arguments": (
                                    json.dumps(
                                        tool_call[
                                            "arguments"
                                        ]
                                    )
                                ),
                            },
                        }
                        for tool_call
                        in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:
                self._emit_event(
                    AgentEvent(
                        kind="tool_call",
                        tool_name=tool_call["name"],
                        tool_arguments=tool_call.get(
                            "arguments",
                            {},
                        ),
                    )
                )

                tool_result = (
                    await self._execute_tool(
                        tool_call
                    )
                )

                content = (
                    tool_result.structured_content
                    if hasattr(
                        tool_result,
                        "structured_content",
                    )
                    else tool_result
                )

                is_error = (
                    isinstance(content, dict)
                    and content.get("is_error") is True
                )

                error_text = ""
                if is_error:
                    error_text = str(
                        content.get(
                            "error",
                            "Tool execution failed.",
                        )
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
                        "tool_call_id": (
                            tool_call["id"]
                        ),
                        "content": (
                            json.dumps(content)
                        ),
                    }
                )

        messages.append(
            {
                "role": "system",
                "content": (
                    "The agent runtime stopped "
                    "further execution because the "
                    "maximum iteration limit was "
                    "reached. This was an automatic "
                    "safety limit, not a user request. "
                    "You may summarize the current "
                    "state, explain what prevented "
                    "completion, and identify what "
                    "should happen next."
                ),
            }
        )

        response = await self.model.generate(
            messages=messages,
            tools=[],
            on_stream=self.stream_handler
        )

        agent_response = (
            self._parse_response(response)
        )

        if agent_response.usage:
            stats = (
                "\n[Session Stats: "
                f"Prompt={agent_response.usage['prompt_tokens']}, "
                f"Completion={agent_response.usage['completion_tokens']}, "
                f"Total={agent_response.usage['total_tokens']}]"
            )

            messages.append(
                {
                    "role": "system",
                    "content": stats,
                }
            )

        if agent_response.error is not None:
            raise RuntimeError(
                "Failed to generate final response "
                "after iteration limit: "
                f"{agent_response.error}"
            )

        content = agent_response.content or ""

        messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )

        state = self.server.get_active_task_state()
        if state.active and state.status == "in_progress":
            blocker = (
                "Maximum iteration limit reached before task completion."
            )
            if blocker not in state.blocked:
                state.blocked.append(blocker)
            state.status = "blocked"
            self.server.save_active_task_state(state)

        return content

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
                usage=usage
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
                        return AgentResponse(error=f"Invalid tool arguments: {exc.msg}")

                parsed_tool_calls.append({
                    "id": tool_call.get("id"),
                    "type": tool_call.get("type", "function"),
                    "name": function["name"],
                    "arguments": arguments,
                })

            return AgentResponse(
                reasoning=reasoning,
                tool_calls=parsed_tool_calls,
                usage=usage
            )

        return AgentResponse(
            content=message.get("content") or "",
            reasoning=reasoning,
            usage=usage
        )

    async def _execute_tool(self, tool_call: dict[str, Any]) -> Any:
        name = tool_call["name"]
        arguments = tool_call.get("arguments", {})
        self._last_heartbeat["last_tool"] = name

        # Rate limiting guard
        self._tool_call_count += 1
        if self._tool_call_count > self._max_tool_calls_per_iter:
            return {"is_error": True, "error": f"Tool call limit exceeded ({self._max_tool_calls_per_iter}/iter). Refining plan."}

        # Cooldown between calls to prevent CPU spinning
        now = time.monotonic()
        if now - self._last_tool_call_time < self._cooldown_interval:
            await asyncio.sleep(self._cooldown_interval - (now - self._last_tool_call_time))
        self._last_tool_call_time = time.monotonic()

        try:
            result = await self.server.mcp.call_tool(name, arguments)
            content = getattr(result, "structured_content", result)
            self._update_file_state(name, arguments, content)
            self._last_heartbeat["last_tool"] = name
            return content
        except Exception as exc:
            # Catch everything that isn't a known ToolError to prevent loop crashes
            error_msg = str(exc) if isinstance(exc, ToolError) else f"Unexpected tool failure: {exc}"
            return {"is_error": True, "error": error_msg}

