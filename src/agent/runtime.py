from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
from agent.mcp_.agent_server import AgentServer
from fastmcp.exceptions import ToolError
import json
import time
import asyncio


TASK_STATE_SYSTEM_PROMPT = """
You are an autonomous coding agent operating inside a bounded workspace.

You have access to tools for inspecting and modifying the project.

Task state has the following semantics:

- `task` is the user's original task. It is runtime-owned and cannot be changed.
- `status` is the runtime's execution status. It is runtime-owned and cannot be changed.
- `files` contains files actually modified or created by the runtime. It is runtime-owned.
- `current` describes the task-level work you are currently performing.
- `completed` is your current understanding of which task-level steps are complete.
- `blocked` contains task-level blockers or reasons you currently believe prevent progress.

Use `set_task_state` to keep `current`, `completed`, and `blocked` up to date.

`completed` and `blocked` are replacement-based lists. When updating either one,
provide the complete current list rather than only newly added items.

Your task-state description is not authoritative about what actually happened.
The runtime determines execution status and records files that were modified.
Do not claim a task is complete merely because you set `completed`; completion is
determined by the runtime after you finish the requested work.

When you encounter an error, inspect the error and attempt to recover when possible.
Do not perform actions outside the available tools or invent tool results.
""".strip()

class ModelClient(Protocol):
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate the next model response."""
        ...

@dataclass
class AgentResponse:
    content: str | None = None
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

        state = self.server.state.load()

        if path not in state.files:
            state.files.append(path)

        self.server.state.save(state)

    async def run(self, task: str) -> str:
        self._initialize_task(task)

        messages = [
            {
                "role": "system",
                "content": TASK_STATE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": task,
            },
        ]

        for i in range(self.max_iterations):
            self._tool_call_count = 0

            tools = await self._get_tools()
            response = await self.model.generate(
                messages=messages,
                tools=tools,
            )

            agent_response = self._parse_response(response)
            if agent_response.usage:
                stats = (
                    f"\n[Session Stats: Prompt={agent_response.usage['prompt_tokens']}, "
                    f"Completion={agent_response.usage['completion_tokens']}, "
                    f"Total={agent_response.usage['total_tokens']}]"
                )
                messages.append({"role": "system", "content": stats})

            if agent_response.error is not None:
                messages.append({
                    "role": "tool",
                    "content": {
                        "is_error": True,
                        "error": agent_response.error,
                    },
                })
                continue

            if not agent_response.is_tool_call:
                state = self.server.state.load()
                state.status = "completed"
                self.server.state.save(state)
                self._last_heartbeat = {
                    "status": state.status,
                    "task": state.task,
                    "iteration": self._tool_call_count,
                    "last_tool": None,  # Updated in _execute_tool
                }

                return agent_response.content or ""

            tool_calls = agent_response.tool_calls

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "type": tool_call["type"],
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"]),
                        },
                    }
                    for tool_call in tool_calls
                ],
            })

            for tool_call in tool_calls:
                tool_result = await self._execute_tool(tool_call)

                content = (
                    tool_result.structured_content
                    if hasattr(tool_result, "structured_content")
                    else tool_result
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(content),
                })

        messages.append({
            "role": "system",
            "content": (
                "The agent runtime stopped further execution because the maximum "
                "iteration limit was reached. This was an automatic safety limit, "
                "not a user request. You may summarize the current state, explain "
                "what prevented completion, and identify what should happen next."
            ),
        })

        response = await self.model.generate(
            messages=messages,
            tools=[],
        )

        agent_response = self._parse_response(response)
        if agent_response.usage:
                stats = (
                    f"\n[Session Stats: Prompt={agent_response.usage['prompt_tokens']}, "
                    f"Completion={agent_response.usage['completion_tokens']}, "
                    f"Total={agent_response.usage['total_tokens']}]"
                )
                messages.append({"role": "system", "content": stats})

        if agent_response.error is not None:
            raise RuntimeError(
                f"Failed to generate final response after iteration limit: "
                f"{agent_response.error}"
            )

        state = self.server.state.load()
        state.status = "blocked"
        state.blocked.append(
            "Maximum iteration limit reached before task completion."
        )
        self.server.state.save(state)

        return agent_response.content or ""

    def _initialize_task(self, task: str) -> None:
        state = self.server.state.load()

        state.task = task
        state.status = "in_progress"
        state.current = ""
        state.completed = []
        state.blocked = []
        state.files = []

        self.server.state.save(state)

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
                tool_calls=parsed_tool_calls,
                usage=usage
            )

        return AgentResponse(
            content=message.get("content") or "",
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

