from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from typing import TextIO

try:
    import readline
except ImportError:
    readline = None

from agent.events import AgentEvent
from agent.runtime import AgentRuntime

if readline is not None:
    readline.set_history_length(1000)

class TerminalStreamRenderer:
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def __init__(self, output: TextIO):
        self.output = output
        self.mode: str | None = None

    def handle(self, event: AgentEvent) -> None:
        if event.kind == "reasoning":
            if self.mode != "reasoning":
                self._switch_to_reasoning()

            print(
                event.text,
                end="",
                file=self.output,
                flush=True,
            )
            return

        if event.kind == "content":
            if self.mode != "content":
                self._switch_to_content()

            print(
                event.text,
                end="",
                file=self.output,
                flush=True,
            )
            return

        if event.kind == "tool_call":
            self._render_tool_call(event)
            return

        if event.kind == "tool_result":
            self._render_tool_result(event)

    def _render_tool_call(self, event: AgentEvent) -> None:
        self._end_text_stream()

        arguments = self._format_tool_arguments(
            event.tool_arguments or {}
        )
        suffix = f" {arguments}" if arguments else ""

        print(
            f"[tool] {event.tool_name or '<unknown>'}{suffix}",
            file=self.output,
            flush=True,
        )

    def _render_tool_result(self, event: AgentEvent) -> None:
        self._end_text_stream()

        name = event.tool_name or "<unknown>"
        if event.is_error:
            detail = f": {event.text}" if event.text else ""
            line = f"[tool] {name} ✗{detail}"
        else:
            line = f"[tool] {name} ✓"

        print(
            line,
            file=self.output,
            flush=True,
        )

    @staticmethod
    def _compact_tool_value(value):
        if isinstance(value, str):
            if len(value) > 120:
                return f"<{len(value)} chars>"
            return value

        if isinstance(value, dict):
            return {
                key: TerminalStreamRenderer._compact_tool_value(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            if len(value) > 8:
                return [
                    TerminalStreamRenderer._compact_tool_value(item)
                    for item in value[:8]
                ] + [f"<{len(value) - 8} more items>"]

            return [
                TerminalStreamRenderer._compact_tool_value(item)
                for item in value
            ]

        return value

    @classmethod
    def _format_tool_arguments(
        cls,
        arguments: dict,
    ) -> str:
        if not arguments:
            return ""

        compact = cls._compact_tool_value(arguments)
        return json.dumps(
            compact,
            ensure_ascii=False,
            separators=(", ", ": "),
        )

    def _end_text_stream(self) -> None:
        if self.mode == "reasoning":
            print(
                self.RESET,
                end="",
                file=self.output,
                flush=True,
            )

        if self.mode in {"reasoning", "content"}:
            print(
                file=self.output,
                flush=True,
            )

        self.mode = None

    def _switch_to_reasoning(self) -> None:
        if self.mode is not None:
            print(
                self.RESET,
                file=self.output,
                flush=True,
            )

        print(
            f"\n{self.DIM}[thinking]\n",
            end="",
            file=self.output,
            flush=True,
        )

        self.mode = "reasoning"

    def _switch_to_content(self) -> None:
        if self.mode == "reasoning":
            print(
                f"{self.RESET}\n",
                file=self.output,
                flush=True,
            )
        elif self.mode is None:
            print(
                file=self.output,
                flush=True,
            )

        self.mode = "content"

    def finish(self) -> None:
        self._end_text_stream()

class ChatSession:
    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        input_fn: Callable[[str], str] = input,
        output: TextIO | None = None,
    ):
        self.runtime = runtime
        self.input_fn = input_fn
        self.output = output or sys.stdout
        self.messages = runtime.create_chat_history()
        self.renderer = TerminalStreamRenderer(self.output)
        self.runtime.set_stream_handler(self.renderer.handle)

    def _input_prompt(self) -> str:
        if readline is not None:
            return (
                f"\001\n{self.renderer.MAGENTA}\002"
                ">>>"
                f"\001{self.renderer.GREEN}\002 "
            )
        return f"\n{self.renderer.MAGENTA}>>>{self.renderer.GREEN}"

    async def run(self, initial_message: str | None = None) -> None:
        self._print_banner()

        if initial_message:
            await self._run_user_message(initial_message)

        while True:
            try:
                raw = await asyncio.to_thread(
                    self.input_fn,
                    self._input_prompt(),
                )
            except EOFError:
                self._write()
                return
            finally:
                print(
                    self.renderer.RESET,
                    end="",
                    file=self.output,
                    flush=True
                )

            message = raw.strip()

            if not message:
                continue

            if message.startswith("/"):
                if self._handle_command(message):
                    return

                continue

            await self._run_user_message(message)

    async def _run_user_message(
        self,
        message: str,
    ) -> None:
        error: Exception | None = None

        try:
            await self.runtime.run_chat_turn(
                self.messages,
                message,
            )
        except Exception as exc:
            error = exc
        finally:
            self.renderer.finish()

        if error is not None:
            self._write(
                f"[error] {error}"
            )

    def _handle_command(self, command: str) -> bool:
        command = command.lower()

        if command in {"/exit", "/quit"}:
            return True

        if command == "/help":
            self._write(
                "\n".join(
                    (
                        "Commands:",
                        "  /help   Show this help message",
                        "  /state  Show the current task state",
                        "  /clear  Clear conversation history",
                        "  /exit   Exit chat mode",
                        "  /quit   Exit chat mode",
                    )
                )
            )
            return False

        if command == "/state":
            state = self.runtime.server.state.load().to_dict()

            self._write(
                json.dumps(
                    state,
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return False

        if command == "/clear":
            self.messages[:] = (
                self.runtime.create_chat_history()
            )

            self._write(
                "Conversation history cleared."
            )
            return False

        self._write(
            f"Unknown command: {command}"
        )
        self._write(
            "Use /help to list available commands."
        )

        return False

    def _print_banner(self) -> None:
        context = getattr(self.runtime.server, "context", None)
        identity = getattr(context, "identity", None)
        name = getattr(identity, "name", "Local Agent")

        self._write(name)
        self._write(
            f"Workspace: "
            f"{self.runtime.server.workspace.root}"
        )
        self._write(
            "Type /help for commands."
        )

    def _write(self, text: str = "", newline: bool = True) -> None:
        if newline:
            text += '\n'

        print(
            text,
            file=self.output,
            flush=True,
        )
