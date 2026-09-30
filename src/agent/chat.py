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

from agent.runtime import AgentRuntime

if readline is not None:
    readline.set_history_length(1000)

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

    async def run(self, initial_message: str | None = None) -> None:
        self._print_banner()

        if initial_message:
            await self._run_user_message(initial_message)

        while True:
            try:
                raw = await asyncio.to_thread(
                    self.input_fn,
                    ">>> ",
                )
            except EOFError:
                self._write()
                return

            message = raw.strip()

            if not message:
                continue

            if message.startswith("/"):
                if self._handle_command(message):
                    return

                continue

            await self._run_user_message(message)

    async def _run_user_message(self, message: str) -> None:
        try:
            response = await self.runtime.run_chat_turn(
                self.messages,
                message,
            )
        except Exception as exc:
            self._write(f"[error] {exc}")
            return

        if response:
            self._write()
            self._write(response)
            self._write(newline=False)

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
        self._write("Local Agent")
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
