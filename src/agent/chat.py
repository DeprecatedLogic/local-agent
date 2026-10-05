from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from typing import TextIO

from agent.events import AgentEvent
from agent.runtime import AgentRuntime
from agent.terminal_input import TerminalInput

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
        input_fn: Callable[[str], str] | None = None,
        output: TextIO | None = None,
        terminal_input: TerminalInput | None = None,
    ):
        self.runtime = runtime
        self.input_fn = input_fn
        self.output = output or sys.stdout
        self.terminal_input = terminal_input
        self.messages = runtime.create_chat_history()
        self._base_message_count = len(self.messages)
        self._persisted_message_count = 0
        self.session_id: str | None = None
        self.chat_store = getattr(runtime.server, "chat_sessions", None)
        self.chat_access = getattr(runtime.server, "chat_access", None)
        self.renderer = TerminalStreamRenderer(self.output)
        self.runtime.set_stream_handler(self.renderer.handle)
        self._set_session_id(None)

    async def _read_input(self) -> str:
        if self.input_fn is not None:
            # Deterministic/headless injection path used by tests and embedders.
            # ANSI styling stays out of the returned text exactly as before.
            return await asyncio.to_thread(
                self.input_fn,
                ">>> ",
            )

        if self.terminal_input is None:
            self.terminal_input = TerminalInput()

        # Keep the visual spacing the old prompt had without putting a newline
        # inside terminal-width bookkeeping. prompt_toolkit owns all rendering
        # from this point until the input is accepted.
        print(
            file=self.output,
            flush=True,
        )
        return await self.terminal_input.read()

    async def run(self, initial_message: str | None = None) -> None:
        self._print_banner()

        if initial_message:
            await self._run_user_message(initial_message)

        while True:
            try:
                raw = await self._read_input()
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

    async def _run_user_message(
        self,
        message: str,
    ) -> None:
        error: Exception | None = None
        message_count_before_turn = len(self.messages)

        try:
            await self.runtime.run_chat_turn(
                self.messages,
                message,
            )
        except Exception as exc:
            error = exc
            # Do not carry a partially constructed user/tool turn into the next
            # request or persist it as a resumable conversation.
            del self.messages[message_count_before_turn:]
        finally:
            self.renderer.finish()

        if error is None:
            try:
                self._persist_completed_turn(message)
                self._sync_active_task_pointer()
            except Exception as exc:
                self._write(f"[warning] chat persistence failed: {exc}")
        else:
            self._sync_active_task_pointer(best_effort=True)

        if error is not None:
            self._write(
                f"[error] {error}"
            )

    def _handle_command(self, command: str) -> bool:
        raw_command = command.strip()
        command_name, _, argument = raw_command.partition(" ")
        command_name = command_name.lower()
        argument = argument.strip()

        if command_name in {"/exit", "/quit"}:
            return True

        if command_name == "/help":
            self._write(
                "\n".join(
                    (
                        "Commands:",
                        "  /help                 Show this help message",
                        "  /state                Show the current task state",
                        "  /sessions             List recent saved chat sessions",
                        "  /resume <session-id>  Resume a saved chat session",
                        "  /new                  Start a fresh conversation",
                        "  /clear                Clear context by starting fresh",
                        "  /delete <session-id>  Permanently delete one session",
                        "  /delete-current       Permanently delete this saved session",
                        "  /delete-all           Permanently delete all saved sessions",
                        "  /allow-sessions <id...|all>  Allow temporary cross-session reads",
                        "  /allowed-sessions     Show cross-session read permissions",
                        "  /deny-sessions <id...|all>   Revoke cross-session reads",
                        "  /exit                 Exit chat mode",
                        "  /quit                 Exit chat mode",
                    )
                )
            )
            return False

        if command_name == "/state":
            state = self.runtime.server.state.load().to_dict()

            self._write(
                json.dumps(
                    state,
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return False

        if command_name == "/sessions":
            self._list_sessions()
            return False

        if command_name == "/resume":
            if not argument:
                self._write("Usage: /resume <session-id>")
                return False

            try:
                session = self._resume_session(argument)
            except Exception as exc:
                self._write(f"[error] {exc}")
                return False

            self._write(
                f"Resumed {self._display_session_id(session.id)}: "
                f"{session.title} ({session.message_count} messages)."
            )
            return False

        if command_name == "/new":
            self._start_new_conversation()
            self._write("Started a new conversation.")
            return False

        if command_name == "/clear":
            self._start_new_conversation()
            self._write(
                "Conversation history cleared. Started a new conversation."
            )
            return False

        if command_name == "/delete":
            if not argument:
                self._write("Usage: /delete <session-id>")
                return False
            try:
                deleted = self._delete_session(argument)
            except Exception as exc:
                self._write(f"[error] {exc}")
                return False
            self._write(
                f"Permanently deleted {self._display_session_id(deleted)}."
            )
            return False

        if command_name == "/delete-current":
            if self.session_id is None:
                self._write("Current conversation has not been saved yet.")
                return False
            deleted = self._delete_session(self.session_id)
            self._write(
                f"Permanently deleted {self._display_session_id(deleted)}. "
                "Started a new conversation."
            )
            return False

        if command_name == "/delete-all":
            if self.chat_store is None:
                self._write("Persistent chat sessions are unavailable.")
                return False
            count = self.chat_store.delete_all_sessions()
            self._start_new_conversation()
            noun = "session" if count == 1 else "sessions"
            self._write(
                f"Permanently deleted {count} saved {noun}. "
                "Started a new conversation."
            )
            return False

        if command_name == "/allow-sessions":
            self._update_session_access(argument, allow=True)
            return False

        if command_name == "/deny-sessions":
            self._update_session_access(argument, allow=False)
            return False

        if command_name == "/allowed-sessions":
            self._show_session_access()
            return False

        self._write(
            f"Unknown command: {raw_command}"
        )
        self._write(
            "Use /help to list available commands."
        )

        return False

    @staticmethod
    def _session_title(message: str) -> str:
        compact = " ".join(message.split())
        if len(compact) <= 80:
            return compact
        return compact[:77].rstrip() + "..."

    @staticmethod
    def _display_session_id(session_id: str) -> str:
        return session_id[:20]

    def _persist_completed_turn(self, first_message: str) -> None:
        if self.chat_store is None:
            return

        if self.session_id is None:
            session = self.chat_store.create_session(
                self._session_title(first_message)
            )
            self._set_session_id(session.id)

        persistent_messages = self.messages[self._base_message_count:]
        new_messages = persistent_messages[self._persisted_message_count:]

        if not new_messages:
            return

        session = self.chat_store.append_messages(
            self.session_id,
            new_messages,
            expected_count=self._persisted_message_count,
        )
        self._persisted_message_count = session.message_count

    def _sync_active_task_pointer(self, *, best_effort: bool = False) -> None:
        if self.chat_store is None or self.session_id is None:
            return

        getter = getattr(
            self.runtime.server,
            "get_active_task_state",
            None,
        )
        if getter is None:
            return

        try:
            state = getter()
            task_id = (
                state.task_id
                if state.active
                and state.status in {"in_progress", "blocked"}
                else None
            )
            self.chat_store.set_active_task(
                self.session_id,
                task_id,
            )
        except Exception:
            if not best_effort:
                raise

    def _restore_active_task_pointer(self, active_task_id: str | None) -> None:
        attach = getattr(self.runtime.server, "attach_task", None)
        if attach is None:
            return

        if active_task_id is None:
            attach(None)
            return

        try:
            state = attach(active_task_id)
        except KeyError:
            if self.chat_store is not None and self.session_id is not None:
                self.chat_store.set_active_task(self.session_id, None)
            self._write(
                f"[warning] linked task no longer exists: {active_task_id}"
            )
            return

        if not state.active and self.chat_store is not None and self.session_id is not None:
            # A completed/cancelled task should not remain an active-session link.
            self.chat_store.set_active_task(self.session_id, None)

    def _resume_session(self, reference: str):
        if self.chat_store is None:
            raise RuntimeError("Persistent chat sessions are unavailable.")

        self._sync_active_task_pointer(best_effort=True)

        session_id = self.chat_store.resolve_session_id(reference)
        session = self.chat_store.get_session(session_id)
        stored_messages = self.chat_store.load_messages(session_id)

        self._reset_session_access()

        base_messages = self.runtime.create_chat_history()
        self.messages[:] = base_messages + stored_messages
        self._base_message_count = len(base_messages)
        self._persisted_message_count = len(stored_messages)
        self._set_session_id(session_id)

        self._restore_active_task_pointer(session.active_task_id)
        return self.chat_store.get_session(session_id)

    def _start_new_conversation(self) -> None:
        self._sync_active_task_pointer(best_effort=True)

        attach = getattr(self.runtime.server, "attach_task", None)
        if attach is not None:
            attach(None)

        base_messages = self.runtime.create_chat_history()
        self.messages[:] = base_messages
        self._base_message_count = len(base_messages)
        self._persisted_message_count = 0
        self._set_session_id(None)
        self._reset_session_access()

    def _set_session_id(self, session_id: str | None) -> None:
        self.session_id = session_id
        server = getattr(self.runtime, "server", None)
        if server is not None and hasattr(server, "current_chat_session_id"):
            server.current_chat_session_id = session_id

    def _reset_session_access(self) -> None:
        if self.chat_access is not None:
            self.chat_access.reset()

    def _delete_session(self, reference: str) -> str:
        if self.chat_store is None:
            raise RuntimeError("Persistent chat sessions are unavailable.")

        session_id = self.chat_store.resolve_session_id(reference)
        was_current = session_id == self.session_id
        self.chat_store.delete_session(session_id)
        if self.chat_access is not None:
            self.chat_access.forget_session(session_id)

        if was_current:
            self._start_new_conversation()

        return session_id

    def _update_session_access(self, argument: str, *, allow: bool) -> None:
        if self.chat_store is None or self.chat_access is None:
            self._write("Cross-session access controls are unavailable.")
            return

        references = argument.split()
        command = "/allow-sessions" if allow else "/deny-sessions"
        if not references:
            self._write(f"Usage: {command} <session-id...|all>")
            return

        if "all" in {reference.lower() for reference in references}:
            if len(references) != 1:
                self._write("[error] 'all' cannot be combined with session IDs.")
                return
            if allow:
                self.chat_access.allow_all_sessions()
                self._write("Cross-session reads allowed for all saved sessions.")
            else:
                self.chat_access.deny_all_sessions()
                self._write("Cross-session reads revoked for all saved sessions.")
            return

        try:
            resolved = [
                self.chat_store.resolve_session_id(reference)
                for reference in references
            ]
        except Exception as exc:
            self._write(f"[error] {exc}")
            return

        for session_id in resolved:
            if allow:
                self.chat_access.allow_session(session_id)
            else:
                self.chat_access.deny_session(session_id)

        verb = "Allowed" if allow else "Revoked"
        rendered = ", ".join(
            self._display_session_id(session_id)
            for session_id in resolved
        )
        self._write(f"{verb} cross-session reads: {rendered}")

    def _show_session_access(self) -> None:
        if self.chat_access is None:
            self._write("Cross-session access controls are unavailable.")
            return

        state = self.chat_access.to_dict()
        if state["mode"] == "all":
            lines = ["Cross-session reads: all saved sessions."]
            denied = state["denied_session_ids"]
            if denied:
                lines.append(
                    "Denied: "
                    + ", ".join(
                        self._display_session_id(session_id)
                        for session_id in denied
                    )
                )
            self._write("\n".join(lines))
            return

        allowed = state["allowed_session_ids"]
        if not allowed:
            self._write("Cross-session reads: none.")
            return
        self._write(
            "Cross-session reads: "
            + ", ".join(
                self._display_session_id(session_id)
                for session_id in allowed
            )
        )

    def _list_sessions(self) -> None:
        if self.chat_store is None:
            self._write("Persistent chat sessions are unavailable.")
            return

        sessions = self.chat_store.list_sessions(limit=10)
        if not sessions:
            self._write("No saved chat sessions.")
            return

        lines = ["Saved sessions:"]
        for session in sessions:
            marker = "*" if session.id == self.session_id else " "
            lines.append(
                f" {marker} {self._display_session_id(session.id)}  "
                f"{session.message_count:>4} msgs  {session.title}"
            )

        lines.append("Use /resume <shown-id> to resume a session.")
        self._write("\n".join(lines))

    def _print_banner(self) -> None:
        context = getattr(self.runtime.server, "context", None)
        identity = getattr(context, "identity", None)
        name = getattr(identity, "name", "Local Agent")

        self._write(f"Name: {name}", newline=False)
        self._write(
            f"Workspace: "
            f"{self.runtime.server.workspace.root}",
            newline=False
        )
        self._write(
            "Type /help for commands.",
            newline=False
        )

    def _write(self, text: str = "", newline: bool = True) -> None:
        if newline:
            text += '\n'

        print(
            text,
            file=self.output,
            flush=True,
        )
