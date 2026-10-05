from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style


class TerminalInput:
    """Interactive multiline terminal editor for chat input."""

    def __init__(self, session: PromptSession[str] | None = None):
        self.session = session or self._create_session()

    @staticmethod
    def _create_key_bindings() -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter")
        def _accept(event) -> None:
            # Keep Enter as the fast, familiar "send" action even though the
            # underlying prompt is multiline-capable.
            event.current_buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        @bindings.add("c-j")
        def _insert_newline(event) -> None:
            # Alt+Enter is encoded by VT terminals as Escape followed by Enter.
            # Ctrl+J is LF and also gives terminals a distinct code they can map
            # Shift+Enter to when legacy Shift+Enter is indistinguishable from
            # plain Enter.
            event.current_buffer.insert_text("\n")

        return bindings

    @staticmethod
    def _continuation_prompt(
        width: int,
        line_number: int,
        is_soft_wrap: bool,
    ) -> str:
        del line_number, is_soft_wrap
        return " " * width

    @classmethod
    def _create_session(cls) -> PromptSession[str]:
        return PromptSession(
            multiline=True,
            wrap_lines=True,
            history=InMemoryHistory(),
            key_bindings=cls._create_key_bindings(),
            prompt_continuation=cls._continuation_prompt,
            style=Style.from_dict(
                {
                    "": "ansigreen",
                    "prompt": "ansimagenta",
                }
            ),
            enable_history_search=True,
            erase_when_done=False,
        )

    @staticmethod
    def prompt_message() -> FormattedText:
        return FormattedText(
            [
                ("class:prompt", ">>>"),
                ("", " "),
            ]
        )

    async def read(self) -> str:
        return await self.session.prompt_async(
            self.prompt_message()
        )
