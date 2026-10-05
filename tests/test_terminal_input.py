from __future__ import annotations

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agent.terminal_input import TerminalInput


async def _read_from_bytes(data: str) -> str:
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            multiline=True,
            wrap_lines=True,
            key_bindings=TerminalInput._create_key_bindings(),
            prompt_continuation=TerminalInput._continuation_prompt,
            input=pipe_input,
            output=DummyOutput(),
        )
        editor = TerminalInput(session)

        pipe_input.send_text(data)
        return await editor.read()


@pytest.mark.anyio
async def test_enter_accepts_message():
    assert await _read_from_bytes("hello\r") == "hello"


@pytest.mark.anyio
async def test_alt_enter_inserts_newline():
    result = await _read_from_bytes(
        "first\x1b\rsecond\r"
    )
    assert result == "first\nsecond"


@pytest.mark.anyio
async def test_ctrl_j_inserts_newline():
    result = await _read_from_bytes(
        "first\nsecond\r"
    )
    assert result == "first\nsecond"


def test_continuation_prompt_matches_prompt_width():
    assert TerminalInput._continuation_prompt(4, 2, False) == "    "
    assert TerminalInput._continuation_prompt(4, 2, True) == "    "


def test_prompt_uses_expected_text_and_styles():
    fragments = list(TerminalInput.prompt_message())
    assert fragments == [
        ("class:prompt", ">>>"),
        ("", " "),
    ]
