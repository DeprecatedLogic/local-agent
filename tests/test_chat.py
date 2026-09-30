from __future__ import annotations

import copy
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.chat import ChatSession
from agent.cli import async_main, create_parser
from agent.mcp_.agent_server import AgentServer
from agent.runtime import AgentRuntime


class SequenceModel:
    def __init__(
        self,
        responses: list[dict],
    ):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def generate(
        self,
        messages,
        tools,
    ):
        self.calls.append(
            {
                "messages": copy.deepcopy(
                    messages
                ),
                "tools": copy.deepcopy(
                    tools
                ),
            }
        )

        return self.responses.pop(0)


def _assistant_response(
    content: str,
) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


@pytest.mark.anyio
async def test_runtime_chat_turn_preserves_conversation_history(
    tmp_path,
):
    server = AgentServer(tmp_path)

    model = SequenceModel(
        [
            _assistant_response(
                "I will remember alpha."
            ),
            _assistant_response(
                "You told me alpha."
            ),
        ]
    )

    runtime = AgentRuntime(
        server,
        model,
        max_iterations=3,
    )

    messages = (
        runtime.create_chat_history()
    )

    first = await runtime.run_chat_turn(
        messages,
        "Remember the word alpha.",
    )

    second = await runtime.run_chat_turn(
        messages,
        "What word did I tell you?",
    )

    assert first == (
        "I will remember alpha."
    )

    assert second == (
        "You told me alpha."
    )

    second_request = (
        model.calls[1]["messages"]
    )

    assert {
        "role": "user",
        "content": (
            "Remember the word alpha."
        ),
    } in second_request

    assert {
        "role": "assistant",
        "content": (
            "I will remember alpha."
        ),
    } in second_request


@pytest.mark.anyio
async def test_runtime_chat_turn_preserves_tool_context(
    tmp_path,
):
    server = AgentServer(tmp_path)

    model = SequenceModel(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": (
                                        "call_project_info"
                                    ),
                                    "type": "function",
                                    "function": {
                                        "name": (
                                            "project_info"
                                        ),
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            _assistant_response(
                "I inspected the workspace."
            ),
            _assistant_response(
                "I still have the tool context."
            ),
        ]
    )

    runtime = AgentRuntime(
        server,
        model,
        max_iterations=3,
    )

    messages = (
        runtime.create_chat_history()
    )

    await runtime.run_chat_turn(
        messages,
        "Inspect the workspace.",
    )

    await runtime.run_chat_turn(
        messages,
        "Do you remember that inspection?",
    )

    second_turn_request = (
        model.calls[2]["messages"]
    )

    tool_call_message = next(
        message
        for message in second_turn_request
        if message["role"] == "assistant"
        and message.get("tool_calls")
    )

    tool_result_message = next(
        message
        for message in second_turn_request
        if message["role"] == "tool"
    )

    assert (
        tool_call_message[
            "tool_calls"
        ][0]["function"]["name"]
        == "project_info"
    )

    assert (
        tool_result_message[
            "tool_call_id"
        ]
        == "call_project_info"
    )

    assert (
        str(tmp_path.resolve())
        in tool_result_message["content"]
    )


class FakeState:
    def to_dict(self) -> dict:
        return {
            "task": "Example task",
            "status": "completed",
            "current": "",
            "completed": ["done"],
            "blocked": [],
            "files": [],
        }


class FakeStateStore:
    def load(self) -> FakeState:
        return FakeState()


class FakeRuntime:
    def __init__(self):
        self.server = SimpleNamespace(
            workspace=SimpleNamespace(
                root=Path("/tmp/project")
            ),
            state=FakeStateStore(),
        )

        self.snapshots: list[
            list[dict]
        ] = []

    def create_chat_history(
        self,
    ) -> list[dict]:
        return [
            {
                "role": "system",
                "content": "system",
            }
        ]

    async def run_chat_turn(
        self,
        messages: list[dict],
        user_message: str,
    ) -> str:
        self.snapshots.append(
            copy.deepcopy(messages)
        )

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        response = (
            f"reply: {user_message}"
        )

        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        return response


@pytest.mark.anyio
async def test_chat_session_keeps_history_between_inputs():
    runtime = FakeRuntime()
    output = io.StringIO()

    inputs = iter(
        [
            "first message",
            "second message",
            "/quit",
        ]
    )

    session = ChatSession(
        runtime,
        input_fn=lambda _: next(inputs),
        output=output,
    )

    await session.run()

    assert len(
        runtime.snapshots
    ) == 2

    assert {
        "role": "user",
        "content": "first message",
    } in runtime.snapshots[1]

    assert {
        "role": "assistant",
        "content": (
            "reply: first message"
        ),
    } in runtime.snapshots[1]


@pytest.mark.anyio
async def test_chat_clear_discards_conversation_history():
    runtime = FakeRuntime()
    output = io.StringIO()

    inputs = iter(
        [
            "first message",
            "/clear",
            "second message",
            "/quit",
        ]
    )

    session = ChatSession(
        runtime,
        input_fn=lambda _: next(inputs),
        output=output,
    )

    await session.run()

    assert len(
        runtime.snapshots
    ) == 2

    assert runtime.snapshots[1] == [
        {
            "role": "system",
            "content": "system",
        }
    ]

    assert (
        "Conversation history cleared."
        in output.getvalue()
    )


@pytest.mark.anyio
async def test_chat_help_and_state_commands_do_not_call_model():
    runtime = FakeRuntime()
    output = io.StringIO()

    inputs = iter(
        [
            "/help",
            "/state",
            "/exit",
        ]
    )

    session = ChatSession(
        runtime,
        input_fn=lambda _: next(inputs),
        output=output,
    )

    await session.run()

    rendered = output.getvalue()

    assert runtime.snapshots == []
    assert "/clear" in rendered
    assert (
        '"task": "Example task"'
        in rendered
    )


def test_parser_accepts_chat_flag(
    tmp_path,
):
    args = create_parser().parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--chat",
        ]
    )

    assert args.chat is True


@pytest.mark.anyio
async def test_cli_chat_path_awaits_chat_session(
    monkeypatch,
    tmp_path,
):
    parser = create_parser()

    args = parser.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--chat",
        ]
    )

    state = SimpleNamespace()

    state_store = SimpleNamespace(
        path=(
            tmp_path
            / ".agent"
            / "state.json"
        ),
        load=lambda: state,
        save=lambda _: None,
    )

    runtime = SimpleNamespace(
        server=SimpleNamespace(
            state=state_store,
        )
    )

    monkeypatch.setattr(
        "agent.cli.create_runtime",
        lambda args, dry_run=False: (
            runtime
        ),
    )

    run_chat_mock = AsyncMock()

    monkeypatch.setattr(
        "agent.cli.run_chat",
        run_chat_mock,
    )

    class FakeHealthMonitor:
        def __init__(
            self,
            state_path,
            interval,
        ):
            self.state_path = state_path
            self.interval = interval

        async def start(self):
            pass

        async def stop(self):
            pass

    monkeypatch.setattr(
        "agent.cli.HealthMonitor",
        FakeHealthMonitor,
    )

    monkeypatch.setattr(
        "agent.cli._notify_systemd",
        lambda _: None,
    )

    await async_main(
        args,
        parser,
    )

    run_chat_mock.assert_awaited_once_with(
        runtime,
        initial_message=None,
    )
