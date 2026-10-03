import json

import pytest

from agent.mcp_.agent_server import AgentServer
from agent.runtime import AgentRuntime


class FakeModel:
    def __init__(self, response):
        self.response = response

    async def generate(self, messages, tools, on_stream=None):
        return self.response


class SequenceModel:
    def __init__(self, responses):
        self.responses = list(responses)

    async def generate(self, messages, tools, on_stream=None):
        return self.responses.pop(0)


@pytest.mark.anyio
async def test_chat_turn_does_not_automatically_create_task(tmp_path):
    server = AgentServer(tmp_path)
    runtime = AgentRuntime(server, FakeModel({"content": "Hello!"}))
    messages = runtime.create_chat_history()

    result = await runtime.run_chat_turn(messages, "Hello")

    assert result == "Hello!"
    assert server.active_task_id is None
    assert server.tasks.list_tasks() == []


@pytest.mark.anyio
async def test_chat_can_explicitly_start_persistent_task(tmp_path):
    server = AgentServer(tmp_path)
    model = SequenceModel([
        {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_start_task",
                        "type": "function",
                        "function": {
                            "name": "start_task",
                            "arguments": json.dumps({
                                "title": "Inspect repository",
                                "goal": "Inspect the repository and report issues.",
                            }),
                        },
                    }],
                },
            }],
        },
        {"content": "I started the tracked investigation."},
    ])
    runtime = AgentRuntime(server, model)
    messages = runtime.create_chat_history()

    await runtime.run_chat_turn(
        messages,
        "Please inspect this repository carefully.",
    )

    state = server.get_active_task_state()
    assert state.active is True
    assert state.title == "Inspect repository"
    assert state.goal == "Inspect the repository and report issues."
    assert state.status == "in_progress"
    assert state.revision == 1


@pytest.mark.anyio
async def test_one_shot_mode_still_tracks_and_completes_task(tmp_path):
    server = AgentServer(tmp_path)
    runtime = AgentRuntime(server, FakeModel({"content": "Done."}))

    result = await runtime.run("Inspect the project")

    assert result == "Done."
    state = server.get_active_task_state()
    assert state.active is True
    assert state.goal == "Inspect the project"
    assert state.status == "completed"
    assert state.revision == 2
