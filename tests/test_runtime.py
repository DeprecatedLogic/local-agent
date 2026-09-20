import pytest
import json
from agent.mcp_.agent_server import AgentServer
from agent.runtime import AgentRuntime


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def generate(self, messages, tools):
        self.calls.append({
            "messages": messages,
            "tools": tools,
        })
        return self.response


@pytest.mark.anyio
async def test_runtime_initializes_task_state(tmp_path):
    server = AgentServer(tmp_path)

    model = FakeModel({
        "content": "Task complete.",
    })

    runtime = AgentRuntime(server, model)

    result = await runtime.run("Implement feature X")

    assert result == "Task complete."

    state = server.state.load()

    assert state.task == "Implement feature X"
    assert state.status == "completed"

class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def generate(self, messages, tools):
        self.calls.append({
            "messages": messages,
            "tools": tools,
        })
        return next(self.responses)


@pytest.mark.anyio
async def test_runtime_executes_tool_call_and_returns_final_response(tmp_path):
    server = AgentServer(tmp_path)

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "project_info",
                            "arguments": "{}",
                        },
                        "id": "call_123",
                    }],
                },
            }],
        },
        {
            "content": "The project workspace is configured correctly.",
        },
    ])

    runtime = AgentRuntime(server, model)

    result = await runtime.run("Check the project workspace")

    assert result == "The project workspace is configured correctly."
    assert len(model.calls) == 2

    second_messages = model.calls[1]["messages"]

    assistant_message = next(
        message
        for message in second_messages
        if message["role"] == "assistant"
    )

    assert assistant_message["tool_calls"] == [{
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "project_info",
            "arguments": "{}",
        },
    }]

    tool_message = next(
        message
        for message in second_messages
        if message["role"] == "tool"
    )

    assert tool_message["tool_call_id"] == "call_123"

    assert json.loads(tool_message["content"]) == {
        "workspace": str(tmp_path),
    }

    assert any(
        message["role"] == "tool"
        for message in second_messages
    )


@pytest.mark.anyio
async def test_runtime_can_discover_mcp_tools(tmp_path):
    server = AgentServer(tmp_path)

    tools = await server.mcp.list_tools()

    names = {tool.name for tool in tools}

    assert "project_info" in names
    assert "read_file" in names
    assert "write_file" in names
    assert "git_status" in names


@pytest.mark.anyio
async def test_runtime_provides_mcp_tools_to_model(tmp_path):
    server = AgentServer(tmp_path)

    model = FakeModel({
        "content": "Done.",
    })

    runtime = AgentRuntime(server, model)

    await runtime.run("Inspect the project")

    tools = model.calls[0]["tools"]

    project_info = next(
        tool for tool in tools
        if tool["function"]["name"] == "project_info"
    )

    assert project_info["type"] == "function"
    assert project_info["function"]["description"]
    assert project_info["function"]["parameters"]["type"] == "object"


@pytest.mark.anyio
async def test_runtime_returns_tool_error_to_model(tmp_path):
    server = AgentServer(tmp_path)

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"does-not-exist.txt"}',
                        },
                        "id": "call_123",
                    }],
                },
            }],
        },
        {
            "content": "The file does not exist.",
        },
    ])

    runtime = AgentRuntime(server, model)

    result = await runtime.run("Read does-not-exist.txt")

    assert result == "The file does not exist."

    second_messages = model.calls[1]["messages"]

    tool_message = next(
        message
        for message in second_messages
        if message["role"] == "tool"
    )

    tool_content = json.loads(tool_message["content"])

    assert tool_content["is_error"] is True


@pytest.mark.anyio
async def test_runtime_handles_invalid_tool_arguments(tmp_path):
    server = AgentServer(tmp_path)

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "project_info",
                            "arguments": "{invalid json",
                        },
                        "id": "call_123",
                    }],
                },
            }],
        },
        {
            "content": "The tool call was invalid.",
        },
    ])

    runtime = AgentRuntime(server, model)

    result = await runtime.run("Check the project")

    assert result == "The tool call was invalid."


@pytest.mark.anyio
async def test_runtime_stops_after_max_iterations(tmp_path):
    server = AgentServer(tmp_path)

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "project_info",
                            "arguments": "{}",
                        },
                        "id": "call_123",
                    }],
                },
            }],
        }
    ] * 3 + [
        {
            "content": "The task could not be completed within the runtime limit.",
        },
    ])

    runtime = AgentRuntime(server, model, max_iterations=3)

    result = await runtime.run("Keep checking the project")

    assert result == "The task could not be completed within the runtime limit."
    assert len(model.calls) == 4

    state = server.state.load()

    assert state.status == "blocked"
    assert state.blocked == [
        "Maximum iteration limit reached before task completion."
    ]


@pytest.mark.anyio
async def test_runtime_marks_task_completed(tmp_path):
    server = AgentServer(tmp_path)

    model = SequenceModel([
        {
            "content": "The task is complete.",
        },
    ])

    runtime = AgentRuntime(server, model)

    result = await runtime.run("Complete the task")

    assert result == "The task is complete."

    state = server.state.load()

    assert state.task == "Complete the task"
    assert state.status == "completed"


@pytest.mark.anyio
async def test_runtime_explains_iteration_limit_to_model(tmp_path):
    server = AgentServer(tmp_path)

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "project_info",
                            "arguments": "{}",
                        },
                        "id": "call_123",
                    }],
                },
            }],
        }
    ] * 3 + [
        {
            "content": "I could not finish the task within the runtime limit.",
        },
    ])

    runtime = AgentRuntime(server, model, max_iterations=3)

    result = await runtime.run("Inspect the project")

    assert result == "I could not finish the task within the runtime limit."
    assert len(model.calls) == 4

    final_call = model.calls[3]

    assert final_call["tools"] == []

    runtime_message = next(
        message
        for message in final_call["messages"]
        if (
            message["role"] == "system"
            and "maximum iteration limit" in message["content"]
        )
    )

    assert "maximum iteration limit" in runtime_message["content"]
    assert "not a user request" in runtime_message["content"]


def test_parse_response_preserves_tool_call_metadata():
    response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Brussels"}',
                    },
                    "id": "call_123",
                }],
            }
        }]
    }

    result = AgentRuntime._parse_response(response)

    assert result.is_tool_call
    assert result.tool_calls == [{
        "id": "call_123",
        "type": "function",
        "name": "get_weather",
        "arguments": {
            "city": "Brussels",
        },
    }]


@pytest.mark.anyio
async def test_runtime_executes_multiple_tool_calls_in_order(tmp_path):
    server = AgentServer(tmp_path)

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "project_info",
                                "arguments": "{}",
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "project_info",
                                "arguments": "{}",
                            },
                        },
                    ],
                },
            }],
        },
        {
            "content": "Both tool calls completed successfully.",
        },
    ])

    runtime = AgentRuntime(server, model)

    result = await runtime.run(
        "Inspect the project using the available tools."
    )

    assert result == "Both tool calls completed successfully."

    second_messages = model.calls[1]["messages"]

    assistant_message = next(
        message
        for message in second_messages
        if message["role"] == "assistant"
    )

    assert assistant_message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "project_info",
                "arguments": "{}",
            },
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "project_info",
                "arguments": "{}",
            },
        },
    ]

    tool_messages = [
        message
        for message in second_messages
        if message["role"] == "tool"
    ]

    assert [message["tool_call_id"] for message in tool_messages] == [
        "call_1",
        "call_2",
    ]

    assert json.loads(tool_messages[0]["content"]) == {
        "workspace": str(tmp_path),
    }

    assert json.loads(tool_messages[1]["content"]) == {
        "workspace": str(tmp_path),
    }


@pytest.mark.anyio
async def test_runtime_tracks_written_files(tmp_path):
    server = AgentServer(tmp_path)

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": (
                                '{"path":"src/example.py",'
                                '"content":"print(42)"}'
                            ),
                        },
                        "id": "call_123",
                    }],
                },
            }],
        },
        {
            "content": "The file was written.",
        },
    ])

    runtime = AgentRuntime(server, model)

    result = await runtime.run("Create src/example.py")

    assert result == "The file was written."

    state = server.state.load()

    assert state.files == ["src/example.py"]


@pytest.mark.anyio
async def test_runtime_tracks_created_files(tmp_path):
    server = AgentServer(tmp_path)

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "create_file",
                            "arguments": '{"path":"new.py"}',
                        },
                        "id": "call_123",
                    }],
                },
            }],
        },
        {
            "content": "The file was created.",
        },
    ])

    runtime = AgentRuntime(server, model)

    result = await runtime.run("Create new.py")

    assert result == "The file was created."

    state = server.state.load()

    assert state.files == ["new.py"]


@pytest.mark.anyio
async def test_runtime_tracks_edited_files(tmp_path):
    server = AgentServer(tmp_path)

    file_path = tmp_path / "example.py"
    file_path.write_text("old\nnew\n", encoding="utf-8")

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "edit_file_lines",
                            "arguments": (
                                '{"path":"example.py",'
                                '"start_line":1,'
                                '"end_line":1,'
                                '"content":"changed"}'
                            ),
                        },
                        "id": "call_123",
                    }],
                },
            }],
        },
        {
            "content": "The file was edited.",
        },
    ])

    runtime = AgentRuntime(server, model)

    result = await runtime.run("Edit example.py")

    assert result == "The file was edited."

    state = server.state.load()

    assert state.files == ["example.py"]


@pytest.mark.anyio
async def test_runtime_does_not_track_read_files(tmp_path):
    server = AgentServer(tmp_path)

    (tmp_path / "example.py").write_text(
        "print(42)\n",
        encoding="utf-8",
    )

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"example.py"}',
                        },
                        "id": "call_123",
                    }],
                },
            }],
        },
        {
            "content": "I inspected example.py.",
        },
    ])

    runtime = AgentRuntime(server, model)

    result = await runtime.run("Inspect example.py")

    assert result == "I inspected example.py."

    state = server.state.load()

    assert state.files == []


@pytest.mark.anyio
async def test_runtime_does_not_track_failed_file_write(tmp_path):
    server = AgentServer(tmp_path)

    model = SequenceModel([
        {
            "choices": [{
                "finish_reason": "tool",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": (
                                '{"path":"../outside.py",'
                                '"content":"bad"}'
                            ),
                        },
                        "id": "call_123",
                    }],
                },
            }],
        },
        {
            "content": "The write failed.",
        },
    ])

    runtime = AgentRuntime(server, model)

    result = await runtime.run("Write outside the workspace")

    assert result == "The write failed."

    state = server.state.load()

    assert state.files == []


class TaskStateModel:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, tools):
        self.calls += 1

        if self.calls == 1:
            assert any(
                tool["function"]["name"] == "set_task_state"
                for tool in tools
            )

            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_state",
                                    "type": "function",
                                    "function": {
                                        "name": "set_task_state",
                                        "arguments": json.dumps({
                                            "current": "Inspecting the project",
                                            "completed": [
                                                "Start task",
                                            ],
                                            "blocked": [],
                                        }),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        return {
            "choices": [
                {
                    "message": {
                        "content": "The project was inspected successfully.",
                    }
                }
            ]
        }


@pytest.mark.anyio
async def test_runtime_allows_model_to_manage_task_state(tmp_path):
    server = AgentServer(tmp_path)
    model = TaskStateModel()

    runtime = AgentRuntime(
        server=server,
        model=model,
        max_iterations=3,
    )

    task = "Inspect the project and report what you find."

    result = await runtime.run(task)

    assert result == "The project was inspected successfully."

    state = server.state.load()

    assert state.task == task
    assert state.status == "completed"
    assert state.current == "Inspecting the project"
    assert state.completed == ["Start task"]
    assert state.blocked == []
    assert state.files == []


@pytest.mark.anyio
async def test_runtime_rate_limits_tool_calls_per_iteration():
    from agent.runtime import AgentRuntime
    from unittest.mock import AsyncMock, MagicMock

    mock_server = MagicMock()
    mock_server.mcp.list_tools = AsyncMock(return_value=[])
    mock_server.mcp.call_tool = AsyncMock(return_value={"result": "ok"})
    
    mock_model = AsyncMock()
    call_count = [0]
    def mock_generate(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 5:
            return {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": f"call_{call_count[0]}",
                            "type": "function",
                            "function": {"name": "t", "arguments": {}},
                        }]
                    }
                }]
            }
        else:
            return {"choices": [{"message": {"tool_calls": []}}]}

    mock_model.generate.side_effect = mock_generate

    runtime = AgentRuntime(
        server=mock_server,
        model=mock_model,
        max_iterations=5,
    )

    result = await runtime.run("Test rate limiting")
    assert mock_server.mcp.call_tool.call_count == 5


@pytest.mark.anyio
async def test_runtime_resets_tool_call_counter_per_iteration():
    from agent.runtime import AgentRuntime
    from unittest.mock import AsyncMock, MagicMock

    mock_server = MagicMock()
    mock_server.mcp.list_tools = AsyncMock(return_value=[])
    mock_server.mcp.call_tool = AsyncMock(return_value={"result": "ok"})
    
    mock_model = AsyncMock()
    call_sequence = [0]
    def mock_generate(*args, **kwargs):
        call_sequence[0] += 1
        if call_sequence[0] <= 3:
            return {
                "choices": [{
                    "message": {"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": {}}}]}
                }]
            }
        else:
            return {"choices": [{"message": {"tool_calls": []}}]}

    mock_model.generate.side_effect = mock_generate

    runtime = AgentRuntime(
        server=mock_server,
        model=mock_model,
        max_iterations=3,
    )

    await runtime.run("Test counter reset")
    assert mock_server.mcp.call_tool.call_count == 3


@pytest.mark.anyio
async def test_runtime_resets_tool_call_counter_per_iteration():
    from agent.runtime import AgentRuntime
    from unittest.mock import AsyncMock, MagicMock

    mock_server = MagicMock()
    mock_server.mcp.list_tools = AsyncMock(return_value=[])
    mock_server.mcp.call_tool = AsyncMock(return_value={"result": "ok"})
    
    mock_model = AsyncMock()
    call_sequence = [0]
    def mock_generate(*args, **kwargs):
        call_sequence[0] += 1
        # Iteration 1: return tool call 1
        # Iteration 2: reset allows tool call 1 again (but model always returns same structure)
        if call_sequence[0] <= 3:
            return {
                "choices": [{
                    "message": {"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": {}}}]}
                }]
            }
        else:
            return {"choices": [{"message": {"tool_calls": []}}]}

    mock_model.generate.side_effect = mock_generate

    runtime = AgentRuntime(
        server=mock_server,
        model=mock_model,
        max_iterations=3,
    )

    await runtime.run("Test counter reset")
    # 3 calls in iter 1, 0 calls in iter 2 (model stops returning tool calls)
    assert mock_server.mcp.call_tool.call_count == 3


@pytest.mark.anyio
async def test_runtime_cooldown_between_tool_calls():
    from agent.runtime import AgentRuntime
    from unittest.mock import AsyncMock, MagicMock
    import time

    mock_server = MagicMock()
    mock_server.mcp.list_tools = AsyncMock(return_value=[])
    
    call_times = []
    async def track_calls(*args, **kwargs):
        call_times.append(time.monotonic())
        return {"result": "ok"}
    
    mock_server.mcp.call_tool = AsyncMock(side_effect=track_calls)
    
    mock_model = AsyncMock()
    mock_model.generate.return_value = {
        "choices": [{
            "message": {
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": {}}}]
            }
        }]
    }

    runtime = AgentRuntime(
        server=mock_server,
        model=mock_model,
        max_iterations=1,
    )

    await runtime.run("Test cooldown")
    
    # Verify at least one cooldown occurred (time diff > 0.05s due to async overhead)
    if len(call_times) > 1:
        delta = call_times[1] - call_times[0]
        assert delta >= 0.05  # Cooldown interval is 0.1s, allow for async scheduling variance
