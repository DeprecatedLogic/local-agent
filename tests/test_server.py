import pytest
import asyncio
import subprocess
from agent.mcp_.agent_server import AgentServer
from fastmcp.exceptions import ToolError, ValidationError


def test_server_registers_expected_tools(tmp_path):
    server = AgentServer(tmp_path)

    tools = asyncio.run(server.mcp.list_tools())
    names = {tool.name for tool in tools}

    expected = {
        "project_info",
        "get_task_state",
        "set_task_state",
        "read_file",
        "write_file",
        "edit_file_lines",
        "create_file",
        "create_folder",
        "list_dir_contents",
        "list_tree",
        "search_in_file",
        "search_files",
        "git_status",
        "git_diff",
        "git_log",
        "git_show",
        "delete_item",
        "move_item",
        "run_command"
    }

    assert names == expected


def test_project_info(tmp_path):
    server = AgentServer(tmp_path)

    result = asyncio.run(
        server.mcp.call_tool("project_info", {})
    )

    print(result)
    print(type(result))
    print(vars(result))
    
    assert not result.is_error
    assert result.structured_content["workspace"] == str(tmp_path.resolve())


def test_task_state_tools(tmp_path):
    server = AgentServer(tmp_path)

    state = server.state.load()
    state.task = "Test server"
    state.status = "in_progress"
    state.files = ["src/example.py"]
    state.current = "Initial work"
    state.completed = ["Inspect project"]
    state.blocked = ["Waiting for dependency"]
    server.state.save(state)

    result = asyncio.run(
        server.mcp.call_tool(
            "set_task_state",
            {
                "current": "Implementing task state",
                "completed": [
                    "Inspect project",
                    "Implement task state",
                ],
                "blocked": [],
            },
        )
    )

    assert not result.is_error
    assert result.structured_content["task"] == "Test server"
    assert result.structured_content["status"] == "in_progress"
    assert result.structured_content["current"] == "Implementing task state"
    assert result.structured_content["completed"] == [
        "Inspect project",
        "Implement task state",
    ]
    assert result.structured_content["blocked"] == []
    assert result.structured_content["files"] == ["src/example.py"]

    result = asyncio.run(
        server.mcp.call_tool("get_task_state", {})
    )

    assert result.structured_content["task"] == "Test server"
    assert result.structured_content["status"] == "in_progress"
    assert result.structured_content["current"] == "Implementing task state"
    assert result.structured_content["completed"] == [
        "Inspect project",
        "Implement task state",
    ]
    assert result.structured_content["blocked"] == []
    assert result.structured_content["files"] == ["src/example.py"]


def test_set_task_state_preserves_omitted_fields(tmp_path):
    server = AgentServer(tmp_path)

    state = server.state.load()
    state.task = "Original task"
    state.status = "in_progress"
    state.current = "Current work"
    state.completed = ["Step one"]
    state.blocked = ["Step two"]
    state.files = ["main.py"]
    server.state.save(state)

    result = asyncio.run(
        server.mcp.call_tool(
            "set_task_state",
            {
                "current": "Updated work",
            },
        )
    )

    assert not result.is_error
    assert result.structured_content == {
        "task": "Original task",
        "status": "in_progress",
        "current": "Updated work",
        "completed": ["Step one"],
        "blocked": ["Step two"],
        "files": ["main.py"],
    }


def test_set_task_state_replaces_completed_and_blocked(tmp_path):
    server = AgentServer(tmp_path)

    state = server.state.load()
    state.task = "Test task"
    state.status = "in_progress"
    state.current = "Working"
    state.completed = [
        "Old step one",
        "Old step two",
    ]
    state.blocked = [
        "Old blocker",
    ]
    state.files = ["test.py"]
    server.state.save(state)

    result = asyncio.run(
        server.mcp.call_tool(
            "set_task_state",
            {
                "completed": [
                    "Reorganized step one",
                    "Reorganized step two",
                    "Reorganized step three",
                ],
                "blocked": [
                    "New blocker",
                ],
            },
        )
    )

    assert not result.is_error
    assert result.structured_content["completed"] == [
        "Reorganized step one",
        "Reorganized step two",
        "Reorganized step three",
    ]
    assert result.structured_content["blocked"] == [
        "New blocker",
    ]


def test_set_task_state_rejects_runtime_owned_fields(tmp_path):
    server = AgentServer(tmp_path)

    state = server.state.load()
    state.task = "Original task"
    state.status = "in_progress"
    state.files = ["existing.py"]
    server.state.save(state)

    for field, value in (
        ("task", "Malicious replacement"),
        ("status", "completed"),
        ("files", ["malicious.py"]),
    ):
        with pytest.raises(ValidationError):
            asyncio.run(
                server.mcp.call_tool(
                    "set_task_state",
                    {field: value},
                )
            )

    state = server.state.load()

    assert state.task == "Original task"
    assert state.status == "in_progress"
    assert state.files == ["existing.py"]


def test_file_tools(tmp_path):
    server = AgentServer(tmp_path)

    result = asyncio.run(
        server.mcp.call_tool(
            "write_file",
            {
                "path": "test.txt",
                "content": "hello\nworld\n",
            },
        )
    )

    assert not result.is_error
    assert result.structured_content["path"] == "test.txt"

    result = asyncio.run(
        server.mcp.call_tool(
            "read_file",
            {
                "path": "test.txt",
            },
        )
    )

    assert not result.is_error
    assert result.structured_content == {
        "1": "hello",
        "2": "world",
    }


def test_mcp_workspace_isolation(tmp_path):
    server = AgentServer(tmp_path)

    with pytest.raises(ToolError, match="Path escapes workspace"):
        asyncio.run(
            server.mcp.call_tool(
                "write_file",
                {
                    "path": "../outside.txt",
                    "content": "should fail",
                },
            )
        )

    assert not (tmp_path.parent / "outside.txt").exists()


def test_search_tools(tmp_path):
    server = AgentServer(tmp_path)

    (tmp_path / "main.py").write_text(
        "def hello():\n"
        "    print('hello')\n"
        "    return 42\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        server.mcp.call_tool(
            "search_in_file",
            {
                "path": "main.py",
                "pattern": "hello",
            },
        )
    )

    assert result.structured_content == {
        "result": [
            {
                "path": "main.py",
                "line": 1,
                "text": "def hello():",
            },
            {
                "path": "main.py",
                "line": 2,
                "text": "    print('hello')",
            },
        ]
    }

    result = asyncio.run(
        server.mcp.call_tool(
            "search_files",
            {
                "pattern": "return",
            },
        )
    )

    assert result.structured_content == {
        "result": [
            {
                "path": "main.py",
                "line": 3,
                "text": "    return 42",
            },
        ]
    }


def test_search_tools_reject_workspace_escape(tmp_path):
    server = AgentServer(tmp_path)

    with pytest.raises(ToolError, match="Path escapes workspace"):
        asyncio.run(
            server.mcp.call_tool(
                "search_in_file",
                {
                    "path": "../outside.txt",
                    "pattern": "test",
                },
            )
        )


def test_git_status_tool(tmp_path):
    server = AgentServer(tmp_path)

    result = asyncio.run(
        server.mcp.call_tool("git_status", {})
    )

    assert result.structured_content["ok"] is False
    assert result.structured_content["repository"] is False
    assert result.structured_content["clean"] is None


def test_git_status_tool_repository(tmp_path):
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    server = AgentServer(tmp_path)

    result = asyncio.run(
        server.mcp.call_tool("git_status", {})
    )

    assert result.structured_content["ok"] is True
    assert result.structured_content["repository"] is True
    assert result.structured_content["clean"] is True


def test_git_diff_tool(tmp_path):
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    (tmp_path / "test.txt").write_text(
        "original\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["git", "add", "test.txt"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    (tmp_path / "test.txt").write_text(
        "modified\n",
        encoding="utf-8",
    )

    server = AgentServer(tmp_path)

    result = asyncio.run(
        server.mcp.call_tool(
            "git_diff",
            {},
        )
    )

    assert result.structured_content["ok"] is True
    assert result.structured_content["staged"] is False
    assert result.structured_content["path"] is None
    assert "original" in result.structured_content["diff"]
    assert "modified" in result.structured_content["diff"]


def test_git_diff_tool_path_filter(tmp_path):
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )

    for name, content in (
        ("first.txt", "first\n"),
        ("second.txt", "second\n"),
    ):
        (tmp_path / name).write_text(content, encoding="utf-8")

    subprocess.run(
        ["git", "add", "."],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    (tmp_path / "first.txt").write_text(
        "first modified\n",
        encoding="utf-8",
    )
    (tmp_path / "second.txt").write_text(
        "second modified\n",
        encoding="utf-8",
    )

    server = AgentServer(tmp_path)

    result = asyncio.run(
        server.mcp.call_tool(
            "git_diff",
            {"path": "first.txt"},
        )
    )

    assert result.structured_content["ok"] is True
    assert "first modified" in result.structured_content["diff"]
    assert "second modified" not in result.structured_content["diff"]


def test_git_log_tool(tmp_path):
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )

    (tmp_path / "test.txt").write_text(
        "hello\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["git", "add", "test.txt"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    server = AgentServer(tmp_path)

    result = asyncio.run(
        server.mcp.call_tool(
            "git_log",
            {"max_count": 1},
        )
    )

    assert result.structured_content["ok"] is True

    commits = result.structured_content["commits"]

    assert len(commits) == 1
    assert commits[0]["subject"] == "Initial commit"
    assert commits[0]["author"] == "Test User"
    assert commits[0]["hash"]
    assert commits[0]["short_hash"]
    assert commits[0]["date"]


def test_git_log_tool_rejects_invalid_count(tmp_path):
    server = AgentServer(tmp_path)

    with pytest.raises(ToolError, match="max_count must be"):
        asyncio.run(
            server.mcp.call_tool(
                "git_log",
                {"max_count": 0},
            )
        )


def test_git_show_tool(tmp_path):
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )

    (tmp_path / "test.txt").write_text(
        "hello\n",
        encoding="utf-8",
    )

    subprocess.run(
        ["git", "add", "test.txt"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    server = AgentServer(tmp_path)

    result = asyncio.run(
        server.mcp.call_tool(
            "git_show",
            {"revision": "HEAD"},
        )
    )

    assert result.structured_content["ok"] is True
    assert result.structured_content["revision"] == "HEAD"
    assert result.structured_content["path"] is None
    assert "Initial commit" in result.structured_content["content"]
    assert "hello" in result.structured_content["content"]


def test_git_show_tool_rejects_empty_revision(tmp_path):
    server = AgentServer(tmp_path)

    with pytest.raises(ToolError, match="revision cannot be empty"):
        asyncio.run(
            server.mcp.call_tool(
                "git_show",
                {"revision": ""},
            )
        )
