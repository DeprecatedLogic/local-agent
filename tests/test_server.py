import asyncio
import subprocess

import pytest
from fastmcp.exceptions import ToolError, ValidationError

from agent.mcp_.agent_server import AgentServer


def call(server: AgentServer, name: str, arguments: dict | None = None):
    return asyncio.run(server.mcp.call_tool(name, arguments or {}))


def init_git_repo(path):
    subprocess.run(
        ["git", "init"], cwd=path, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path, check=True,
    )


def commit_file(path, name="test.txt", content="hello\n"):
    (path / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", name], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=path, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def test_server_registers_expected_tools(tmp_path):
    server = AgentServer(tmp_path)
    names = {tool.name for tool in asyncio.run(server.mcp.list_tools())}

    assert names == {
        "project_info",
        "get_agent_context",
        "start_task",
        "resume_task",
        "get_task_state",
        "set_task_state",
        "finish_task",
        "list_tasks",
        "search_tasks",
        "get_task_history",
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
        "run_command",
    }


def test_project_info(tmp_path):
    result = call(AgentServer(tmp_path), "project_info")
    assert not result.is_error
    assert result.structured_content["workspace"] == str(tmp_path.resolve())


def test_agent_context_index(tmp_path):
    result = call(AgentServer(tmp_path), "get_agent_context")
    assert not result.is_error
    assert result.structured_content["identity"]["name"] == "Local Agent"
    assert "evidence" in result.structured_content["topics"]


def test_chat_has_no_implicit_task_state(tmp_path):
    server = AgentServer(tmp_path)
    result = call(server, "get_task_state")

    assert not result.is_error
    assert result.structured_content["active"] is False
    assert result.structured_content["task_id"] is None
    assert server.tasks.list_tasks() == []


def test_task_lifecycle_and_bounded_history_tools(tmp_path):
    server = AgentServer(tmp_path)

    started = call(
        server,
        "start_task",
        {
            "title": "Inspect project",
            "goal": "Inspect the project and report issues.",
        },
    ).structured_content
    task_id = started["task_id"]

    updated = call(
        server,
        "set_task_state",
        {
            "current": "Inspecting source",
            "completed": ["Inspect tree"],
            "blocked": [],
        },
    ).structured_content

    assert updated["task_id"] == task_id
    assert updated["revision"] == 2
    assert updated["current"] == "Inspecting source"

    history = call(
        server,
        "get_task_history",
        {"task_id": task_id, "limit": 1},
    ).structured_content

    assert history["task"]["id"] == task_id
    assert len(history["revisions"]) == 1
    assert history["revisions"][0]["revision"] == 2
    assert history["next_before_revision"] == 2

    finished = call(
        server,
        "finish_task",
        {"status": "completed"},
    ).structured_content
    assert finished["status"] == "completed"


def test_task_search_and_resume(tmp_path):
    server = AgentServer(tmp_path)
    first = server.start_task("Config loader", "Improve config loading")
    first.current = "Inspect parser"
    server.save_active_task_state(first)
    server.finish_active_task("blocked")

    search = call(
        server,
        "search_tasks",
        {"query": "parser", "limit": 5},
    ).structured_content["result"]
    assert search[0]["task_id"] == first.task_id
    assert "goal_preview" in search[0]
    assert "goal" not in search[0]

    resumed = call(
        server,
        "resume_task",
        {"task_id": first.task_id},
    ).structured_content
    assert resumed["status"] == "in_progress"


def test_set_task_state_requires_active_task(tmp_path):
    server = AgentServer(tmp_path)
    with pytest.raises(ToolError, match="No active task"):
        call(server, "set_task_state", {"current": "No task"})


def test_set_task_state_rejects_runtime_owned_fields(tmp_path):
    server = AgentServer(tmp_path)
    server.start_task("Test", "Test runtime-owned fields")

    for field, value in (
        ("task_id", "other"),
        ("title", "replacement"),
        ("goal", "replacement"),
        ("status", "completed"),
        ("files", ["malicious.py"]),
        ("revision", 999),
    ):
        with pytest.raises(ValidationError):
            call(server, "set_task_state", {field: value})


def test_file_tools_and_workspace_isolation(tmp_path):
    server = AgentServer(tmp_path)

    written = call(
        server,
        "write_file",
        {"path": "test.txt", "content": "hello\nworld\n"},
    )
    assert written.structured_content["path"] == "test.txt"

    read = call(server, "read_file", {"path": "test.txt"})
    assert read.structured_content == {"1": "hello", "2": "world"}

    with pytest.raises(ToolError, match="Path escapes workspace"):
        call(
            server,
            "write_file",
            {"path": "../outside.txt", "content": "no"},
        )


def test_search_tools(tmp_path):
    server = AgentServer(tmp_path)
    (tmp_path / "main.py").write_text(
        "def hello():\n    print('hello')\n    return 42\n",
        encoding="utf-8",
    )

    one = call(
        server,
        "search_in_file",
        {"path": "main.py", "pattern": "hello"},
    ).structured_content["result"]
    assert [item["line"] for item in one] == [1, 2]

    many = call(
        server,
        "search_files",
        {"pattern": "return"},
    ).structured_content["result"]
    assert many[0]["line"] == 3

    with pytest.raises(ToolError, match="Path escapes workspace"):
        call(
            server,
            "search_in_file",
            {"path": "../outside.txt", "pattern": "test"},
        )


def test_git_tools(tmp_path):
    server = AgentServer(tmp_path)
    status = call(server, "git_status").structured_content
    assert status["ok"] is False
    assert status["repository"] is False

    init_git_repo(tmp_path)
    commit_file(tmp_path)
    (tmp_path / "test.txt").write_text("modified\n", encoding="utf-8")
    server = AgentServer(tmp_path)

    status = call(server, "git_status").structured_content
    assert status["ok"] is True
    assert status["repository"] is True
    assert status["clean"] is False

    diff = call(server, "git_diff").structured_content
    assert "hello" in diff["diff"]
    assert "modified" in diff["diff"]

    log = call(server, "git_log", {"max_count": 1}).structured_content
    assert log["commits"][0]["subject"] == "Initial commit"
    assert log["commits"][0]["author"] == "Test User"

    shown = call(server, "git_show", {"revision": "HEAD"}).structured_content
    assert shown["ok"] is True
    assert "Initial commit" in shown["content"]
    assert "hello" in shown["content"]


def test_git_tool_validation(tmp_path):
    server = AgentServer(tmp_path)
    with pytest.raises(ToolError, match="max_count must be"):
        call(server, "git_log", {"max_count": 0})
    with pytest.raises(ToolError, match="revision cannot be empty"):
        call(server, "git_show", {"revision": ""})
