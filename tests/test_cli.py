import argparse
import pytest
from agent.cli import create_parser
from agent.cli import create_runtime
from agent.cli import run_agent


def test_parser_requires_workspace():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_accepts_task():
    parser = create_parser()

    args = parser.parse_args([
        "--workspace",
        "/tmp/project",
        "Add",
        "a",
        "README",
    ])

    assert args.workspace == "/tmp/project"
    assert args.task == ["Add", "a", "README"]


def test_parser_has_agent_defaults():
    parser = create_parser()

    args = parser.parse_args([
        "--workspace",
        "/tmp/project",
    ])

    assert args.model_url == "http://127.0.0.1:8080"
    assert args.model == "local-agent"
    assert args.max_iterations == 50
    assert args.task == []


def test_parser_accepts_runtime_configuration():
    parser = create_parser()

    args = parser.parse_args([
        "--workspace",
        "/tmp/project",
        "--model-url",
        "http://localhost:9000",
        "--model",
        "test-model",
        "--max-iterations",
        "12",
        "Inspect",
        "the",
        "project",
    ])

    assert args.model_url == "http://localhost:9000"
    assert args.model == "test-model"
    assert args.max_iterations == 12
    assert args.task == ["Inspect", "the", "project"]


def test_create_runtime_constructs_agent_components(tmp_path):
    parser = create_parser()

    args = parser.parse_args([
        "--workspace",
        str(tmp_path),
        "--model-url",
        "http://127.0.0.1:8080",
        "--model",
        "test-model",
        "--max-iterations",
        "7",
    ])

    runtime = create_runtime(args)

    assert runtime.server.workspace.root == tmp_path.resolve()
    assert runtime.model.base_url == "http://127.0.0.1:8080"
    assert runtime.model.model == "test-model"
    assert runtime.max_iterations == 7


class FakeRuntime:
    def __init__(self):
        self.tasks = []

    async def run(self, task: str) -> str:
        self.tasks.append(task)
        return "agent response"


@pytest.mark.anyio
async def test_run_agent_delegates_to_runtime():
    runtime = FakeRuntime()

    result = await run_agent(runtime, "Inspect the project")

    assert result == "agent response"
    assert runtime.tasks == ["Inspect the project"]

def test_parser_accepts_dry_run_flag():
    parser = create_parser()

    args = parser.parse_args([
        "--workspace",
        "/tmp/project",
        "--dry-run",
        "Do something",
    ])

    assert args.dry_run is True


def test_parser_defaults_dry_run_to_false():
    parser = create_parser()

    args = parser.parse_args([
        "--workspace",
        "/tmp/project",
        "Some task",
    ])

    assert args.dry_run is False


def test_create_runtime_passes_dry_run_flag(tmp_path):
    parser = create_parser()

    args = parser.parse_args([
        "--workspace",
        str(tmp_path),
        "--dry-run",
    ])

    runtime = create_runtime(args, dry_run=args.dry_run)

    # Verify workspace is set correctly
    assert runtime.server.workspace.root == tmp_path.resolve()
    
    # Verify dry-run patched the filesystem tools
    result = runtime.server.filesystem.write_file("test.txt", "data")
    assert result.get("dry_run") is True
    assert "Filesystem.write_file" in result.get("original", "")


def test_create_runtime_without_dry_run_does_not_patch(tmp_path):
    parser = create_parser()

    args = parser.parse_args([
        "--workspace",
        str(tmp_path),
    ])

    runtime = create_runtime(args, dry_run=args.dry_run)

    # Normal behavior: should return actual file operation result (or raise if path invalid)
    with pytest.raises(FileNotFoundError):
        runtime.server.filesystem.read_file("nonexistent.txt")


@pytest.mark.anyio
async def test_run_agent_with_dry_run_mode(tmp_path, monkeypatch):
    """Verify dry-run doesn't mutate filesystem during agent execution."""
    from agent.cli import create_runtime, run_agent
    from unittest.mock import AsyncMock

    parser = create_parser()
    args = parser.parse_args([
        "--workspace",
        str(tmp_path),
        "--dry-run",
        "--max-iterations",
        "2",
    ])

    # Mock the model client to return a tool call that would normally write a file
    mock_model = AsyncMock()
    mock_model.generate.return_value = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": {"path": "test.txt", "content": "mocked"},
                    },
                }]
            }
        }]
    }

    runtime = create_runtime(args, dry_run=True)
    runtime.model = mock_model

    result = await run_agent(runtime, "Create test.txt")

    # Verify no actual file was created on disk
    assert not (tmp_path / "test.txt").exists()
    
    # Verify agent returned a response (not crashed)
    assert isinstance(result, str)
