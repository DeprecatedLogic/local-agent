import argparse

import pytest

from agent.cli import create_parser
from agent.cli import create_runtime
from agent.cli import run_agent


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


def test_parser_uses_models_toml_by_default():
    parser = create_parser()

    args = parser.parse_args([
        "--workspace",
        "/tmp/project",
    ])

    assert args.model_url is None
    assert args.model is None
    assert args.embedding_url is None
    assert args.embedding_model is None
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
    config = tmp_path / "config"

    args = parser.parse_args([
        "--workspace",
        str(tmp_path),
        "--config-dir",
        str(config),
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
    assert (config / "models.toml").is_file()


def test_create_runtime_loads_model_generation_from_config(tmp_path):
    parser = create_parser()
    config = tmp_path / "config"
    config.mkdir()
    (config / "models.toml").write_text(
        """
[defaults]
primary_backend = "main"
embedding_backend = "default"

[backends.main]
type = "chat"
url = "http://127.0.0.1:9100"
model = "configured-model"

[backends.main.generation]
temperature = 0.25
top_p = 0.88
top_k = 31
min_p = 0.03
repeat_penalty = 1.07
reasoning = "auto"
reasoning_budget = 7777

[backends.default]
type = "embedding"
url = "http://127.0.0.1:9101"
model = "configured-embedding"
""".strip() + "\n",
        encoding="utf-8",
    )

    args = parser.parse_args([
        "--workspace",
        str(tmp_path),
        "--config-dir",
        str(config),
    ])

    runtime = create_runtime(args)

    assert runtime.model.base_url == "http://127.0.0.1:9100"
    assert runtime.model.model == "configured-model"
    assert runtime.model.generation.temperature == 0.25
    assert runtime.model.generation.top_p == 0.88
    assert runtime.model.generation.top_k == 31
    assert runtime.model.generation.min_p == 0.03
    assert runtime.model.generation.repeat_penalty == 1.07
    assert runtime.model.generation.reasoning_budget == 7777
    assert runtime.server.chat_search.embedding_provider.base_url == (
        "http://127.0.0.1:9101"
    )
    assert runtime.server.chat_search.embedding_provider.model == (
        "configured-embedding"
    )


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
        "--config-dir",
        str(tmp_path / "config"),
        "--dry-run",
    ])

    runtime = create_runtime(args, dry_run=args.dry_run)

    assert runtime.server.workspace.root == tmp_path.resolve()

    result = runtime.server.filesystem.write_file("test.txt", "data")
    assert result.get("dry_run") is True
    assert "Filesystem.write_file" in result.get("original", "")


def test_create_runtime_without_dry_run_does_not_patch(tmp_path):
    parser = create_parser()

    args = parser.parse_args([
        "--workspace",
        str(tmp_path),
        "--config-dir",
        str(tmp_path / "config"),
    ])

    runtime = create_runtime(args, dry_run=args.dry_run)

    with pytest.raises(FileNotFoundError):
        runtime.server.filesystem.read_file("nonexistent.txt")


@pytest.mark.anyio
async def test_run_agent_with_dry_run_mode(tmp_path, monkeypatch):
    """Verify dry-run doesn't mutate filesystem during agent execution."""
    from unittest.mock import AsyncMock

    parser = create_parser()
    args = parser.parse_args([
        "--workspace",
        str(tmp_path),
        "--config-dir",
        str(tmp_path / "config"),
        "--dry-run",
        "--max-iterations",
        "2",
    ])

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

    assert not (tmp_path / "test.txt").exists()
    assert isinstance(result, str)
