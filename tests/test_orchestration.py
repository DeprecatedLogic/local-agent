from pathlib import Path

import pytest

from agent.capabilities import ToolPolicy
from agent.mcp_.agent_server import AgentServer
from agent.orchestration import (
    AgentRegistry,
    DelegationLimits,
    DelegationManager,
    build_orchestrator_system_suffix,
)


class RecordingModel:
    def __init__(self, responses=None):
        self.responses = list(responses or [{"content": "Specialist complete."}])
        self.calls = []
        self.reasoning_budgets = []

    async def generate(self, messages, tools, on_stream=None):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "on_stream": on_stream,
        })
        if self.responses:
            return self.responses.pop(0)
        return {"content": "Specialist complete."}

    def with_reasoning_budget(self, budget):
        self.reasoning_budgets.append(budget)
        return self


def registry_for(tmp_path: Path) -> AgentRegistry:
    return AgentRegistry.from_config_dir(tmp_path / "config")


def test_registry_auto_seeds_editable_agents_toml(tmp_path):
    registry = registry_for(tmp_path)

    path = tmp_path / "config" / "agents.toml"
    assert path.is_file()
    assert registry.source == path.resolve()
    assert registry.ids() == ("coder", "docs", "reviewer", "tester")


def test_registry_loads_custom_agent_without_python_changes(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "agents.toml").write_text(
        '''
[tool_groups]
observe = ["project_info", "read_file"]

[agents.security]
name = "Security Auditor"
description = "Looks for security-sensitive mistakes."
instructions = "Inspect evidence and report concrete risks."
backend = "primary"
tool_groups = ["observe"]
tools = ["git_diff"]
max_iterations = 3
max_tool_calls = 7
max_context_tokens = 16384
max_reasoning_tokens = 2048
timeout_seconds = 45
max_result_chars = 2000

[agents.disabled]
enabled = false
name = "Disabled"
description = "Not available."
instructions = "Do nothing."
tools = []
''',
        encoding="utf-8",
    )

    registry = AgentRegistry.from_config_dir(config)
    assert registry.ids() == ("security",)

    spec = registry.get("security")
    assert spec.allowed_tools == frozenset({
        "project_info",
        "read_file",
        "git_diff",
    })
    assert spec.max_iterations == 3
    assert spec.max_tool_calls == 7
    assert spec.max_context_tokens == 16384
    assert spec.max_reasoning_tokens == 2048
    assert spec.timeout_seconds == 45.0
    assert spec.max_result_chars == 2000


def test_registry_rejects_unknown_tool_group(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "agents.toml").write_text(
        '''
[agents.custom]
name = "Custom"
description = "Custom agent."
instructions = "Work carefully."
tool_groups = ["missing"]
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown tool group"):
        AgentRegistry.from_config_dir(config)


def test_system_prompt_catalog_is_dynamic(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "agents.toml").write_text(
        '''
[agents.researcher]
name = "Researcher"
description = "Investigates a focused question."
instructions = "Gather evidence."
tools = ["read_file"]
''',
        encoding="utf-8",
    )

    registry = AgentRegistry.from_config_dir(config)
    suffix = build_orchestrator_system_suffix(registry)

    assert "researcher" in suffix
    assert "Investigates a focused question." in suffix
    assert "coder" not in suffix


@pytest.mark.anyio
async def test_default_specialists_have_bounded_capabilities(tmp_path):
    server = AgentServer(tmp_path)
    model = RecordingModel()
    manager = DelegationManager(
        server,
        {"primary": model},
        registry=registry_for(tmp_path),
    )

    result = await manager.delegate("reviewer", "Review the current project.")

    assert result["status"] == "completed"
    tools = {
        tool["function"]["name"]
        for tool in model.calls[0]["tools"]
    }
    assert "read_file" in tools
    assert "git_diff" in tools
    assert "write_file" not in tools
    assert "run_command" not in tools
    assert "delegate_task" not in tools
    assert "list_agents" not in tools


@pytest.mark.anyio
async def test_custom_agent_is_immediately_delegatable(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "agents.toml").write_text(
        '''
[agents.architect]
name = "Architect"
description = "Examines architecture."
instructions = "Read the project and explain tradeoffs."
backend = "primary"
tools = ["project_info", "list_tree", "read_file"]
max_context_tokens = 12000
max_reasoning_tokens = 1234
''',
        encoding="utf-8",
    )

    server = AgentServer(tmp_path)
    model = RecordingModel()
    manager = DelegationManager(
        server,
        {"primary": model},
        registry=AgentRegistry.from_config_dir(config),
    )

    result = await manager.delegate("architect", "Inspect the architecture.")

    assert result["agent"] == "architect"
    assert model.reasoning_budgets == [1234]
    tools = {tool["function"]["name"] for tool in model.calls[0]["tools"]}
    assert tools == {"project_info", "list_tree", "read_file"}


@pytest.mark.anyio
async def test_worker_rejects_hallucinated_delegation_call(tmp_path):
    server = AgentServer(tmp_path)
    model = RecordingModel([
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_delegate",
                        "type": "function",
                        "function": {
                            "name": "delegate_task",
                            "arguments": (
                                '{"agent":"coder","task":"delegate again"}'
                            ),
                        },
                    }],
                },
            }],
        },
        {"content": "I cannot delegate further."},
    ])
    manager = DelegationManager(
        server,
        {"primary": model},
        registry=registry_for(tmp_path),
    )

    result = await manager.delegate("coder", "Try to delegate again.")

    assert result["status"] == "completed"
    second_messages = model.calls[1]["messages"]
    tool_message = next(
        message for message in second_messages
        if message["role"] == "tool"
    )
    assert "not permitted" in tool_message["content"]


@pytest.mark.anyio
async def test_runtime_owned_worker_tools_cannot_be_granted(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "agents.toml").write_text(
        '''
[agents.powerful]
name = "Powerful"
description = "Requests protected tools."
instructions = "Try things."
tools = ["read_file", "delegate_task", "finish_task", "delete_chat_session"]
''',
        encoding="utf-8",
    )

    server = AgentServer(tmp_path)
    model = RecordingModel()
    manager = DelegationManager(
        server,
        {"primary": model},
        registry=AgentRegistry.from_config_dir(config),
    )

    await manager.delegate("powerful", "Inspect one file.")
    tools = {tool["function"]["name"] for tool in model.calls[0]["tools"]}

    assert "read_file" in tools
    assert "delegate_task" not in tools
    assert "finish_task" not in tools
    assert "delete_chat_session" not in tools


@pytest.mark.anyio
async def test_unknown_configured_tool_is_reported_at_delegation(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "agents.toml").write_text(
        '''
[agents.typo]
name = "Typo"
description = "Has a typo."
instructions = "Try to work."
tools = ["read_flie"]
''',
        encoding="utf-8",
    )

    server = AgentServer(tmp_path)
    manager = DelegationManager(
        server,
        {"primary": RecordingModel()},
        registry=AgentRegistry.from_config_dir(config),
    )

    with pytest.raises(ValueError, match="unknown tool.*read_flie"):
        await manager.delegate("typo", "Read something.")


@pytest.mark.anyio
async def test_delegation_limit_resets_per_turn(tmp_path):
    server = AgentServer(tmp_path)
    model = RecordingModel([
        {"content": "one"},
        {"content": "two"},
    ])
    manager = DelegationManager(
        server,
        {"primary": model},
        registry=registry_for(tmp_path),
        limits=DelegationLimits(max_delegations_per_turn=1),
    )

    await manager.delegate("reviewer", "First review.")
    with pytest.raises(RuntimeError, match="Delegation limit reached"):
        await manager.delegate("reviewer", "Second review.")

    manager.reset_turn()
    result = await manager.delegate("reviewer", "Second review.")
    assert result["status"] == "completed"


@pytest.mark.anyio
async def test_worker_iteration_limit_does_not_block_parent_task(tmp_path):
    server = AgentServer(tmp_path)
    parent = server.start_task("Parent", "Keep parent task active")
    model = RecordingModel([
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_info",
                        "type": "function",
                        "function": {
                            "name": "project_info",
                            "arguments": "{}",
                        },
                    }],
                },
            }],
        },
        {"content": "I ran out of worker iterations."},
    ])
    manager = DelegationManager(
        server,
        {"primary": model},
        registry=registry_for(tmp_path),
        limits=DelegationLimits(worker_max_iterations=1),
    )

    result = await manager.delegate("reviewer", "Inspect the project.")

    assert result["status"] == "blocked"
    state = server.tasks.get_state(parent.task_id)
    assert state.status == "in_progress"
    assert "Maximum iteration limit reached before task completion." not in state.blocked


def test_registry_seeds_global_token_defaults(tmp_path):
    registry = registry_for(tmp_path)
    path = tmp_path / "config" / "agents.toml"
    config_text = path.read_text(encoding="utf-8")

    assert "[defaults]" in config_text
    assert "max_context_tokens = 32768" in config_text
    assert "max_reasoning_tokens = 4096" in config_text

    spec = registry.get("coder")
    assert spec.max_context_tokens == 32768
    assert spec.max_reasoning_tokens == 4096


def test_registry_applies_global_and_per_agent_token_budgets(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "agents.toml").write_text(
        """
[defaults]
max_context_tokens = 24576
max_reasoning_tokens = 3072

[agents.defaulted]
name = "Defaulted"
description = "Uses global token budgets."
instructions = "Inspect carefully."
tools = ["read_file"]

[agents.custom]
name = "Custom"
description = "Overrides both token budgets."
instructions = "Inspect deeply."
tools = ["read_file"]
max_context_tokens = 8192
max_reasoning_tokens = 1024
""",
        encoding="utf-8",
    )

    registry = AgentRegistry.from_config_dir(config)
    defaulted = registry.get("defaulted")
    custom = registry.get("custom")

    assert defaulted.max_context_tokens == 24576
    assert defaulted.max_reasoning_tokens == 3072
    assert custom.max_context_tokens == 8192
    assert custom.max_reasoning_tokens == 1024

    summary = next(
        item for item in registry.summaries()
        if item["id"] == "custom"
    )
    assert summary["max_context_tokens"] == 8192
    assert summary["max_reasoning_tokens"] == 1024


def test_registry_keeps_legacy_reasoning_budget_compatible(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "agents.toml").write_text(
        """
[agents.legacy]
name = "Legacy"
description = "Uses the previous key."
instructions = "Work carefully."
tools = ["read_file"]
reasoning_budget = 2048
""",
        encoding="utf-8",
    )

    spec = AgentRegistry.from_config_dir(config).get("legacy")
    assert spec.max_context_tokens == 32768
    assert spec.max_reasoning_tokens == 2048


def test_registry_rejects_reasoning_budget_larger_than_context(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "agents.toml").write_text(
        """
[agents.invalid]
name = "Invalid"
description = "Has impossible token limits."
instructions = "Work."
tools = ["read_file"]
max_context_tokens = 1024
max_reasoning_tokens = 2048
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_reasoning_tokens cannot exceed"):
        AgentRegistry.from_config_dir(config)


@pytest.mark.anyio
async def test_delegation_propagates_per_agent_reasoning_budget(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "agents.toml").write_text(
        """
[agents.focused]
name = "Focused"
description = "Uses explicit token budgets."
instructions = "Inspect the project."
tools = ["project_info"]
max_context_tokens = 10000
max_reasoning_tokens = 1500
""",
        encoding="utf-8",
    )

    server = AgentServer(tmp_path)
    model = RecordingModel()
    manager = DelegationManager(
        server,
        {"primary": model},
        registry=AgentRegistry.from_config_dir(config),
    )

    result = await manager.delegate("focused", "Inspect the project.")

    assert result["status"] == "completed"
    assert model.reasoning_budgets == [1500]
    system_prompt = model.calls[0]["messages"][0]["content"]
    assert "Context token budget: 10000" in system_prompt
    assert "Reasoning token budget: 1500" in system_prompt


def test_tool_policy_denied_wins():
    policy = ToolPolicy(
        allowed=frozenset({"read_file", "delegate_task"}),
        denied=frozenset({"delegate_task"}),
    )
    assert policy.allows("read_file") is True
    assert policy.allows("delegate_task") is False
    assert policy.allows("write_file") is False
