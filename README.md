# Local Agent

A local coding agent built around `llama.cpp`. It can inspect and modify projects, execute commands, interact with Git, run tests, and keep track of its current task through a controlled set of tools.

The project is primarily an experiment in building a capable coding agent around local models while keeping inference, tools, state, and execution under local control.

## Features

* Local inference through an OpenAI-compatible `llama.cpp` server
* Autonomous tool-calling loop
* Filesystem and project-wide search tools
* Git inspection tools
* Sandboxed command execution
* Persistent task state
* Token usage telemetry
* Workspace path isolation
* Health monitoring and systemd watchdog support
* Dedicated Linux sandbox environment
* Unit and live `llama.cpp` integration tests

## How it works

The agent consists of three main parts:

**Runtime:** manages the conversation with the model, executes requested tool calls, tracks token usage, and continues the agent loop until the task finishes.

**Project tools:** give the model controlled access to the workspace. They provide filesystem operations, search, Git inspection, command execution, and task-state management.

**Sandbox:** optionally runs the agent under a dedicated Linux user with systemd hardening and restricted access to the host system.

The model itself is served separately through `llama.cpp`, so the agent is not tied to a specific model.

## Requirements

* Linux
* Python 3.14+
* `llama.cpp` with an OpenAI-compatible server
* A compatible local language model

Python dependencies are defined in `pyproject.toml`.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/DeprecatedLogic/local-agent
cd local-agent

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev]"
```

## Usage

Start a compatible `llama.cpp` server, then run:

```bash
local-agent --workspace /path/to/project
```

The agent accepts a task and can autonomously inspect and modify the configured workspace using its available tools.

Run:

```bash
local-agent --help
```

to see the available model and runtime options.

## Tools

The agent currently has tools for:

* reading, creating, editing, moving, and deleting files
* listing directories and project trees
* searching files and project contents
* inspecting Git status, diffs, logs, and revisions
* executing shell commands
* reading and updating persistent task state

Filesystem operations are restricted to the configured workspace.

## Task state

Task progress is persisted under `.agent/` rather than existing only inside the model's context.

The state keeps track of the current task, its status, progress, completed work, blockers, and files changed by the agent.

This is intended to provide a small amount of durable working memory and will eventually form part of the agent's context recovery system.

## Sandboxing

`scripts/local-agent-sandbox.sh` can be used to create an isolated environment for the agent with a dedicated Linux user and workspace.

The sandbox uses systemd hardening, filesystem restrictions, resource limits, and restricted privileges to reduce the amount of access given to agent-executed commands.

It is an additional isolation layer, not a guarantee of complete security.

## Testing

Run the unit tests with:

```bash
pytest tests/test_*.py
```

The unit test suite covers the runtime, model client, CLI, state management, command execution, workspace restrictions, filesystem operations, search, Git integration, health monitoring, MCP server, and usage telemetry.

Live tests against a running `llama.cpp` server are also available:

```bash
pytest tests/integration/test_llama_server.py
```

These integration tests require the model server to be running. Some tests exercise model behavior, including instruction following and tool-call generation, so their results can vary depending on the model being used.  
A model that does not reliably follow the expected prompts or tool-calling format may fail individual integration tests even when the underlying agent infrastructure is functioning correctly.

## Roadmap

The project is still under active development. Some of the next areas of work are:

* persistent interactive sessions
* task history and long-term memory
* context-budget awareness
* context compaction and recovery
* structured event logging
* model request recovery
* stronger task-completion verification

The goal is to progressively make the agent capable of handling longer autonomous development tasks without depending on cloud inference or giving the model unrestricted access to the host system.

## License

See [LICENSE](LICENSE).
