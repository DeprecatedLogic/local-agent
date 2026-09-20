import argparse
import asyncio
import inspect
from pathlib import Path
from cysystemd import daemon
from agent.logging_config import setup_logging
from agent.mcp_.agent_server import AgentServer
from agent.model import LlamaCppClient
from agent.runtime import AgentRuntime
from agent.health import HealthMonitor


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local coding agent.",
    )

    parser.add_argument(
        "--workspace",
        required=True,
        help="Path to the project workspace.",
    )

    parser.add_argument(
        "--model-url",
        default="http://127.0.0.1:8080",
        help="Base URL of the llama-server instance.",
    )

    parser.add_argument(
        "--model",
        default="local-agent",
        help="Model identifier sent to llama-server.",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of agent iterations.",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate tool execution without modifying the workspace.",
    )
    
    parser.add_argument(
        "task",
        nargs="*",
        help="Task for the agent. If omitted, read one line interactively.",
    )

    return parser


def create_runtime(args: argparse.Namespace, dry_run: bool = False) -> AgentRuntime:
    server = AgentServer(args.workspace)
    
    if dry_run:
        # Automatically find all public methods that look like mutating filesystem tools
        fs = server.filesystem
        mutating_methods = [
            name for name, method in inspect.getmembers(fs, predicate=inspect.ismethod)
            if not name.startswith("_")
            and any(kw in name for kw in ("write", "edit", "create", "move", "delete", "remove"))
        ]

        for tool_name in mutating_methods:
            original_tool = getattr(fs, tool_name)
            # Capture the current value to avoid the late-binding bug
            setattr(
                fs,
                tool_name,
                lambda *a, original=original_tool, **k: {
                    "dry_run": True,
                    "original": str(original),
                },
            )

    model = LlamaCppClient(
        base_url=args.model_url,
        model=args.model,
    )

    return AgentRuntime(
        server=server,
        model=model,
        max_iterations=args.max_iterations,
    )


async def run_agent(
    runtime: AgentRuntime,
    task: str,
) -> str:
    return await runtime.run(task)


def notify_systemd(message: str) -> None:
    try:
        daemon.sd_notify(message)
    except (RuntimeError, OSError):
        pass

async def async_main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    setup_logging()

    health = HealthMonitor(
        state_path=Path(args.workspace) / ".agent" / "state.json",
        interval=10.0,
    )

    runtime = create_runtime(
        args,
        dry_run=args.dry_run,
    )

    task = " ".join(args.task).strip()

    if not task:
        try:
            task = input("> ").strip()
        except EOFError:
            parser.error("no task was provided")

    if not task:
        parser.error("task cannot be empty")

    notify_systemd("READY=1")

    await health.start()

    try:
        result = await run_agent(runtime, task)
        print(result)
    finally:
        await health.stop()
        notify_systemd("STOPPING=1")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nAgent interrupted.")


if __name__ == "__main__":
    main()
