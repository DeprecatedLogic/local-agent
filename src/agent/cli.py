import argparse
import asyncio
import inspect

from cysystemd import daemon

from agent.chat import ChatSession
from agent.health import HealthMonitor
from agent.logging_config import setup_logging
from agent.mcp_.agent_server import AgentServer
from agent.model import LlamaCppClient
from agent.runtime import AgentRuntime


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local agent.",
    )

    parser.add_argument(
        "--workspace",
        required=False,
        default="/srv/local-agent-workspace",
        help="Path to the project workspace.",
    )

    parser.add_argument(
        "--config-dir",
        default=None,
        help=(
            "Agent configuration directory. Defaults to "
            "$LOCAL_AGENT_CONFIG_DIR or ~/.config/local-agent."
        ),
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
        help=(
            "Maximum number of agent iterations "
            "per user turn."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Simulate tool execution without "
            "modifying the workspace."
        ),
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help=(
            "Start an interactive terminal chat "
            "session."
        ),
    )

    parser.add_argument(
        "task",
        nargs="*",
        help=(
            "Task for the agent. In chat mode, "
            "this becomes the optional first "
            "user message."
        ),
    )

    return parser


def create_runtime(
    args: argparse.Namespace,
    dry_run: bool = False,
) -> AgentRuntime:
    server = AgentServer(
        args.workspace,
        config_dir=args.config_dir,
    )

    if dry_run:
        fs = server.filesystem

        mutating_methods = [
            name
            for name, method
            in inspect.getmembers(
                fs,
                predicate=inspect.ismethod,
            )
            if not name.startswith("_")
            and any(
                keyword in name
                for keyword in (
                    "write",
                    "edit",
                    "create",
                    "move",
                    "delete",
                    "remove",
                )
            )
        ]

        for tool_name in mutating_methods:
            original_tool = getattr(
                fs,
                tool_name,
            )

            setattr(
                fs,
                tool_name,
                lambda *args,
                original=original_tool,
                **kwargs: {
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


async def run_chat(
    runtime: AgentRuntime,
    initial_message: str | None = None,
) -> None:
    session = ChatSession(runtime)

    await session.run(
        initial_message=initial_message,
    )


def _notify_systemd(
    message: str,
) -> None:
    try:
        daemon.sd_notify(message)
    except RuntimeError:
        # Running outside a systemd service.
        pass


async def async_main(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    runtime = create_runtime(
        args,
        dry_run=args.dry_run,
    )

    health = HealthMonitor(
        state_path=runtime.server.state.path,
        interval=10.0,
    )

    await health.start()

    _notify_systemd("READY=1")

    try:
        task = " ".join(
            args.task
        ).strip()

        if args.chat:
            await run_chat(
                runtime,
                initial_message=task or None,
            )
            return

        if not task:
            try:
                task = (
                    await asyncio.to_thread(
                        input,
                        "> ",
                    )
                ).strip()
            except EOFError:
                parser.error(
                    "no task was provided"
                )

        if not task:
            parser.error(
                "task cannot be empty"
            )

        result = await run_agent(
            runtime,
            task,
        )

        print(result)

    finally:
        # SQLite task writes are committed transactionally at mutation time;
        # there is no mutable in-memory state that needs a shutdown flush.
        await health.stop()
        _notify_systemd("STOPPING=1")


def main() -> None:
    setup_logging()

    parser = create_parser()
    args = parser.parse_args()

    try:
        asyncio.run(
            async_main(
                args,
                parser,
            )
        )
    except KeyboardInterrupt:
        print("\nAgent interrupted.")


if __name__ == "__main__":
    main()
