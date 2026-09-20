from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any


class CommandExecutor:
    """Execute shell commands inside a bounded workspace."""

    def __init__(
        self,
        workspace: str | Path,
        timeout: float = 30.0,
        max_output: int = 64 * 1024,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0.")

        if max_output < 1:
            raise ValueError("max_output must be at least 1.")

        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.max_output = max_output

    async def run(self, command: str) -> dict[str, Any]:
        if not command.strip():
            raise ValueError("command cannot be empty.")

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        stdout_task = asyncio.create_task(
            self._read_output(process.stdout)
        )
        stderr_task = asyncio.create_task(
            self._read_output(process.stderr)
        )

        timed_out = False

        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            self._kill_process_group(process.pid)
            await process.wait()

        stdout, stderr = await asyncio.gather(
            stdout_task,
            stderr_task,
        )

        return {
            "command": command,
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
        }

    async def _read_output(
        self,
        stream: asyncio.StreamReader | None,
    ) -> str:
        if stream is None:
            return ""

        chunks: list[bytes] = []
        total = 0

        while True:
            chunk = await stream.read(8192)

            if not chunk:
                break

            remaining = self.max_output - total

            if remaining > 0:
                accepted = chunk[:remaining]
                chunks.append(accepted)
                total += len(accepted)

        return b"".join(chunks).decode(
            "utf-8",
            errors="replace",
        )

    @staticmethod
    def _kill_process_group(pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

