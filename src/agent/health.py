import asyncio
import json
from pathlib import Path
from typing import Any
from cysystemd import daemon


class HealthMonitor:
    """Periodic heartbeat writer for systemd watchdog + external status."""

    def __init__(self, state_path: Path, interval: float = 10.0):
        self.state_path = state_path
        self.interval = interval
        self._running = False
        self._last_heartbeat: dict[str, Any] = {}

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        self._running = False

    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                self._last_heartbeat.update({
                    "status": "alive",
                    "timestamp": asyncio.get_running_loop().time(),
                })
            except Exception:
                pass
            
            try:
                daemon.notify("WATCHDOG=1")
            except Exception:
                pass

            try:
                health_file = self.state_path.with_suffix(".health.json")
                with health_file.open("w") as f:
                    json.dump(self._last_heartbeat, f, indent=2)
            except Exception:
                pass
            
            await asyncio.sleep(self.interval)
