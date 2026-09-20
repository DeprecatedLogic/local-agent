import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest


@pytest.mark.anyio
async def test_health_monitor_writes_status_file(tmp_path):
    from agent.health import HealthMonitor

    health = HealthMonitor(
        state_path=tmp_path / "state.json",
        interval=0.05,  # Faster for testing
    )
    
    await health.start()
    await asyncio.sleep(0.15)  # Wait for at least one cycle
    await health.stop()

    health_file = tmp_path / "state.health.json"
    assert health_file.exists(), f"Health file not found. Files in tmp_path: {list(tmp_path.iterdir())}"
    
    data = json.loads(health_file.read_text())
    assert data["status"] == "alive"
    assert "timestamp" in data


@pytest.mark.anyio
async def test_health_monitor_notifies_systemd(tmp_path):
    from agent.health import HealthMonitor
    from unittest.mock import patch

    with patch("cysystemd.daemon.notify") as mock_notify:
        health = HealthMonitor(
            state_path=tmp_path / "state.json",
            interval=0.1,
        )
        
        await health.start()
        await asyncio.sleep(0.2)
        await health.stop()

        assert mock_notify.called
        calls = [call[0][0] for call in mock_notify.call_args_list]
        assert "WATCHDOG=1" in calls


@pytest.mark.anyio
async def test_health_monitor_handles_errors_gracefully(tmp_path):
    from agent.health import HealthMonitor

    health = HealthMonitor(
        state_path=Path("/proc/nonexistent/health.json"),
        interval=0.1,
    )
    
    await health.start()
    await asyncio.sleep(0.2)
    await health.stop()

    assert not health._running
