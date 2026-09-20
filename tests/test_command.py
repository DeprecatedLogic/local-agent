import asyncio

import pytest

from agent.command import CommandExecutor


@pytest.mark.anyio
async def test_runs_command_in_workspace(tmp_path):
    executor = CommandExecutor(tmp_path)

    result = await executor.run("pwd")

    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert result["stdout"].strip() == str(tmp_path.resolve())
    assert result["stderr"] == ""


@pytest.mark.anyio
async def test_captures_stdout(tmp_path):
    executor = CommandExecutor(tmp_path)

    result = await executor.run("printf 'hello world'")

    assert result["exit_code"] == 0
    assert result["stdout"] == "hello world"
    assert result["stderr"] == ""
    assert result["timed_out"] is False


@pytest.mark.anyio
async def test_captures_stderr(tmp_path):
    executor = CommandExecutor(tmp_path)

    result = await executor.run(
        "printf 'error message' >&2; exit 3"
    )

    assert result["exit_code"] == 3
    assert result["stdout"] == ""
    assert result["stderr"] == "error message"
    assert result["timed_out"] is False


@pytest.mark.anyio
async def test_captures_stdout_and_stderr(tmp_path):
    executor = CommandExecutor(tmp_path)

    result = await executor.run(
        "printf 'out'; printf 'err' >&2"
    )

    assert result["exit_code"] == 0
    assert result["stdout"] == "out"
    assert result["stderr"] == "err"
    assert result["timed_out"] is False


@pytest.mark.anyio
async def test_timeout_kills_command(tmp_path):
    executor = CommandExecutor(
        tmp_path,
        timeout=0.1,
    )

    result = await executor.run("sleep 10")

    assert result["timed_out"] is True
    assert result["exit_code"] is not None


@pytest.mark.anyio
async def test_timeout_kills_process_group(tmp_path):
    executor = CommandExecutor(
        tmp_path,
        timeout=0.1,
    )

    result = await executor.run(
        "sleep 10 & wait"
    )

    assert result["timed_out"] is True
    assert result["exit_code"] is not None


@pytest.mark.anyio
async def test_output_is_bounded(tmp_path):
    executor = CommandExecutor(
        tmp_path,
        max_output=100,
    )

    result = await executor.run(
        "python -c \"print('x' * 10000)\""
    )

    assert result["exit_code"] == 0
    assert len(result["stdout"]) <= 100


@pytest.mark.anyio
async def test_large_stderr_does_not_deadlock(tmp_path):
    executor = CommandExecutor(
        tmp_path,
        max_output=100,
    )

    result = await executor.run(
        "python -c \"import sys; sys.stderr.write('x' * 1000000)\""
    )

    assert result["exit_code"] == 0
    assert len(result["stderr"]) <= 100


def test_rejects_invalid_timeout(tmp_path):
    with pytest.raises(ValueError):
        CommandExecutor(tmp_path, timeout=0)


def test_rejects_invalid_output_limit(tmp_path):
    with pytest.raises(ValueError):
        CommandExecutor(tmp_path, max_output=0)


@pytest.mark.anyio
async def test_rejects_empty_command(tmp_path):
    executor = CommandExecutor(tmp_path)

    with pytest.raises(ValueError):
        await executor.run("")


@pytest.mark.anyio
async def test_preserves_command_in_result(tmp_path):
    executor = CommandExecutor(tmp_path)

    result = await executor.run("printf 'test'")

    assert result["command"] == "printf 'test'"

