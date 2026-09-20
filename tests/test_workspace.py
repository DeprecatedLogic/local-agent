from pathlib import Path
import pytest
from agent.mcp_.workspace import Workspace


def test_resolve_relative_path(tmp_path):
    workspace = Workspace(tmp_path)

    resolved = workspace.resolve("src/main.py")

    assert resolved == (tmp_path / "src/main.py").resolve()


def test_resolve_rejects_absolute_path(tmp_path):
    workspace = Workspace(tmp_path)

    with pytest.raises(ValueError, match="Absolute paths are not allowed"):
        workspace.resolve("/etc/passwd")


def test_resolve_rejects_path_escape(tmp_path):
    workspace = Workspace(tmp_path)

    with pytest.raises(ValueError, match="Path escapes workspace"):
        workspace.resolve("../outside.txt")


def test_resolve_rejects_symlink_escape(tmp_path):
    workspace = Workspace(tmp_path)

    outside = tmp_path.parent / "outside"
    outside.mkdir()

    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="Path escapes workspace"):
        workspace.resolve("link/file.txt")


def test_relative_returns_workspace_relative_path(tmp_path):
    workspace = Workspace(tmp_path)

    path = workspace.resolve("src/main.py")

    assert workspace.relative(path) == "src/main.py"


def test_relative_rejects_outside_path(tmp_path):
    workspace = Workspace(tmp_path)
    outside = tmp_path.parent / "outside.txt"

    outside.touch()

    with pytest.raises(ValueError, match="outside workspace"):
        workspace.relative(outside)
