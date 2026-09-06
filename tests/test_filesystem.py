from pathlib import Path
import pytest
from agent.mcp.filesystem import Filesystem
from agent.mcp.workspace import Workspace


@pytest.fixture
def filesystem(tmp_path):
    return Filesystem(Workspace(tmp_path))

def test_create_and_read_file(filesystem):
    filesystem.create_file("test.txt")

    result = filesystem.read_file("test.txt")

    assert result == {}

def test_write_and_read_file(filesystem):
    filesystem.write_file("test.txt", "line 1\nline 2\nline 3\n")

    result = filesystem.read_file("test.txt")

    assert result == {
        1: "line 1",
        2: "line 2",
        3: "line 3",
    }

def test_read_file_range(filesystem):
    filesystem.write_file("test.txt", "line 1\nline 2\nline 3\nline 4\n")

    result = filesystem.read_file(
        "test.txt",
        start_line=2,
        end_line=3,
    )

    assert result == {
        2: "line 2",
        3: "line 3",
    }

def test_read_file_rejects_invalid_line_range(filesystem):
    filesystem.write_file("test.txt", "line 1\n")

    with pytest.raises(ValueError):
        filesystem.read_file(
            "test.txt",
            start_line=0,
            end_line=1,
        )

    with pytest.raises(ValueError):
        filesystem.read_file(
            "test.txt",
            start_line=2,
            end_line=1,
        )

def test_write_file_append(filesystem):
    filesystem.write_file("test.txt", "hello\n")
    filesystem.write_file("test.txt", "world\n", append=True)

    result = filesystem.read_file("test.txt")

    assert result == {
        1: "hello",
        2: "world",
    }

def test_create_file_rejects_existing_file(filesystem):
    filesystem.create_file("test.txt")

    with pytest.raises(FileExistsError):
        filesystem.create_file("test.txt")

def test_edit_file_lines(filesystem):
    filesystem.write_file(
        "test.txt",
        "line 1\nline 2\nline 3\nline 4\n",
    )

    filesystem.edit_file_lines(
        "test.txt",
        start_line=2,
        end_line=3,
        content="replacement\n",
    )

    assert filesystem.read_file("test.txt") == {
        1: "line 1",
        2: "replacement",
        3: "line 4",
    }

def test_edit_file_lines_can_insert(filesystem):
    filesystem.write_file(
        "test.txt",
        "line 1\nline 3\n",
    )

    filesystem.edit_file_lines(
        "test.txt",
        start_line=-1,
        end_line=1,
        content="line 2\n",
    )

    assert filesystem.read_file("test.txt") == {
        1: "line 1",
        2: "line 2",
        3: "line 3",
    }

def test_edit_file_lines_rejects_invalid_insert(filesystem):
    filesystem.write_file("test.txt", "line 1\n")

    with pytest.raises(ValueError):
        filesystem.edit_file_lines(
            "test.txt",
            start_line=-1,
            end_line=2,
            content="invalid\n",
        )

def test_create_folder(filesystem):
    result = filesystem.create_folder("src")

    assert result["created"] is True
    assert filesystem.list_dir_contents() == [
        {
            "name": "src",
            "path": "src",
            "type": "directory",
        }
    ]

def test_move_item(filesystem):
    filesystem.write_file("old.txt", "content")

    result = filesystem.move_item("old.txt", "new.txt")

    assert result["old_path"] == "old.txt"
    assert result["new_path"] == "new.txt"
    assert filesystem.read_file("new.txt") == {1: "content"}

def test_move_item_rejects_existing_destination(filesystem):
    filesystem.write_file("old.txt", "old")
    filesystem.write_file("new.txt", "new")

    with pytest.raises(FileExistsError):
        filesystem.move_item("old.txt", "new.txt")

def test_delete_file(filesystem):
    filesystem.write_file("test.txt", "content")

    result = filesystem.delete_item("test.txt")

    assert result["deleted"] is True

    with pytest.raises(FileNotFoundError):
        filesystem.read_file("test.txt")

def test_list_tree(filesystem):
    filesystem.write_file("src/main.py", "print('hello')\n")
    filesystem.write_file("README.md", "# Test\n")

    result = filesystem.list_tree()

    paths = {entry["path"] for entry in result}

    assert "src" in paths
    assert "src/main.py" in paths
    assert "README.md" in paths

def test_list_tree_ignores_git(filesystem):
    filesystem.write_file(".git/config", "test\n")
    filesystem.write_file("src/main.py", "test\n")

    result = filesystem.list_tree()

    paths = {entry["path"] for entry in result}

    assert ".git" not in paths
    assert ".git/config" not in paths
    assert "src/main.py" in paths
