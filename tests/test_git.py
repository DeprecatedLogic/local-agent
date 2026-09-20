import subprocess
import pytest
from agent.mcp_.git import Git
from agent.mcp_.workspace import Workspace


def run_git(path, *args):
    subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture
def git_repo(tmp_path):
    run_git(tmp_path, "init")
    run_git(tmp_path, "config", "user.name", "Test User")
    run_git(tmp_path, "config", "user.email", "test@example.com")

    return Git(Workspace(tmp_path))


def test_status_non_repository(tmp_path):
    git = Git(Workspace(tmp_path))

    result = git.status()

    assert result["ok"] is False
    assert result["repository"] is False
    assert result["clean"] is None
    assert result["staged"] == []
    assert result["unstaged"] == []
    assert result["untracked"] == []


def test_status_clean_repository(git_repo):
    result = git_repo.status()

    assert result["ok"] is True
    assert result["repository"] is True
    assert result["clean"] is True
    assert result["staged"] == []
    assert result["unstaged"] == []
    assert result["untracked"] == []
    assert result["conflicted"] == []


def test_status_untracked_file(git_repo):
    path = git_repo.workspace.resolve("test.txt")
    path.write_text("hello\n", encoding="utf-8")

    result = git_repo.status()

    assert result["ok"] is True
    assert result["clean"] is False
    assert result["untracked"] == [
        {"path": "test.txt"}
    ]


def test_status_staged_file(git_repo):
    path = git_repo.workspace.resolve("test.txt")
    path.write_text("hello\n", encoding="utf-8")

    run_git(git_repo.workspace.root, "add", "test.txt")

    result = git_repo.status()

    assert result["ok"] is True
    assert result["clean"] is False
    assert len(result["staged"]) == 1
    assert result["staged"][0]["path"] == "test.txt"
    assert result["staged"][0]["index_status"] == "A"


def test_status_unstaged_file(git_repo):
    path = git_repo.workspace.resolve("test.txt")
    path.write_text("hello\n", encoding="utf-8")

    run_git(git_repo.workspace.root, "add", "test.txt")
    run_git(git_repo.workspace.root, "commit", "-m", "Initial commit")

    path.write_text("modified\n", encoding="utf-8")

    result = git_repo.status()

    assert result["ok"] is True
    assert result["clean"] is False
    assert len(result["unstaged"]) == 1
    assert result["unstaged"][0]["path"] == "test.txt"


def test_current_branch(git_repo):
    run_git(git_repo.workspace.root, "checkout", "-b", "feature/test")

    assert git_repo.current_branch() == "feature/test"


def test_diff_unstaged(git_repo):
    path = git_repo.workspace.resolve("test.txt")
    path.write_text("original\n", encoding="utf-8")

    run_git(git_repo.workspace.root, "add", "test.txt")
    run_git(git_repo.workspace.root, "commit", "-m", "Initial commit")

    path.write_text("modified\n", encoding="utf-8")

    result = git_repo.diff()

    assert result["ok"] is True
    assert result["staged"] is False
    assert "modified" in result["diff"]
    assert "original" in result["diff"]


def test_diff_staged(git_repo):
    path = git_repo.workspace.resolve("test.txt")
    path.write_text("original\n", encoding="utf-8")

    run_git(git_repo.workspace.root, "add", "test.txt")
    run_git(git_repo.workspace.root, "commit", "-m", "Initial commit")

    path.write_text("modified\n", encoding="utf-8")
    run_git(git_repo.workspace.root, "add", "test.txt")

    result = git_repo.diff(staged=True)

    assert result["ok"] is True
    assert result["staged"] is True
    assert "modified" in result["diff"]
    assert "original" in result["diff"]


def test_diff_path_filter(git_repo):
    first = git_repo.workspace.resolve("first.txt")
    second = git_repo.workspace.resolve("second.txt")

    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    run_git(git_repo.workspace.root, "add", ".")
    run_git(git_repo.workspace.root, "commit", "-m", "Initial commit")

    first.write_text("first modified\n", encoding="utf-8")
    second.write_text("second modified\n", encoding="utf-8")

    result = git_repo.diff(path="first.txt")

    assert result["ok"] is True
    assert "first modified" in result["diff"]
    assert "second modified" not in result["diff"]


def test_log(git_repo):
    path = git_repo.workspace.resolve("test.txt")
    path.write_text("hello\n", encoding="utf-8")

    run_git(git_repo.workspace.root, "add", "test.txt")
    run_git(git_repo.workspace.root, "commit", "-m", "Initial commit")

    path.write_text("world\n", encoding="utf-8")
    run_git(git_repo.workspace.root, "add", "test.txt")
    run_git(git_repo.workspace.root, "commit", "-m", "Second commit")

    result = git_repo.log(max_count=2)

    assert result["ok"] is True
    assert len(result["commits"]) == 2

    assert result["commits"][0]["subject"] == "Second commit"
    assert result["commits"][1]["subject"] == "Initial commit"

    for commit in result["commits"]:
        assert commit["hash"]
        assert commit["short_hash"]
        assert commit["author"] == "Test User"
        assert commit["date"]


def test_log_rejects_invalid_count(git_repo):
    with pytest.raises(ValueError):
        git_repo.log(max_count=0)

    with pytest.raises(ValueError):
        git_repo.log(max_count=101)


def test_show_commit(git_repo):
    path = git_repo.workspace.resolve("test.txt")
    path.write_text("hello\n", encoding="utf-8")

    run_git(git_repo.workspace.root, "add", "test.txt")
    run_git(git_repo.workspace.root, "commit", "-m", "Initial commit")

    result = git_repo.show("HEAD")

    assert result["ok"] is True
    assert result["revision"] == "HEAD"
    assert "Initial commit" in result["content"]
    assert "hello" in result["content"]


def test_show_commit_path_filter(git_repo):
    first = git_repo.workspace.resolve("first.txt")
    second = git_repo.workspace.resolve("second.txt")

    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    run_git(git_repo.workspace.root, "add", ".")
    run_git(git_repo.workspace.root, "commit", "-m", "Initial commit")

    result = git_repo.show("HEAD", path="first.txt")

    assert result["ok"] is True
    assert "first" in result["content"]
    assert "second" not in result["content"]


def test_git_path_rejects_absolute_path(git_repo, tmp_path):
    with pytest.raises(ValueError, match="Absolute paths are not allowed"):
        git_repo.diff(path="/etc/passwd")


def test_git_path_rejects_workspace_escape(git_repo):
    with pytest.raises(ValueError, match="Path escapes workspace"):
        git_repo.diff(path="../outside.txt")


def test_show_rejects_empty_revision(git_repo):
    with pytest.raises(ValueError, match="revision cannot be empty"):
        git_repo.show("")
