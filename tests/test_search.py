import pytest
from agent.mcp_.search import Search
from agent.mcp_.workspace import Workspace


@pytest.fixture
def search(tmp_path):
    return Search(Workspace(tmp_path))


def test_search_in_file(search):
    search.workspace.resolve("test.py").write_text(
        "foo = 1\n"
        "bar = 2\n"
        "foo = 3\n",
        encoding="utf-8",
    )

    result = search.search_in_file("test.py", "foo")

    assert result == [
        {
            "path": "test.py",
            "line": 1,
            "text": "foo = 1",
        },
        {
            "path": "test.py",
            "line": 3,
            "text": "foo = 3",
        },
    ]


def test_search_in_file_case_insensitive(search):
    search.workspace.resolve("test.py").write_text(
        "Hello\n"
        "hello\n"
        "HELLO\n",
        encoding="utf-8",
    )

    result = search.search_in_file(
        "test.py",
        "hello",
        case_sensitive=False,
    )

    assert len(result) == 3


def test_search_files(search):
    search.workspace.resolve("a.py").write_text(
        "target\n",
        encoding="utf-8",
    )
    search.workspace.resolve("b.py").write_text(
        "nothing\n",
        encoding="utf-8",
    )

    result = search.search_files("target")

    assert result == [
        {
            "path": "a.py",
            "line": 1,
            "text": "target",
        }
    ]


def test_search_files_respects_max_results(search):
    for i in range(5):
        search.workspace.resolve(f"{i}.txt").write_text(
            "target\n",
            encoding="utf-8",
        )

    result = search.search_files("target", max_results=2)

    assert len(result) == 2


def test_search_rejects_empty_pattern(search):
    with pytest.raises(ValueError, match="pattern cannot be empty"):
        search.search_files("")


def test_search_ignores_git_and_build_directories(search):
    search.workspace.resolve(".git").mkdir()
    search.workspace.resolve("build").mkdir()

    search.workspace.resolve(".git/ignored.txt").write_text(
        "target\n",
        encoding="utf-8",
    )
    search.workspace.resolve("build/ignored.txt").write_text(
        "target\n",
        encoding="utf-8",
    )
    search.workspace.resolve("real.txt").write_text(
        "target\n",
        encoding="utf-8",
    )

    result = search.search_files("target")

    assert [entry["path"] for entry in result] == ["real.txt"]
