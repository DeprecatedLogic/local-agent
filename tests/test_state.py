from agent.state import StateStore, TaskState


def test_task_state_defaults():
    state = TaskState()

    assert state.task == ""
    assert state.status == "idle"
    assert state.current == ""
    assert state.completed == []
    assert state.blocked == []
    assert state.files == []

def test_task_state_round_trip():
    state = TaskState(
        task="Implement Git support",
        status="in_progress",
        current="Testing Git status",
        completed=["Workspace", "Filesystem"],
        blocked=[],
        files=["src/agent/mcp/git.py"],
    )

    restored = TaskState.from_dict(state.to_dict())

    assert restored == state

def test_state_store_missing_file(tmp_path):
    store = StateStore(tmp_path / ".agent/state.json")

    state = store.load()

    assert state == TaskState()

def test_state_store_save_and_load(tmp_path):
    store = StateStore(tmp_path / ".agent/state.json")

    state = TaskState(
        task="Test agent",
        status="in_progress",
        current="Writing tests",
        completed=["Workspace"],
        blocked=["Nothing"],
        files=["tests/test_workspace.py"],
    )

    store.save(state)

    assert store.load() == state

def test_state_store_creates_parent_directory(tmp_path):
    path = tmp_path / ".agent" / "nested" / "state.json"
    store = StateStore(path)

    store.save(TaskState(task="test"))

    assert path.is_file()

def test_state_store_replaces_existing_state(tmp_path):
    store = StateStore(tmp_path / ".agent/state.json")

    store.save(TaskState(task="first"))
    store.save(TaskState(task="second"))

    assert store.load().task == "second"
