from agent.state import TaskState, TaskStore


def test_task_state_defaults_are_not_an_active_task():
    state = TaskState()

    assert state.active is False
    assert state.task_id is None
    assert state.status == "idle"
    assert state.goal == ""
    assert state.current == ""
    assert state.completed == []
    assert state.blocked == []
    assert state.files == []


def test_task_store_creates_versioned_task(tmp_path):
    store = TaskStore(tmp_path / ".agent" / "task_history.db")

    state = store.create_task(
        "Refactor config loader",
        "Refactor the config loader and preserve behavior.",
    )

    assert state.active is True
    assert state.task_id.startswith("task_")
    assert state.title == "Refactor config loader"
    assert state.goal == "Refactor the config loader and preserve behavior."
    assert state.status == "in_progress"
    assert state.revision == 1
    assert store.path.is_file()


def test_save_state_creates_revision_only_when_state_changes(tmp_path):
    store = TaskStore(tmp_path / "task_history.db")
    state = store.create_task("Test", "Test versioned state")

    unchanged = store.save_state(state)
    assert unchanged.revision == 1

    state.current = "Inspecting files"
    revision_two = store.save_state(state)
    assert revision_two.revision == 2

    revision_two.completed = ["Inspect files"]
    revision_three = store.save_state(revision_two)
    assert revision_three.revision == 3


def test_history_preserves_status_per_revision(tmp_path):
    store = TaskStore(tmp_path / "task_history.db")
    state = store.create_task("Test", "Preserve historical lifecycle state")

    state.current = "Working"
    state = store.save_state(state)
    state.status = "completed"
    completed = store.save_state(state)

    assert completed.revision == 3
    assert store.get_state(completed.task_id, revision=1).status == "in_progress"
    assert store.get_state(completed.task_id, revision=2).status == "in_progress"
    assert store.get_state(completed.task_id, revision=3).status == "completed"


def test_history_is_bounded_and_paginates_by_revision(tmp_path):
    store = TaskStore(tmp_path / "task_history.db")
    state = store.create_task("Test", "Generate several revisions")

    for number in range(1, 6):
        state.current = f"step {number}"
        state = store.save_state(state)

    first_page = store.get_history(state.task_id, limit=2)
    assert [item.revision for item in first_page] == [6, 5]

    second_page = store.get_history(
        state.task_id,
        limit=2,
        before_revision=5,
    )
    assert [item.revision for item in second_page] == [4, 3]


def test_list_and_search_return_compact_summaries(tmp_path):
    store = TaskStore(tmp_path / "task_history.db")
    long_goal = "config " * 100
    state = store.create_task("Config work", long_goal)
    state.current = "Inspecting parser"
    store.save_state(state)

    listed = store.list_tasks(limit=1)
    searched = store.search_tasks("parser", limit=1)

    assert len(listed) == 1
    assert len(searched) == 1
    assert "goal" not in listed[0]
    assert len(listed[0]["goal_preview"]) <= 240
    assert searched[0]["task_id"] == state.task_id


def test_store_reopens_existing_database(tmp_path):
    path = tmp_path / ".agent" / "task_history.db"
    first = TaskStore(path)
    state = first.create_task("Persistent", "Persist across store instances")
    state.current = "saved"
    state = first.save_state(state)

    second = TaskStore(path)
    restored = second.get_state(state.task_id)

    assert restored.current == "saved"
    assert restored.revision == 2
