from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.chat import ChatSession
from agent.session import ChatSessionStore


def test_chat_session_store_round_trip(tmp_path):
    store = ChatSessionStore(tmp_path / ".agent" / "chat_history.db")
    session = store.create_session("Inspect the project")

    messages = [
        {"role": "user", "content": "Inspect the project."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "project_info",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"workspace":"/tmp/project"}',
        },
        {"role": "assistant", "content": "Done."},
    ]

    updated = store.append_messages(
        session.id,
        messages,
        expected_count=0,
    )

    assert updated.message_count == 4
    assert store.load_messages(session.id) == messages


def test_append_rejects_stale_expected_count(tmp_path):
    store = ChatSessionStore(tmp_path / "chat_history.db")
    session = store.create_session("Test")

    store.append_messages(
        session.id,
        [{"role": "user", "content": "one"}],
        expected_count=0,
    )

    with pytest.raises(RuntimeError, match="changed concurrently"):
        store.append_messages(
            session.id,
            [{"role": "assistant", "content": "two"}],
            expected_count=0,
        )


def test_list_sessions_is_bounded_and_newest_first(tmp_path):
    store = ChatSessionStore(tmp_path / "chat_history.db")
    first = store.create_session("First")
    second = store.create_session("Second")

    store.append_messages(
        first.id,
        [{"role": "user", "content": "touch first"}],
        expected_count=0,
    )

    sessions = store.list_sessions(limit=1)

    assert len(sessions) == 1
    assert sessions[0].id == first.id
    assert second.id != first.id


def test_session_prefix_resolution(tmp_path):
    store = ChatSessionStore(tmp_path / "chat_history.db")
    session = store.create_session("Persistent chat")

    assert store.resolve_session_id(session.id) == session.id
    assert store.resolve_session_id(session.id[:16]) == session.id

    with pytest.raises(ValueError, match="at least 8"):
        store.resolve_session_id("sess")


def test_active_task_pointer_is_session_metadata(tmp_path):
    store = ChatSessionStore(tmp_path / "chat_history.db")
    session = store.create_session("Task chat")

    updated = store.set_active_task(session.id, "task_abc")
    assert updated.active_task_id == "task_abc"

    cleared = store.set_active_task(session.id, None)
    assert cleared.active_task_id is None


class FakeInactiveTaskState:
    active = False
    task_id = None
    status = "idle"


class PersistentFakeRuntime:
    def __init__(self, store: ChatSessionStore):
        self.server = SimpleNamespace(
            workspace=SimpleNamespace(root=Path("/tmp/project")),
            chat_sessions=store,
            active_task_id=None,
        )
        self.server.get_active_task_state = lambda: FakeInactiveTaskState()
        self.server.attach_task = self._attach_task
        self.stream_handler = None
        self.snapshots: list[list[dict]] = []

    def _attach_task(self, task_id):
        self.server.active_task_id = task_id
        return FakeInactiveTaskState()

    def set_stream_handler(self, handler):
        self.stream_handler = handler

    def create_chat_history(self):
        return [{"role": "system", "content": "fresh system prompt"}]

    async def run_chat_turn(self, messages, user_message):
        self.snapshots.append(copy.deepcopy(messages))
        messages.append({"role": "user", "content": user_message})
        messages.append(
            {
                "role": "assistant",
                "content": f"reply: {user_message}",
            }
        )
        return f"reply: {user_message}"


@pytest.mark.anyio
async def test_chat_session_persists_and_resumes_across_instances(tmp_path):
    store = ChatSessionStore(tmp_path / "chat_history.db")

    first_runtime = PersistentFakeRuntime(store)
    first_chat = ChatSession(first_runtime)
    await first_chat._run_user_message("remember alpha")

    session_id = first_chat.session_id
    assert session_id is not None

    persisted = store.load_messages(session_id)
    assert persisted == [
        {"role": "user", "content": "remember alpha"},
        {"role": "assistant", "content": "reply: remember alpha"},
    ]

    second_runtime = PersistentFakeRuntime(store)
    second_chat = ChatSession(second_runtime)
    second_chat._resume_session(session_id[:16])

    # The system prompt is regenerated rather than loaded from SQLite.
    assert second_chat.messages[0] == {
        "role": "system",
        "content": "fresh system prompt",
    }

    await second_chat._run_user_message("what did I say?")

    assert {
        "role": "user",
        "content": "remember alpha",
    } in second_runtime.snapshots[0]
    assert {
        "role": "assistant",
        "content": "reply: remember alpha",
    } in second_runtime.snapshots[0]


def test_new_conversation_does_not_create_empty_database_row(tmp_path):
    store = ChatSessionStore(tmp_path / "chat_history.db")
    runtime = PersistentFakeRuntime(store)
    chat = ChatSession(runtime)

    chat._start_new_conversation()

    assert chat.session_id is None
    assert store.list_sessions() == []


class KeywordEmbeddingProvider:
    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "kv" in lowered or "cache" in lowered else 0.0,
            1.0 if "parser" in lowered else 0.0,
            1.0 if "database" in lowered else 0.0,
            0.1,
        ]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def test_delete_session_cascades_messages_and_chunks(tmp_path):
    from agent.session import ChatHistorySearch

    store = ChatSessionStore(tmp_path / "chat_history.db")
    session = store.create_session("KV cache")
    store.append_messages(
        session.id,
        [
            {"role": "user", "content": "KV cache recomputation is expensive."},
            {"role": "assistant", "content": "Use stable session slots."},
        ],
        expected_count=0,
    )

    search = ChatHistorySearch(store, KeywordEmbeddingProvider())
    assert search.search("KV cache", [session.id])

    store.delete_session(session.id)

    assert store.list_sessions() == []
    with pytest.raises(KeyError):
        store.get_session(session.id)


def test_delete_all_sessions_returns_deleted_count(tmp_path):
    store = ChatSessionStore(tmp_path / "chat_history.db")
    store.create_session("One")
    store.create_session("Two")

    assert store.delete_all_sessions() == 2
    assert store.list_sessions() == []


def test_read_messages_filters_user_and_agent_roles(tmp_path):
    store = ChatSessionStore(tmp_path / "chat_history.db")
    session = store.create_session("Role filtering")
    store.append_messages(
        session.id,
        [
            {"role": "user", "content": "I said alpha."},
            {"role": "assistant", "content": "I replied beta."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [],
            },
            {"role": "tool", "content": "internal tool result"},
            {"role": "user", "content": "I also said gamma."},
        ],
        expected_count=0,
    )

    user_only = store.read_messages(session.id, role="user")
    agent_only = store.read_messages(session.id, role="agent")
    both = store.read_messages(session.id)

    assert [item["content"] for item in user_only["messages"]] == [
        "I said alpha.",
        "I also said gamma.",
    ]
    assert [item["content"] for item in agent_only["messages"]] == [
        "I replied beta.",
    ]
    assert [item["role"] for item in both["messages"]] == [
        "user",
        "agent",
        "user",
    ]


def test_read_messages_can_center_around_sequence(tmp_path):
    store = ChatSessionStore(tmp_path / "chat_history.db")
    session = store.create_session("Centered read")
    store.append_messages(
        session.id,
        [
            {"role": "user", "content": f"message {number}"}
            for number in range(1, 8)
        ],
        expected_count=0,
    )

    result = store.read_messages(
        session.id,
        around_sequence=4,
        limit=3,
        role="user",
    )

    assert [item["sequence"] for item in result["messages"]] == [3, 4, 5]


def test_chat_session_access_supports_allow_all_with_exclusions():
    from agent.session import ChatSessionAccess

    access = ChatSessionAccess()
    access.allow_all_sessions()

    assert access.can_read("session_a") is True

    access.deny_session("session_a")
    assert access.can_read("session_a") is False
    assert access.can_read("session_b") is True

    access.allow_session("session_a")
    assert access.can_read("session_a") is True

    access.deny_all_sessions()
    assert access.can_read("session_a") is False
    assert access.can_read("session_b") is False


def test_semantic_search_uses_role_specific_chunk_views(tmp_path):
    from agent.session import ChatHistorySearch

    store = ChatSessionStore(tmp_path / "chat_history.db")

    first = store.create_session("Cache discussion")
    store.append_messages(
        first.id,
        [
            {"role": "user", "content": "I hate KV cache recomputation."},
            {"role": "assistant", "content": "Keep the parser simple."},
        ],
        expected_count=0,
    )

    second = store.create_session("Parser discussion")
    store.append_messages(
        second.id,
        [
            {"role": "user", "content": "The parser is broken."},
            {"role": "assistant", "content": "KV cache slots can help."},
        ],
        expected_count=0,
    )

    search = ChatHistorySearch(
        store,
        KeywordEmbeddingProvider(),
        target_tokens=50,
        max_tokens=80,
        overlap_tokens=10,
    )

    user_results = search.search(
        "KV cache",
        [first.id, second.id],
        role="user",
        limit=2,
    )
    agent_results = search.search(
        "KV cache",
        [first.id, second.id],
        role="agent",
        limit=2,
    )

    assert user_results[0]["session_id"] == first.id
    assert agent_results[0]["session_id"] == second.id


def test_semantic_search_indexes_embeddings_lazily_by_view(tmp_path):
    from agent.session import ChatHistorySearch

    class RecordingProvider(KeywordEmbeddingProvider):
        def __init__(self):
            self.document_batches = []

        def embed_documents(self, texts):
            self.document_batches.append(list(texts))
            return super().embed_documents(texts)

    store = ChatSessionStore(tmp_path / "chat_history.db")
    session = store.create_session("Lazy views")
    store.append_messages(
        session.id,
        [
            {"role": "user", "content": "KV cache matters to me."},
            {"role": "assistant", "content": "Parser advice from the agent."},
        ],
        expected_count=0,
    )
    provider = RecordingProvider()
    search = ChatHistorySearch(store, provider)

    search.search("KV cache", [session.id], role="user")
    first_batch_count = len(provider.document_batches)
    search.search("KV cache", [session.id], role="user")

    assert first_batch_count == 1
    assert len(provider.document_batches) == 1
