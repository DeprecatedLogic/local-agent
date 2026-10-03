from agent.context import AgentContextStore, build_system_prompt


def test_default_config_is_seeded_and_prompt_stays_compact(tmp_path):
    config = tmp_path / "config"
    store = AgentContextStore(config)
    prompt = build_system_prompt(store.identity)

    assert (config / "identity.toml").is_file()
    assert (config / "manual").is_dir()
    assert store.identity.name == "Local Agent"
    assert "Local Agent" in prompt
    assert "get_agent_context" in prompt
    assert "Ordinary conversation is not automatically a persistent task" in prompt
    assert len(prompt) < 3500

    topics = set(store.topics())
    assert {"evidence", "linux", "software", "tasks"}.issubset(topics)

    for topic in ("evidence", "linux", "software", "tasks"):
        assert (config / "manual" / f"{topic}.md").is_file()


def test_identity_can_be_overridden_without_editing_prompt_code(tmp_path):
    config = tmp_path / "config"

    # First construction creates the editable defaults.
    AgentContextStore(config)

    (config / "identity.toml").write_text(
        """
[identity]
name = "Astra"
role = "autonomous systems agent"
description = "Evidence-first local assistant."
""".strip(),
        encoding="utf-8",
    )

    store = AgentContextStore(config)
    prompt = build_system_prompt(store.identity)

    assert store.identity.name == "Astra"
    assert store.identity.role == "autonomous systems agent"
    assert "Astra" in prompt
    assert "Evidence-first local assistant." in prompt


def test_existing_identity_is_not_overwritten(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    identity = config / "identity.toml"
    custom = """
[identity]
name = "Custom"
role = "custom role"
description = "Custom description."
""".strip() + "\n"
    identity.write_text(custom, encoding="utf-8")

    AgentContextStore(config)

    assert identity.read_text(encoding="utf-8") == custom


def test_existing_manual_is_not_overwritten(tmp_path):
    config = tmp_path / "config"
    manual = config / "manual"
    manual.mkdir(parents=True)
    custom = "custom evidence guidance\n"
    (manual / "evidence.md").write_text(custom, encoding="utf-8")

    store = AgentContextStore(config)
    result = store.get("evidence")

    assert result["source"] == "config"
    assert result["content"] == custom


def test_missing_builtin_manual_is_seeded_without_touching_other_files(tmp_path):
    config = tmp_path / "config"
    store = AgentContextStore(config)

    evidence = config / "manual" / "evidence.md"
    linux = config / "manual" / "linux.md"
    custom_linux = "my custom linux guidance\n"

    evidence.unlink()
    linux.write_text(custom_linux, encoding="utf-8")

    store = AgentContextStore(config)

    assert evidence.is_file()
    assert evidence.read_text(encoding="utf-8")
    assert linux.read_text(encoding="utf-8") == custom_linux
    assert "evidence" in store.topics()


def test_custom_manual_topic_is_discovered(tmp_path):
    config = tmp_path / "config"
    store = AgentContextStore(config)

    custom = config / "manual" / "research.md"
    custom.write_text("research guidance\n", encoding="utf-8")

    store = AgentContextStore(config)
    result = store.get("research")

    assert "research" in store.topics()
    assert result == {
        "topic": "research",
        "content": "research guidance\n",
        "source": "config",
    }


def test_context_index_lists_available_topics(tmp_path):
    store = AgentContextStore(tmp_path / "config")
    index = store.get()

    assert index["identity"]["name"] == "Local Agent"
    assert {"evidence", "linux", "software", "tasks"}.issubset(index["topics"])
