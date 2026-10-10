from pathlib import Path

import pytest

from agent.backends import BackendRegistry


def write_models(config: Path, text: str) -> None:
    config.mkdir(parents=True, exist_ok=True)
    (config / "models.toml").write_text(text.strip() + "\n", encoding="utf-8")


def test_registry_seeds_default_models_config(tmp_path):
    config = tmp_path / "config"

    registry = BackendRegistry.from_config_dir(config)

    assert (config / "models.toml").is_file()
    assert registry.primary_backend_id == "primary"
    assert registry.embedding_backend_id == "embedding"
    seed = (config / "models.toml").read_text(encoding="utf-8")
    assert "[backends.embedding]" in seed
    assert 'type = "embedding"' in seed
    assert "[embeddings." not in seed

    primary = registry.model_spec("primary")
    assert primary.base_url == "http://127.0.0.1:8080"
    assert primary.model == "local-agent"
    assert primary.generation.temperature == 0.7
    assert primary.generation.top_p == 0.95
    assert primary.generation.top_k == 20
    assert primary.generation.min_p == 0.0
    assert primary.generation.repeat_penalty == 1.0
    assert primary.generation.reasoning == "auto"
    assert primary.generation.reasoning_budget == 16384

    embedding = registry.embedding_spec("embedding")
    assert embedding.base_url == "http://127.0.0.1:8081"
    assert embedding.model == "bge-small-en-v1.5"
    assert embedding.timeout_seconds == 30.0


def test_registry_does_not_overwrite_existing_config(tmp_path):
    config = tmp_path / "config"
    custom = """
[defaults]
primary_backend = "main"
embedding_backend = "search"

[backends.main]
type = "chat"
url = "http://localhost:9000"
model = "qwen-custom"

[backends.search]
type = "embedding"
url = "http://localhost:9001"
model = "custom-embedding"
query_prefix = ""
"""
    write_models(config, custom)
    before = (config / "models.toml").read_text(encoding="utf-8")

    registry = BackendRegistry.from_config_dir(config)

    assert (config / "models.toml").read_text(encoding="utf-8") == before
    assert registry.model_spec("main").model == "qwen-custom"
    assert registry.embedding_spec("search").query_prefix == ""


def test_registry_loads_multiple_model_backends_and_generation_settings(tmp_path):
    config = tmp_path / "config"
    write_models(
        config,
        """
[defaults]
primary_backend = "primary"
embedding_backend = "default"

[backends.primary]
type = "chat"
url = "http://127.0.0.1:8080/"
model = "qwen3.5-9b"

[backends.primary.generation]
temperature = 0.6
top_p = 0.9
top_k = 40
min_p = 0.05
repeat_penalty = 1.08
reasoning = "auto"
reasoning_budget = 12000
reasoning_budget_message = "budget reached"

[backends.summarizer]
type = "chat"
url = "http://127.0.0.1:8082"
model = "gemma-4-e2b"

[backends.summarizer.generation]
temperature = 1.0
top_p = 0.95
top_k = 64
min_p = 0.0
repeat_penalty = 1.0
reasoning = "none"
reasoning_budget = 1024

[backends.default]
type = "embedding"
url = "http://127.0.0.1:8081"
model = "bge-small-en-v1.5"
""",
    )

    registry = BackendRegistry.from_config_dir(config)
    primary = registry.model_spec("primary")
    summarizer = registry.model_spec("summarizer")

    assert primary.base_url == "http://127.0.0.1:8080"
    assert primary.generation.temperature == 0.6
    assert primary.generation.top_p == 0.9
    assert primary.generation.top_k == 40
    assert primary.generation.min_p == 0.05
    assert primary.generation.repeat_penalty == 1.08
    assert primary.generation.reasoning_budget == 12000
    assert primary.generation.reasoning_budget_message == "budget reached"

    assert summarizer.model == "gemma-4-e2b"
    assert summarizer.generation.temperature == 1.0
    assert summarizer.generation.top_k == 64
    assert summarizer.generation.reasoning == "none"


def test_model_clients_use_config_and_only_override_primary(tmp_path):
    config = tmp_path / "config"
    write_models(
        config,
        """
[defaults]
primary_backend = "primary"
embedding_backend = "default"

[backends.primary]
type = "chat"
url = "http://primary:8080"
model = "primary-model"

[backends.worker]
type = "chat"
url = "http://worker:8082"
model = "worker-model"

[backends.default]
type = "embedding"
url = "http://embedding:8081"
model = "embedding-model"
""",
    )

    registry = BackendRegistry.from_config_dir(config)
    clients = registry.model_clients(
        primary_url_override="http://override:9999/",
        primary_model_override="override-model",
    )

    assert clients["primary"].base_url == "http://override:9999"
    assert clients["primary"].model == "override-model"
    assert clients["worker"].base_url == "http://worker:8082"
    assert clients["worker"].model == "worker-model"


def test_model_clients_expose_stable_primary_alias(tmp_path):
    config = tmp_path / "config"
    write_models(
        config,
        """
[defaults]
primary_backend = "main"
embedding_backend = "default"

[backends.main]
type = "chat"
url = "http://main:8080"
model = "main-model"

[backends.summarizer]
type = "chat"
url = "http://summarizer:8082"
model = "summarizer-model"

[backends.default]
type = "embedding"
url = "http://embedding:8081"
model = "embedding-model"
""",
    )

    registry = BackendRegistry.from_config_dir(config)
    clients = registry.model_clients()

    assert clients["primary"] is clients["main"]
    assert clients["primary"].base_url == "http://main:8080"
    assert clients["primary"].model == "main-model"
    assert clients["summarizer"].model == "summarizer-model"


def test_embedding_client_uses_config_and_cli_overrides(tmp_path):
    config = tmp_path / "config"
    write_models(
        config,
        """
[defaults]
primary_backend = "primary"
embedding_backend = "search"

[backends.primary]
type = "chat"
url = "http://primary:8080"
model = "primary-model"

[backends.search]
type = "embedding"
url = "http://embedding:8081/"
model = "embedding-model"
timeout_seconds = 12.5
query_prefix = "search: "
""",
    )

    registry = BackendRegistry.from_config_dir(config)
    client = registry.embedding_client(
        url_override="http://override:9001/",
        model_override="override-embedding",
    )

    assert client.base_url == "http://override:9001"
    assert client.model == "override-embedding"
    assert client.timeout == 12.5
    assert client.query_prefix == "search: "


def test_registry_rejects_unknown_default_backend(tmp_path):
    config = tmp_path / "config"
    write_models(
        config,
        """
[defaults]
primary_backend = "missing"
embedding_backend = "default"

[backends.primary]
type = "chat"
url = "http://127.0.0.1:8080"
model = "model"

[backends.default]
type = "embedding"
url = "http://127.0.0.1:8081"
model = "embedding"
""",
    )

    with pytest.raises(ValueError, match="unknown backend: missing"):
        BackendRegistry.from_config_dir(config)


def test_registry_validates_generation_values(tmp_path):
    config = tmp_path / "config"
    write_models(
        config,
        """
[defaults]
primary_backend = "primary"
embedding_backend = "default"

[backends.primary]
type = "chat"
url = "http://127.0.0.1:8080"
model = "model"

[backends.primary.generation]
top_p = 1.5

[backends.default]
type = "embedding"
url = "http://127.0.0.1:8081"
model = "embedding"
""",
    )

    with pytest.raises(ValueError, match="top_p must be <= 1.0"):
        BackendRegistry.from_config_dir(config)


def test_rejects_legacy_embedding_namespace(tmp_path):
    config = tmp_path / "config"
    write_models(config, """
[defaults]
primary_backend = "primary"
embedding_backend = "embedding"

[backends.primary]
type = "chat"
url = "http://chat:8080"
model = "qwen"

[backends.embedding]
type = "embedding"
url = "http://embeddings:8081"
model = "bge"

[embeddings.old]
url = "http://legacy:8081"
model = "old"
""")
    with pytest.raises(ValueError, match="unsupported models.toml section"):
        BackendRegistry.from_config_dir(config)


def test_rejects_missing_explicit_backend_type(tmp_path):
    config = tmp_path / "config"
    write_models(config, """
[defaults]
primary_backend = "primary"
embedding_backend = "embedding"

[backends.primary]
url = "http://chat:8080"
model = "qwen"

[backends.embedding]
type = "embedding"
url = "http://embeddings:8081"
model = "bge"
""")
    with pytest.raises(ValueError, match="type must be explicitly"):
        BackendRegistry.from_config_dir(config)


def test_embedding_backend_is_not_a_chat_client(tmp_path):
    registry = BackendRegistry.from_config_dir(tmp_path / "config")
    clients = registry.model_clients()
    assert "embedding" not in clients
    assert {item["type"] for item in registry.summaries()} == {
        "chat", "embedding"
    }
    with pytest.raises(ValueError, match="not a chat backend"):
        registry.model_spec("embedding")
    with pytest.raises(ValueError, match="not an embedding backend"):
        registry.embedding_spec("primary")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ('type = "invalid"', "type must be explicitly"),
        ('type = "embedding"\n[backends.embedding.generation]\ntop_p = 0.9',
         "unsupported backends.embedding setting"),
    ],
)
def test_invalid_backend_configuration_is_rejected(tmp_path, override, message):
    config = tmp_path / "config"
    write_models(config, f"""
[defaults]
primary_backend = "primary"
embedding_backend = "embedding"

[backends.primary]
type = "chat"
url = "http://chat:8080"
model = "qwen"

[backends.embedding]
url = "http://embeddings:8081"
model = "bge"
{override}
""")
    with pytest.raises(ValueError, match=message):
        BackendRegistry.from_config_dir(config)


def test_rejects_default_referring_to_wrong_backend_type(tmp_path):
    config = tmp_path / "config"
    write_models(config, """
[defaults]
primary_backend = "embedding"
embedding_backend = "primary"

[backends.primary]
type = "chat"
url = "http://chat:8080"
model = "qwen"

[backends.embedding]
type = "embedding"
url = "http://embeddings:8081"
model = "bge"
""")
    with pytest.raises(ValueError, match="primary_backend must reference a chat"):
        BackendRegistry.from_config_dir(config)


