from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from agent.context import resolve_config_dir
from agent.embeddings import BGE_QUERY_PREFIX, LlamaCppEmbeddingClient
from agent.model import GenerationConfig, LlamaCppClient


DEFAULT_MODELS_TOML = '''# Local Agent model/backend configuration.
#
# This file is created automatically when missing. Existing values are never
# overwritten during updates. Add additional model backends here and reference
# them from agents.toml with `backend = "..."`.

[defaults]
primary_backend = "primary"
embedding_backend = "embedding"

[backends.primary]
type = "chat"
url = "http://127.0.0.1:8080"
model = "local-agent"

[backends.primary.generation]
temperature = 0.7
top_p = 0.95
top_k = 20
min_p = 0.0
repeat_penalty = 1.0
reasoning = "auto"
reasoning_budget = 16384
reasoning_budget_message = "<system>Reasoning token-limit reached. Maintain identity and conversation context. Answer from existing conclusions.</system>"

[backends.embedding]
type = "embedding"
url = "http://127.0.0.1:8081"
model = "bge-small-en-v1.5"
timeout_seconds = 30.0
query_prefix = "Represent this sentence for searching relevant passages: "
'''


@dataclass(frozen=True)
class ModelBackendSpec:
    id: str
    base_url: str
    model: str
    generation: GenerationConfig
    type: Literal["chat"] = "chat"

    def create_client(self) -> LlamaCppClient:
        return LlamaCppClient(
            base_url=self.base_url,
            model=self.model,
            generation=self.generation,
        )


@dataclass(frozen=True)
class EmbeddingBackendSpec:
    id: str
    base_url: str
    model: str
    timeout_seconds: float = 30.0
    query_prefix: str = BGE_QUERY_PREFIX
    type: Literal["embedding"] = "embedding"

    def create_client(self) -> LlamaCppEmbeddingClient:
        return LlamaCppEmbeddingClient(
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout_seconds,
            query_prefix=self.query_prefix,
        )


BackendSpec: TypeAlias = ModelBackendSpec | EmbeddingBackendSpec


class BackendRegistry:
    """Config-driven model and embedding backend registry."""

    def __init__(
        self,
        backends: dict[str, BackendSpec],
        *,
        primary_backend_id: str,
        embedding_backend_id: str,
        config_dir: Path,
        source: Path,
    ):
        self._backends = dict(backends)
        self.primary_backend_id = primary_backend_id
        self.embedding_backend_id = embedding_backend_id
        self.config_dir = config_dir
        self.source = source

    @staticmethod
    def _identifier(value: object, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError(f"{field_name} cannot be empty")
        if not normalized.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"{field_name} contains invalid characters")
        return normalized

    @staticmethod
    def _text(
        value: object,
        *,
        field_name: str,
        default: str | None = None,
        allow_empty: bool = False,
    ) -> str:
        if value is None:
            if default is None:
                raise ValueError(f"{field_name} is required")
            return default
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        normalized = value.strip()
        if not normalized and not allow_empty:
            raise ValueError(f"{field_name} cannot be empty")
        return normalized

    @staticmethod
    def _raw_text(
        value: object,
        *,
        field_name: str,
        default: str | None = None,
    ) -> str:
        if value is None:
            if default is None:
                raise ValueError(f"{field_name} is required")
            return default
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        return value

    @staticmethod
    def _number(
        value: object,
        *,
        field_name: str,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
    ) -> float:
        if value is None:
            result = float(default)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number")
            result = float(value)

        if minimum is not None:
            invalid = result < minimum if minimum_inclusive else result <= minimum
            if invalid:
                operator = ">=" if minimum_inclusive else ">"
                raise ValueError(f"{field_name} must be {operator} {minimum}")
        if maximum is not None and result > maximum:
            raise ValueError(f"{field_name} must be <= {maximum}")
        return result

    @staticmethod
    def _integer(
        value: object,
        *,
        field_name: str,
        default: int,
        minimum: int,
    ) -> int:
        if value is None:
            result = default
        else:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
            result = value
        if result < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}")
        return result

    @classmethod
    def _generation_config(
        cls,
        raw: object,
        *,
        field_name: str,
    ) -> GenerationConfig:
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"{field_name} must be a table")

        valid_keys = {
            "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
            "reasoning", "reasoning_budget", "reasoning_budget_message",
        }
        unknown = set(raw) - valid_keys
        if unknown:
            raise ValueError(
                f"unsupported {field_name} setting(s): "
                + ", ".join(sorted(unknown))
            )

        defaults = GenerationConfig()
        return GenerationConfig(
            temperature=cls._number(
                raw.get("temperature"),
                field_name=f"{field_name}.temperature",
                default=defaults.temperature,
                minimum=0.0,
            ),
            top_p=cls._number(
                raw.get("top_p"),
                field_name=f"{field_name}.top_p",
                default=defaults.top_p,
                minimum=0.0,
                maximum=1.0,
            ),
            top_k=cls._integer(
                raw.get("top_k"),
                field_name=f"{field_name}.top_k",
                default=defaults.top_k,
                minimum=0,
            ),
            min_p=cls._number(
                raw.get("min_p"),
                field_name=f"{field_name}.min_p",
                default=defaults.min_p,
                minimum=0.0,
                maximum=1.0,
            ),
            repeat_penalty=cls._number(
                raw.get("repeat_penalty"),
                field_name=f"{field_name}.repeat_penalty",
                default=defaults.repeat_penalty,
                minimum=0.0,
                minimum_inclusive=False,
            ),
            reasoning=cls._text(
                raw.get("reasoning"),
                field_name=f"{field_name}.reasoning",
                default=defaults.reasoning,
            ),
            reasoning_budget=cls._integer(
                raw.get("reasoning_budget"),
                field_name=f"{field_name}.reasoning_budget",
                default=defaults.reasoning_budget,
                minimum=1,
            ),
            reasoning_budget_message=cls._raw_text(
                raw.get("reasoning_budget_message"),
                field_name=f"{field_name}.reasoning_budget_message",
                default=defaults.reasoning_budget_message,
            ),
        )

    @classmethod
    def from_config_dir(
        cls,
        config_dir: str | Path | None,
    ) -> "BackendRegistry":
        resolved = resolve_config_dir(config_dir)
        resolved.mkdir(parents=True, exist_ok=True)
        path = resolved / "models.toml"

        try:
            with path.open("x", encoding="utf-8") as file:
                file.write(DEFAULT_MODELS_TOML)
        except FileExistsError:
            pass

        with path.open("rb") as file:
            data = tomllib.load(file)

        defaults = data.get("defaults", {})
        if not isinstance(defaults, dict):
            raise ValueError("models.toml [defaults] must be a table")
        unexpected_defaults = set(defaults) - {
            "primary_backend", "embedding_backend",
        }
        if unexpected_defaults:
            raise ValueError(
                "unsupported models.toml [defaults] setting(s): "
                + ", ".join(sorted(unexpected_defaults))
            )

        primary_backend_id = cls._identifier(
            defaults.get("primary_backend", "primary"),
            field_name="defaults.primary_backend",
        )
        embedding_backend_id = cls._identifier(
            defaults.get("embedding_backend", "embedding"),
            field_name="defaults.embedding_backend",
        )

        raw_backends = data.get("backends", {})
        if not isinstance(raw_backends, dict) or not raw_backends:
            raise ValueError("models.toml must define at least one [backends.*] table")

        # All backends share one namespace and MUST declare a type.
        # Unknown top-level sections are errors, never silently ignored.
        unsupported = set(data) - {"defaults", "backends"}
        if unsupported:
            raise ValueError(
                "unsupported models.toml section(s): "
                + ", ".join(sorted(unsupported))
                + "; define every backend under [backends.<id>]"
            )

        backends: dict[str, BackendSpec] = {}
        for raw_id, raw_spec in raw_backends.items():
            field_name = f"backends.{raw_id}"
            backend_id = cls._identifier(raw_id, field_name=field_name)
            if backend_id in backends:
                raise ValueError(f"duplicate backend id: {backend_id}")
            if not isinstance(raw_spec, dict):
                raise ValueError(f"{field_name} must be a table")

            kind = raw_spec.get("type")
            if kind not in ("chat", "embedding") or not isinstance(kind, str):
                raise ValueError(
                    f"{field_name}.type must be explicitly 'chat' or 'embedding'"
                )

            allowed = (
                {"type", "url", "model", "generation"}
                if kind == "chat"
                else {"type", "url", "model", "timeout_seconds", "query_prefix"}
            )
            unexpected = set(raw_spec) - allowed
            if unexpected:
                raise ValueError(
                    f"unsupported {field_name} setting(s): "
                    + ", ".join(sorted(unexpected))
                )

            url = cls._text(
                raw_spec.get("url"), field_name=f"{field_name}.url"
            ).rstrip("/")
            model = cls._text(
                raw_spec.get("model"), field_name=f"{field_name}.model"
            )

            if kind == "chat":
                backends[backend_id] = ModelBackendSpec(
                    id=backend_id,
                    base_url=url,
                    model=model,
                    generation=cls._generation_config(
                        raw_spec.get("generation"),
                        field_name=f"{field_name}.generation",
                    ),
                )
            else:
                backends[backend_id] = EmbeddingBackendSpec(
                    id=backend_id,
                    base_url=url,
                    model=model,
                    timeout_seconds=cls._number(
                        raw_spec.get("timeout_seconds"),
                        field_name=f"{field_name}.timeout_seconds",
                        default=30.0,
                        minimum=0.0,
                        minimum_inclusive=False,
                    ),
                    query_prefix=cls._raw_text(
                        raw_spec.get("query_prefix"),
                        field_name=f"{field_name}.query_prefix",
                        default=BGE_QUERY_PREFIX,
                    ),
                )

        if primary_backend_id not in backends:
            raise ValueError(
                "defaults.primary_backend references unknown backend: "
                f"{primary_backend_id}"
            )
        if not isinstance(backends[primary_backend_id], ModelBackendSpec):
            raise ValueError("defaults.primary_backend must reference a chat backend")

        if embedding_backend_id not in backends:
            raise ValueError(
                "defaults.embedding_backend references unknown backend: "
                f"{embedding_backend_id}"
            )
        if not isinstance(backends[embedding_backend_id], EmbeddingBackendSpec):
            raise ValueError(
                "defaults.embedding_backend must reference an embedding backend"
            )

        # 'primary' in agents.toml is a semantic alias. Do not silently hide a
        # different concrete chat backend that happens to have this name.
        if primary_backend_id != "primary" and isinstance(
            backends.get("primary"), ModelBackendSpec
        ):
            raise ValueError(
                "backend id 'primary' is reserved for the selected primary "
                "backend; remove or rename the conflicting chat backend"
            )

        return cls(
            backends,
            primary_backend_id=primary_backend_id,
            embedding_backend_id=embedding_backend_id,
            config_dir=resolved,
            source=path,
        )

    def model_spec(self, backend_id: str) -> ModelBackendSpec:
        key = self._identifier(backend_id, field_name="backend id")
        spec = self._backends.get(key)
        if spec is None:
            raise KeyError(f"unknown model backend '{key}'")
        if not isinstance(spec, ModelBackendSpec):
            raise ValueError(f"backend '{key}' is not a chat backend")
        return spec

    def embedding_spec(self, backend_id: str) -> EmbeddingBackendSpec:
        key = self._identifier(backend_id, field_name="embedding backend id")
        spec = self._backends.get(key)
        if spec is None:
            raise KeyError(f"unknown embedding backend '{key}'")
        if not isinstance(spec, EmbeddingBackendSpec):
            raise ValueError(f"backend '{key}' is not an embedding backend")
        return spec

    def model_clients(
        self,
        *,
        primary_url_override: str | None = None,
        primary_model_override: str | None = None,
    ) -> dict[str, LlamaCppClient]:
        clients: dict[str, LlamaCppClient] = {}
        for backend_id, spec in self._backends.items():
            if not isinstance(spec, ModelBackendSpec):
                continue
            base_url = spec.base_url
            model = spec.model
            if backend_id == self.primary_backend_id:
                if primary_url_override is not None:
                    base_url = self._text(
                        primary_url_override,
                        field_name="--model-url",
                    ).rstrip("/")
                if primary_model_override is not None:
                    model = self._text(
                        primary_model_override,
                        field_name="--model",
                    )

            clients[backend_id] = LlamaCppClient(
                base_url=base_url,
                model=model,
                generation=spec.generation,
            )

        # `primary` is the role alias for the backend selected in [defaults].
        clients["primary"] = clients[self.primary_backend_id]

        return clients

    def embedding_client(
        self,
        *,
        url_override: str | None = None,
        model_override: str | None = None,
    ) -> LlamaCppEmbeddingClient:
        spec = self.embedding_spec(self.embedding_backend_id)
        base_url = spec.base_url
        model = spec.model

        if url_override is not None:
            base_url = self._text(
                url_override,
                field_name="--embedding-url",
            ).rstrip("/")
        if model_override is not None:
            model = self._text(
                model_override,
                field_name="--embedding-model",
            )

        return LlamaCppEmbeddingClient(
            base_url=base_url,
            model=model,
            timeout=spec.timeout_seconds,
            query_prefix=spec.query_prefix,
        )

    def summaries(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for spec in sorted(self._backends.values(), key=lambda item: item.id):
            entry: dict[str, Any] = {
                "id": spec.id,
                "type": spec.type,
                "url": spec.base_url,
                "model": spec.model,
                "primary": spec.id == self.primary_backend_id,
            }
            if isinstance(spec, ModelBackendSpec):
                entry["generation"] = {
                    "temperature": spec.generation.temperature,
                    "top_p": spec.generation.top_p,
                    "top_k": spec.generation.top_k,
                    "min_p": spec.generation.min_p,
                    "repeat_penalty": spec.generation.repeat_penalty,
                    "reasoning": spec.generation.reasoning,
                    "reasoning_budget": spec.generation.reasoning_budget,
                }
            else:
                entry["embedding"] = {
                    "timeout_seconds": spec.timeout_seconds,
                    "query_prefix": spec.query_prefix,
                }
            result.append(entry)
        return result
