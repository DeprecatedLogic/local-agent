from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import tomllib


@dataclass(frozen=True)
class AgentIdentity:
    name: str = "Local Agent"
    role: str = "autonomous local Linux agent"
    description: str = """
        A general-purpose local agent that investigates, reasons, and acts through the
        tools exposed by its runtime.

        You operate as a Linux user inside boundaries enforced by the runtime. Your tools are
        an extension of your ability to observe and act on reality.

        Prefer direct evidence over assumptions when a claim depends on current,
        environment-specific, system-dependent, filesystem-dependent, repository-dependent,
        runtime-dependent, version-dependent, or otherwise observable information. Never
        invent tool results, command output, file contents, software versions, system state,
        test results, successful actions, or other observations. If temporal reasoning depends
        on whether something is past, present, or future, observe the current date/time first.

        Use tools when they materially improve correctness. Respect runtime boundaries and do
        not assume privileges or capabilities you have not established. Before meaningful
        modifications, inspect relevant state; afterward, validate the result using the
        strongest practical evidence available.

        Ordinary conversation is not automatically a persistent task. Use start_task only for
        substantive work that benefits from progress tracking or later resumption. Do not
        create tasks for greetings, casual conversation, or trivial factual questions. When a
        persistent task is active, keep its progress accurate with the task-state tools and
        finish it explicitly when appropriate.

        Detailed operating guidance is available through get_agent_context. Consult only the
        relevant topic when more detail is needed; do not load guidance merely for appearance.

        When evidence is insufficient, preserve the uncertainty. Correctness, evidence, and
        truthfulness take priority over confidence, fluency, speed, or completeness.
    """.strip()


def resolve_config_dir(config_dir: str | Path | None = None) -> Path:
    """Resolve the Local Agent config directory without mutating it."""
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve()

    configured = os.environ.get("LOCAL_AGENT_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return (base / "local-agent").resolve()


class AgentContextStore:
    """Identity and lazily loaded operating guidance for an agent runtime."""

    def __init__(self, config_dir: str | Path | None = None):
        self.package_manual_dir = Path(__file__).with_name("manual")
        self.config_dir = resolve_config_dir(config_dir)
        self.manual_dir = self.config_dir / "manual"

        self._ensure_config()
        self.identity = self._load_identity()

    def _ensure_config(self) -> None:
        """Seed editable config defaults without overwriting user-owned files."""
        self.manual_dir.mkdir(parents=True, exist_ok=True)

        self._write_if_missing(
            self.config_dir / "identity.toml",
            self._default_identity_toml(),
        )

        # All editable defaults are part of the same first-run setup.
        # Import lazily because both registries also use config-directory helpers.
        from agent.backends import DEFAULT_MODELS_TOML
        from agent.orchestration import DEFAULT_AGENTS_TOML

        self._write_if_missing(
            self.config_dir / "models.toml",
            DEFAULT_MODELS_TOML,
        )
        self._write_if_missing(
            self.config_dir / "agents.toml",
            DEFAULT_AGENTS_TOML,
        )

        if not self.package_manual_dir.is_dir():
            return

        for source in sorted(self.package_manual_dir.glob("*.md")):
            self._write_if_missing(
                self.manual_dir / source.name,
                source.read_text(encoding="utf-8"),
            )

    @staticmethod
    def _write_if_missing(path: Path, content: str) -> None:
        """Create a file exactly once and leave any existing file untouched."""
        try:
            with path.open("x", encoding="utf-8") as file:
                file.write(content)
        except FileExistsError:
            pass

    @staticmethod
    def _default_identity_toml() -> str:
        defaults = AgentIdentity()

        # JSON string syntax is valid TOML basic-string syntax and gives us
        # correct escaping without maintaining a second set of defaults.
        name = json.dumps(defaults.name, ensure_ascii=False)
        role = json.dumps(defaults.role, ensure_ascii=False)
        description = json.dumps(defaults.description, ensure_ascii=False)

        return (
            "# Local Agent identity configuration.\n"
            "#\n"
            "# This file is created automatically when missing.\n"
            "# Existing values are never overwritten during updates.\n"
            "\n"
            "[identity]\n"
            f"name = {name}\n"
            f"role = {role}\n"
            f"description = {description}\n"
        )

    def _load_identity(self) -> AgentIdentity:
        path = self.config_dir / "identity.toml"

        with path.open("rb") as file:
            data = tomllib.load(file)

        identity = data.get("identity", {})
        if not isinstance(identity, dict):
            raise ValueError("identity.toml [identity] must be a table")

        defaults = AgentIdentity()
        return AgentIdentity(
            name=self._text(identity.get("name"), defaults.name),
            role=self._text(identity.get("role"), defaults.role),
            description=self._text(
                identity.get("description"),
                defaults.description,
            ),
        )

    @staticmethod
    def _text(value: object, default: str) -> str:
        if value is None:
            return default
        if not isinstance(value, str):
            raise ValueError("identity values must be strings")
        value = value.strip()
        return value or default

    def topics(self) -> list[str]:
        if not self.manual_dir.is_dir():
            return []

        return sorted(path.stem for path in self.manual_dir.glob("*.md"))

    def get(self, topic: str | None = None) -> dict:
        if topic is None:
            return {
                "identity": {
                    "name": self.identity.name,
                    "role": self.identity.role,
                    "description": self.identity.description,
                },
                "topics": self.topics(),
            }

        normalized = topic.strip().lower()
        if not normalized:
            raise ValueError("topic cannot be empty")
        if not normalized.replace("-", "").replace("_", "").isalnum():
            raise ValueError("topic contains invalid characters")

        path = self.manual_dir / f"{normalized}.md"
        if not path.is_file():
            available = ", ".join(self.topics()) or "none"
            raise KeyError(
                f"unknown agent context topic: {normalized}; available: {available}"
            )

        return {
            "topic": normalized,
            "content": path.read_text(encoding="utf-8"),
            "source": "config",
        }


def build_system_prompt(identity: AgentIdentity) -> str:
    return f"""
You are {identity.name}, {identity.role}.
{identity.description}
""".strip()
