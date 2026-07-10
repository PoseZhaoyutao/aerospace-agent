"""Validated configuration for the local-first LangGraph agent.

Configuration is intentionally small and explicit: YAML supplies the runtime
defaults, three environment variables are supported, and all writable paths
are constrained to the selected workspace.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field


_DEFAULT_CONFIG_PATH = Path("config/langgraph_agent.yaml")


class _SettingsModel(BaseModel):
    """Common strict model configuration for settings sections."""

    model_config = ConfigDict(extra="forbid")


class LLMSettings(_SettingsModel):
    endpoint: str = "http://127.0.0.1:8000/v1"
    model: str = "qwythos"


class RuntimeSettings(_SettingsModel):
    max_steps: int = 15
    recursion_limit: int = 40
    cycle_max_repeats: int = 3


class ContextSettings(_SettingsModel):
    max_tokens: int = 8192
    recent_turns: int = 8
    artifacts_dir: Path


class KnowledgeSettings(_SettingsModel):
    workspace: Path
    data_dir: Path
    graph_output: Path


class CheckpointSettings(_SettingsModel):
    backend: str = "sqlite"
    path: Path


class EvolutionSettings(_SettingsModel):
    enabled: bool = True
    idle_minutes: int = 10
    min_turns: int = 6
    data_dir: Path
    allowed_roots: list[Path] = Field(default_factory=list)


class MCPSettings(_SettingsModel):
    transport: str = "stdio"
    command: str = "python"
    args: list[str] = Field(default_factory=lambda: ["-m", "aerospace_agent.mcp.server"])


class AgentSettings(_SettingsModel):
    schema_version: str = "1.0.0"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    context: ContextSettings
    knowledge: KnowledgeSettings
    checkpoint: CheckpointSettings
    evolution: EvolutionSettings
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None = None,
        *,
        workspace: str | Path | None = None,
    ) -> "AgentSettings":
        """Merge a partial mapping with defaults and validate its paths.

        ``workspace`` is the root against which all relative paths are
        resolved.  A resolved path outside that root (including a symlink
        target) is rejected before constructing the settings model.
        """

        root = Path(workspace or Path.cwd()).resolve()
        merged = _deep_merge(copy.deepcopy(_default_mapping()), mapping or {})

        # Every path that can be written by the runtime is resolved centrally.
        knowledge = _section(merged, "knowledge")
        context = _section(merged, "context")
        checkpoint = _section(merged, "checkpoint")
        evolution = _section(merged, "evolution")

        knowledge["workspace"] = _resolve_workspace_path(knowledge["workspace"], root)
        knowledge["data_dir"] = _resolve_workspace_path(knowledge["data_dir"], root)
        knowledge["graph_output"] = _resolve_workspace_path(knowledge["graph_output"], root)
        context["artifacts_dir"] = _resolve_workspace_path(context["artifacts_dir"], root)
        checkpoint["path"] = _resolve_workspace_path(checkpoint["path"], root)
        evolution["data_dir"] = _resolve_workspace_path(evolution["data_dir"], root)
        evolution["allowed_roots"] = [
            _resolve_workspace_path(path, root) for path in evolution["allowed_roots"]
        ]

        return cls.model_validate(merged)


def _default_mapping() -> dict[str, Any]:
    """Return settings defaults matching ``config/langgraph_agent.yaml``."""

    return {
        "schema_version": "1.0.0",
        "llm": {
            "endpoint": "http://127.0.0.1:8000/v1",
            "model": "qwythos",
        },
        "runtime": {
            "max_steps": 15,
            "recursion_limit": 40,
            "cycle_max_repeats": 3,
        },
        "context": {
            "max_tokens": 8192,
            "recent_turns": 8,
            "artifacts_dir": "data/langgraph/artifacts",
        },
        "knowledge": {
            "workspace": "knowledge",
            "data_dir": "data/langgraph/rag",
            "graph_output": "reports/knowledge_graph.html",
        },
        "checkpoint": {
            "backend": "sqlite",
            "path": "data/langgraph/checkpoints.sqlite",
        },
        "evolution": {
            "enabled": True,
            "idle_minutes": 10,
            "min_turns": 6,
            "data_dir": "data/langgraph/evolution",
            "allowed_roots": [
                "knowledge",
                "memory",
                "evolved_skills",
                "workflows/evolved",
            ],
        },
        "mcp": {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "aerospace_agent.mcp.server"],
        },
    }


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge a user mapping without mutating either input."""

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _section(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} settings must be a mapping")
    return value


def _resolve_workspace_path(value: str | os.PathLike[str], workspace: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"path escapes workspace: {resolved}")
    return resolved


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"LangGraph configuration not found: {path}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"LangGraph configuration must be a mapping: {path}")
    return loaded


def load_settings(*, workspace: str | Path | None = None) -> AgentSettings:
    """Load YAML settings, apply the explicit environment overrides, and validate."""

    configured_path = os.environ.get("AEROSPACE_LANGGRAPH_CONFIG")
    if configured_path:
        config_path = Path(configured_path)
        if not config_path.is_absolute():
            config_base = Path(workspace).resolve() if workspace is not None else Path.cwd()
            config_path = config_base / config_path
    else:
        config_path = _DEFAULT_CONFIG_PATH
        if not config_path.is_absolute() and not config_path.exists():
            # Keep the default discoverable when called outside the repository cwd.
            package_root_config = Path(__file__).resolve().parents[2] / _DEFAULT_CONFIG_PATH
            if package_root_config.exists():
                config_path = package_root_config

    mapping = copy.deepcopy(dict(_read_yaml(config_path)))
    llm = mapping.setdefault("llm", {})
    if not isinstance(llm, dict):
        raise ValueError("llm settings must be a mapping")
    if "AEROSPACE_LOCAL_LLM_BASE_URL" in os.environ:
        llm["endpoint"] = os.environ["AEROSPACE_LOCAL_LLM_BASE_URL"]
    if "AEROSPACE_LOCAL_LLM_MODEL" in os.environ:
        llm["model"] = os.environ["AEROSPACE_LOCAL_LLM_MODEL"]
    return AgentSettings.from_mapping(mapping, workspace=workspace)


__all__ = [
    "AgentSettings",
    "CheckpointSettings",
    "ContextSettings",
    "EvolutionSettings",
    "KnowledgeSettings",
    "LLMSettings",
    "MCPSettings",
    "RuntimeSettings",
    "load_settings",
]
