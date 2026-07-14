"""Runtime composition root for the LangGraph aerospace agent.

This module creates process-local services and keeps them outside graph state.
Its returned facade deliberately has no shutdown method: the agent that uses
the runtime handles is responsible for their lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import AgentSettings, MCPSettings
from ..graph import ServiceBundle
from .context import ContextService
from .evolution import EvolutionService
from .evolution_policy import EvolutionPolicy
from .knowledge import KnowledgeService
from .mcp_gateway import MCPGateway, create_mcp_gateway
from .planner import LLMPlanner


GatewayBuilder = Callable[[MCPSettings], tuple[MCPGateway, Sequence[str]]]


def create_context_service(workspace: str | Path, settings: Any) -> ContextService:
    """Build the bounded context service for callers without a full factory."""

    context = ContextService(
        workspace,
        max_tokens=int(getattr(settings, "max_tokens", 8192)),
        recent_turns=int(getattr(settings, "recent_turns", 8)),
    )
    context.artifact_dir = Path(getattr(settings, "artifacts_dir", Path(workspace) / "data" / "langgraph" / "artifacts"))
    return context


@dataclass(frozen=True)
class RuntimeServices:
    """Runtime-only service handles assembled for one agent lifetime."""

    bundle: ServiceBundle
    context: ContextService
    knowledge: KnowledgeService
    evolution: EvolutionService
    llm: Any
    gateway: MCPGateway
    warnings: tuple[str, ...] = ()

    @property
    def services(self) -> ServiceBundle:
        """Compatibility name for callers that expect the graph bundle."""

        return self.bundle


class RuntimeServicesFactory:
    """Build the concrete runtime services from validated settings.

    ``gateway_builder`` is injectable to keep unit tests independent of the
    stdio transport.  Production uses :func:`create_mcp_gateway`.
    """

    def __init__(
        self,
        settings: AgentSettings,
        *,
        project_root: str | Path,
        allow_degraded_mcp: bool = False,
        mock_llm: bool = False,
        check_llm_endpoint: bool = False,
        gateway_builder: GatewayBuilder | None = None,
    ) -> None:
        self.settings = settings
        self.project_root = Path(project_root).resolve()
        if self.project_root != settings.workspace_root:
            raise ValueError(
                "project root does not match the selected workspace: "
                f"{self.project_root} != {settings.workspace_root}"
            )
        self._validate_runtime_paths()
        self._mock_llm = bool(mock_llm)
        self._check_llm_endpoint = bool(check_llm_endpoint)
        if gateway_builder is None:
            self._gateway_builder = lambda mcp_settings: create_mcp_gateway(
                mcp_settings,
                allow_inprocess_fallback=allow_degraded_mcp,
            )
        else:
            self._gateway_builder = gateway_builder

    def _validate_runtime_paths(self) -> None:
        """Reject settings validated for a different workspace root."""

        settings = self.settings
        paths = (
            settings.knowledge.workspace,
            settings.knowledge.data_dir,
            settings.knowledge.graph_output,
            settings.context.artifacts_dir,
            settings.checkpoint.path,
            settings.evolution.data_dir,
            *settings.evolution.allowed_roots,
        )
        for configured_path in paths:
            resolved_path = Path(configured_path).resolve()
            if not resolved_path.is_relative_to(self.project_root):
                raise ValueError(
                    f"configured runtime path escapes project root: {resolved_path}"
                )

    @staticmethod
    def _create_local_llm(endpoint: str, model: str, *, timeout: float = 60.0) -> Any:
        # ``SimpleLLMClient`` remains in the public agent module for
        # compatibility.  Delaying the import avoids a services-package
        # import cycle while the legacy agent module is still imported.
        from ..agent import SimpleLLMClient

        return SimpleLLMClient(endpoint=endpoint, model=model, timeout=timeout)

    @classmethod
    def check_llm_endpoint(cls, endpoint: str, model: str) -> bool:
        """Probe a local endpoint without exposing LLM construction to the CLI."""

        return bool(cls._create_local_llm(endpoint, model, timeout=2.0).is_available())

    def create_knowledge_service(self) -> KnowledgeService:
        """Build the knowledge service for non-agent CLI operations."""

        settings = self.settings
        return KnowledgeService(
            self.project_root,
            wiki_dir=settings.knowledge.workspace,
            data_dir=settings.knowledge.data_dir,
        )

    def create_evolution_service(
        self,
        *,
        knowledge_service: KnowledgeService | None = None,
    ) -> EvolutionService:
        """Build the evolution service for non-agent CLI operations."""

        settings = self.settings
        return EvolutionService(
            workspace=self.project_root,
            data_dir=settings.evolution.data_dir,
            allowed_roots=settings.evolution.allowed_roots,
            knowledge_service=knowledge_service,
            policy=EvolutionPolicy(
                enabled=settings.evolution.enabled,
                idle_minutes=settings.evolution.idle_minutes,
                min_turns=settings.evolution.min_turns,
                allowed_roots=settings.evolution.allowed_roots,
            ),
        )

    @staticmethod
    def list_tool_definitions() -> list[dict[str, Any]]:
        """Return declared MCP tools without exposing their registry to the CLI."""

        from aerospace_agent.mcp.tools import get_tool_definitions

        return list(get_tool_definitions())

    def create(self) -> RuntimeServices:
        """Create runtime dependencies without taking ownership of shutdown."""

        settings = self.settings
        workspace = self.project_root
        # Materialize only the workspace roots declared by the configuration.
        # Service constructors create their own databases lazily, but the
        # project contract requires the durable roots to exist immediately
        # after runtime initialization (and makes a fresh project observable
        # without writing seed content).
        declared_directories = (
            settings.knowledge.workspace,
            settings.knowledge.data_dir,
            settings.context.artifacts_dir,
            settings.evolution.data_dir,
            *settings.evolution.allowed_roots,
        )
        for configured_path in declared_directories:
            path = Path(configured_path).resolve()
            if not path.is_relative_to(workspace):
                raise ValueError(f"configured runtime path escapes project root: {path}")
            path.mkdir(parents=True, exist_ok=True)
        context = create_context_service(workspace, settings.context)
        knowledge = self.create_knowledge_service()
        evolution = self.create_evolution_service(knowledge_service=knowledge)
        llm = None if self._mock_llm else self._create_local_llm(settings.llm.endpoint, settings.llm.model)
        if llm is not None and self._check_llm_endpoint and not llm.is_available():
            raise RuntimeError("Qwen endpoint is unavailable")
        gateway, warnings = self._gateway_builder(settings.mcp)
        tool_names = [
            str(item.get("name", ""))
            for item in self.list_tool_definitions()
            if isinstance(item, dict) and item.get("name")
        ]
        planner = LLMPlanner(llm, tool_names=tool_names) if llm is not None else None
        bundle = ServiceBundle(
            knowledge=knowledge,
            context=context,
            evolution=evolution,
            planner=planner,
            mcp_gateway=gateway,
            llm=llm,
            model_name="mock" if self._mock_llm else settings.llm.model,
            endpoint="" if self._mock_llm else settings.llm.endpoint,
            runtime_warnings=tuple(warnings),
        )
        return RuntimeServices(
            bundle=bundle,
            context=context,
            knowledge=knowledge,
            evolution=evolution,
            llm=llm,
            gateway=gateway,
            warnings=tuple(warnings),
        )


__all__ = ["RuntimeServices", "RuntimeServicesFactory", "create_context_service"]
