"""Production composition root for workspace-bound Agent Core execution.

The graph receives this object only as a runtime dependency.  It owns the
concrete tool catalog and execution boundary, while creating a distinct
SessionMemoryService/catalog/registry tuple for each thread.  No raw handler
is exposed to graph state or planners.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from .approval import CapabilityApprovalLedger, CapabilityApprovalVerifier
from .capabilities import ALLOWED_IMPORT_ROOTS, CapabilityRegistry
from .confirmation import ConfirmationService
from .dag import CanonicalMetadataVocabulary, CheckpointedDAGExecutor
from .execution import (
    AuthorizedExecutor,
    ExecutionContext,
    ExecutionRegistry,
    ExecutionRequest,
    ExecutionService,
)
from .evolution import EvolutionService as CandidateEvolutionService
from .integrations import CapabilityAcquisitionService, IntegrationTrustService
from .execution_checkpoints import ExecutionCheckpointStore
from .git_service import GitService
from .models import CheckpointRef, ToolError, ToolResult
from .planning import PlanExecutionVerifier
from .rag_gate import ExecutionRunStore
from .routing import CapabilityRoute, CapabilityRouter
from .scheduler import SchedulerService
from .session_memory import SessionMemoryService
from .tools import (
    CoreToolCatalog,
    CoreToolServices,
    FileService,
    TerminalService,
    build_core_tool_catalog,
)
from .tools.browser import (
    BrowserService,
    build_playwright_navigation_adapter,
    build_playwright_screenshot_adapter,
)
from .tools.web import WebService
from .workflows import WorkflowRegistry
from ..services.task_planning import DeterministicReviewAssessor, LLMTaskPlanService
from .review import ReviewService


CheckpointValidator = Callable[[CheckpointRef, str], bool]

_ADAPTER_IMPORT_ROOTS = frozenset(
    {
        "aerospace_agent.mcp.tools",
        "aerospace_agent.integrations",
        "aerospace_agent.domains",
    }
)


@dataclass(frozen=True)
class TrustedCodeRoot:
    """One explicit, versioned source-code trust boundary."""

    root_id: str
    schema_version: Literal["1.0"]
    purpose: Literal["adapter", "input_contract"]
    path: Path
    import_root: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported trusted code root schema version")
        if not self.root_id.strip():
            raise ValueError("trusted code root_id is required")
        resolved = Path(self.path).resolve(strict=True)
        object.__setattr__(self, "path", resolved)
        if self.purpose == "adapter":
            if self.import_root not in _ADAPTER_IMPORT_ROOTS:
                raise ValueError("allowed adapter import root is required")
            if self.import_root not in ALLOWED_IMPORT_ROOTS:
                raise ValueError("adapter import root is not executable")
            if not resolved.is_dir():
                raise ValueError("adapter trusted code root must be a directory")
            expected = Path(*str(self.import_root).split("."))
            if tuple(resolved.parts[-len(expected.parts) :]) != expected.parts:
                raise ValueError("adapter path does not match its import root")
        else:
            if self.import_root is not None:
                raise ValueError("input contract root cannot declare an import root")
            if not resolved.is_file():
                raise ValueError("input contract trusted code root must be an exact file")

    def contains(self, candidate: Path) -> bool:
        resolved = candidate.resolve(strict=True)
        if self.purpose == "input_contract":
            return resolved == self.path
        return resolved.is_relative_to(self.path)

    def descriptor(self) -> dict[str, str | None]:
        return {
            "root_id": self.root_id,
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "path": str(self.path),
            "import_root": self.import_root,
        }


@dataclass(frozen=True)
class ThreadExecutionRuntime:
    """One immutable execution composition bound to exactly one thread."""

    thread_id: str | None
    services: CoreToolServices
    catalog: CoreToolCatalog
    capabilities: CapabilityRegistry
    router: CapabilityRouter
    registry: ExecutionRegistry
    execution_service: ExecutionService


class WorkspaceExecutionRegistry(ExecutionRegistry):
    """Execution registry with separate project-data and trusted-code roots.

    ``ExecutionRegistry`` correctly uses ``workspace_root`` for every target
    path policy.  A project initialized outside the source checkout still
    needs to load this package's verified built-in adapters, so registration
    additionally accepts only the resolved current-repository package tree.
    Runtime tool paths remain constrained to the initialized project root.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        trusted_code_roots: tuple[TrustedCodeRoot, ...],
        **kwargs: Any,
    ) -> None:
        roots = tuple(trusted_code_roots)
        if not roots:
            raise ValueError("trusted_code_roots must be explicit")
        root_ids = [root.root_id for root in roots]
        if len(root_ids) != len(set(root_ids)):
            raise ValueError("trusted code root IDs must be unique")
        self.trusted_code_roots = roots
        encoded = json.dumps(
            [root.descriptor() for root in roots],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.trusted_code_roots_sha256 = hashlib.sha256(encoded).hexdigest()
        super().__init__(workspace_root, **kwargs)

    def _trusted_root_for(
        self,
        path: Path,
        *,
        purpose: Literal["adapter", "input_contract"] | None = None,
        import_root: str | None = None,
    ) -> TrustedCodeRoot | None:
        for root in self.trusted_code_roots:
            if purpose is not None and root.purpose != purpose:
                continue
            if import_root is not None and root.import_root != import_root:
                continue
            if root.contains(path):
                return root
        return None

    def _resolve_registration_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self._workspace_root / candidate
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"registration path is not a file: {resolved}")
        if resolved.is_relative_to(self._workspace_root):
            return resolved
        if self._trusted_root_for(resolved) is not None:
            return resolved
        raise ValueError(f"registration path is outside workspace/explicit trusted roots: {resolved}")

    def _validate_entrypoint_and_handler(
        self,
        *,
        kind: Any,
        manifest: Any,
        entrypoint: str,
        handler: Any,
        adapter_path: Path,
        input_model: Any,
    ) -> None:
        matched_root = self._matched_import_root(entrypoint)
        relative_root = Path(*matched_root.split("."))
        project_directory = (self._workspace_root / relative_root).resolve()
        if not (
            adapter_path.is_relative_to(project_directory)
            or self._trusted_root_for(
                adapter_path,
                purpose="adapter",
                import_root=matched_root,
            )
            is not None
        ):
            raise ValueError("adapter path does not belong to entrypoint import root")
        if not adapter_path.is_file():
            raise ValueError(f"adapter file does not exist: {adapter_path}")
        if not (
            manifest.source == matched_root
            or manifest.source.startswith(f"{matched_root}.")
        ):
            raise ValueError("manifest source is outside the matched import root")
        if not (
            entrypoint.startswith(f"{manifest.source}.")
            or entrypoint == manifest.source
        ):
            raise ValueError("entrypoint does not belong to manifest source")
        if kind == "human":
            return
        if handler is None:
            raise ValueError(f"{kind} executor requires a handler")
        target = inspect.unwrap(handler.__func__ if inspect.ismethod(handler) else handler)
        actual_entrypoint = f"{target.__module__}.{target.__qualname__}"
        if entrypoint != actual_entrypoint:
            raise ValueError(
                f"entrypoint does not match actual handler identity: {actual_entrypoint}"
            )
        source_file = inspect.getsourcefile(target)
        if source_file is None or Path(source_file).resolve() != adapter_path:
            raise ValueError("handler source file does not match adapter path")
        signature = inspect.signature(handler)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            accepted = {
                name
                for name, parameter in signature.parameters.items()
                if parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            }
            unexpected = set(input_model.model_fields) - accepted
            if unexpected:
                raise ValueError(
                    f"input model fields are not accepted by handler: {sorted(unexpected)}"
                )

    def _capture_input_model_evidence(self, input_model: Any) -> tuple[str, Path, str, str]:
        source_file = inspect.getsourcefile(input_model)
        if source_file is None:
            raise ValueError("input model must have inspectable source code")
        path = Path(source_file).resolve()
        if not (
            path.is_relative_to(self._workspace_root)
            or self._trusted_root_for(path) is not None
        ):
            raise ValueError(
                "input model source must stay inside the project or trusted package"
            )
        identity = f"{input_model.__module__}.{input_model.__qualname__}"
        return (
            identity,
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            self._input_model_implementation_sha256(input_model),
        )

    def snapshot(self, capability_id: str):
        snapshot = super().snapshot(capability_id)
        return snapshot.model_copy(
            update={"trusted_code_roots_sha256": self.trusted_code_roots_sha256}
        )


class AgentCoreRuntime:
    """Own concrete Core services and the sole safe direct-execution facade."""

    _MEMORY_LINE = re.compile(
        r"^\s*(?:[-*]\s*)?"
        r"(?P<label>约束|constraint|决定|decision|记住|remember|假设|assumption|assume)"
        r"\s*[:：]\s*(?P<content>.+?)\s*$",
        re.IGNORECASE,
    )
    _PREFERENCE_PATTERNS = (
        re.compile(
            r"^\s*(?:以后|今后|从现在起|请)?\s*(?:都\s*)?(?:称呼我|叫我)\s*(?P<value>[^。！？!?]{1,80})\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:please\s+)?call\s+me\s+(?P<value>[^.!?]{1,80})\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*address\s+me\s+as\s+(?P<value>[^.!?]{1,80})\s*$",
            re.IGNORECASE,
        ),
    )

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        project_id: str,
        session_database_path: str | Path,
        execution_run_store: ExecutionRunStore,
        checkpoint_validator: CheckpointValidator,
        direct_execution_confidence_threshold: float = 0.75,
        terminal_allowed_commands: tuple[str, ...] = ("git", "python", "python3"),
        web_search_endpoint: str | None = None,
        web_search_providers: list[dict[str, Any]] | None = None,
        web_default_search_provider: str | None = None,
        browser_playwright_enabled: bool = True,
        llm: Any = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        if not self.workspace_root.is_dir():
            raise ValueError("workspace_root must be an existing directory")
        if not project_id:
            raise ValueError("project_id is required")
        self.project_id = str(project_id)
        self.session_database_path = Path(session_database_path).resolve()
        self.execution_run_store = execution_run_store
        self._checkpoint_validator = checkpoint_validator
        self._confidence_threshold = float(direct_execution_confidence_threshold)
        self._lock = threading.RLock()
        self._threads: dict[str, ThreadExecutionRuntime] = {}
        self._memory_results: dict[str, dict[str, Any]] = {}

        source_package = Path(__file__).resolve().parents[3] / "aerospace_agent"
        input_contract_path = Path(
            inspect.getsourcefile(build_core_tool_catalog) or ""
        ).resolve(strict=True)
        self.trusted_code_roots = (
            TrustedCodeRoot(
                root_id="current-source-mcp-tools",
                schema_version="1.0",
                purpose="adapter",
                path=source_package / "mcp" / "tools",
                import_root="aerospace_agent.mcp.tools",
            ),
            TrustedCodeRoot(
                root_id="current-source-integrations",
                schema_version="1.0",
                purpose="adapter",
                path=source_package / "integrations",
                import_root="aerospace_agent.integrations",
            ),
            TrustedCodeRoot(
                root_id="current-source-domains",
                schema_version="1.0",
                purpose="adapter",
                path=source_package / "domains",
                import_root="aerospace_agent.domains",
            ),
            TrustedCodeRoot(
                root_id="core-tool-input-contracts",
                schema_version="1.0",
                purpose="input_contract",
                path=input_contract_path,
            ),
        )

        data = self.workspace_root / "data" / "langgraph"
        data.mkdir(parents=True, exist_ok=True)
        self.files = FileService(self.workspace_root)
        self.terminal = TerminalService(
            self.workspace_root,
            allowed_commands=terminal_allowed_commands,
        )
        self.web = WebService(
            self.workspace_root,
            search_endpoint=web_search_endpoint,
            search_providers=web_search_providers,
            default_search_provider=web_default_search_provider,
        )
        screenshot_adapter = (
            build_playwright_screenshot_adapter()
            if browser_playwright_enabled
            else None
        )
        navigation_adapter = (
            build_playwright_navigation_adapter()
            if browser_playwright_enabled
            else None
        )
        self.browser = BrowserService(
            self.workspace_root,
            web_service=self.web,
            screenshot_adapter=screenshot_adapter,
            navigation_adapter=navigation_adapter,
        )
        self.git = GitService(self.workspace_root)
        self.confirmations = ConfirmationService(data / "confirmations.sqlite")
        approval_ledger = CapabilityApprovalLedger(
            data / "capability_approvals.sqlite",
            trusted_public_keys={},
        )
        self.approval_verifier = CapabilityApprovalVerifier(approval_ledger)
        self.workflows = WorkflowRegistry(
            data / "workflows.sqlite",
            approval_verifier=self.approval_verifier,
        )
        self.integration_trust = IntegrationTrustService(
            self.workspace_root,
            approval_verifier=self.approval_verifier,
        )
        self.acquisition = CapabilityAcquisitionService(
            self.workspace_root,
            database_path=data / "capability_acquisition.sqlite",
            project_id=self.project_id,
            confirmation_service=self.confirmations,
        )
        self.evolution_candidates = CandidateEvolutionService(
            self.workspace_root,
            database_path=data / "evolution_candidates.sqlite",
            project_id=self.project_id,
            approval_verifier=self.approval_verifier,
            workflow_registry=self.workflows,
            confirmation_service=self.confirmations,
        )
        self.scheduler = SchedulerService(
            data / "scheduler.sqlite",
            workflow_registry=self.workflows,
            execution_run_store=execution_run_store,
        )
        self._audit_database_path = data / "execution_audit.sqlite"
        self._execution_checkpoint_path = data / "execution_checkpoints.sqlite"
        # One persistent verifier is shared by the startup and every
        # thread-bound registry.  Future TaskPlan/DAG wiring must register into
        # this exact store so an active plan closes the direct-execution path.
        self.plan_execution_verifier = PlanExecutionVerifier(
            data / "plan_execution.sqlite"
        )

        # The startup view is intentionally not bound to a fabricated thread:
        # memory tools are unavailable in this catalog.  Direct execution uses
        # the per-thread view below, where memory and scheduler namespaces are
        # concrete and immutable for the duration of the call.
        base = self._build_runtime(thread_id=None)
        self.core_tool_services = base.services
        self.core_tool_catalog = base.catalog
        self.capability_registry = base.capabilities
        self.execution_registry = base.registry
        self.execution_service = base.execution_service
        self._base_router = base.router
        # Complex-task execution uses the same startup registry and verifier
        # as direct execution.  Memory-bound tools remain thread-local and are
        # therefore not advertised by this immutable planning view.
        self.dag_executor = CheckpointedDAGExecutor(
            database_path=data / "dag.sqlite",
            workspace_root=self.workspace_root,
            registry=self.execution_registry,
            execution_service=self.execution_service,
            plan_verifier=self.plan_execution_verifier,
            metadata_vocabulary=CanonicalMetadataVocabulary(),
        )
        self.review_service = ReviewService(checkpoint_verifier=self.dag_executor)
        self.review_assessor = DeterministicReviewAssessor()
        self.task_plan_service = (
            LLMTaskPlanService(llm, self.capability_registry, self.execution_registry)
            if llm is not None
            else None
        )

    def _service_catalog(
        self,
        *,
        thread_id: str | None,
    ) -> tuple[CoreToolServices, CoreToolCatalog, CapabilityRegistry]:
        memory = (
            SessionMemoryService(
                self.session_database_path,
                project_id=self.project_id,
                thread_id=thread_id,
                checkpoint_validator=self._checkpoint_validator,
            )
            if thread_id is not None
            else None
        )
        placeholder = CapabilityRegistry([])
        services = CoreToolServices(
            files=self.files,
            terminal=self.terminal,
            browser=self.browser,
            web=self.web,
            scheduler=self.scheduler,
            memory=memory,
            git=self.git,
            workflows=self.workflows,
            capabilities=placeholder,
        )
        first_catalog = build_core_tool_catalog(
            self.workspace_root,
            services,
            project_id=self.project_id,
            thread_id=thread_id,
        )
        capabilities = CapabilityRegistry(
            [entry.manifest for entry in first_catalog.entries()]
        )
        services = replace(services, capabilities=capabilities)
        catalog = build_core_tool_catalog(
            self.workspace_root,
            services,
            project_id=self.project_id,
            thread_id=thread_id,
        )
        return services, catalog, capabilities

    def _build_runtime(self, *, thread_id: str | None) -> ThreadExecutionRuntime:
        services, catalog, capabilities = self._service_catalog(thread_id=thread_id)
        registry = WorkspaceExecutionRegistry(
            self.workspace_root,
            trusted_code_roots=self.trusted_code_roots,
            audit_database_path=self._audit_database_path,
            confirmation_service=self.confirmations,
            plan_execution_verifier=self.plan_execution_verifier,
        )
        catalog.register_into(registry)
        execution_service = ExecutionService(
            registry,
            checkpoint_store=ExecutionCheckpointStore(self._execution_checkpoint_path),
        )
        return ThreadExecutionRuntime(
            thread_id=thread_id,
            services=services,
            catalog=catalog,
            capabilities=capabilities,
            router=CapabilityRouter(
                capabilities,
                direct_execution_confidence_threshold=self._confidence_threshold,
            ),
            registry=registry,
            execution_service=execution_service,
        )

    def for_thread(self, thread_id: str) -> ThreadExecutionRuntime:
        """Return the cached runtime whose memory adapter is fixed to thread_id."""

        normalized = str(thread_id or "").strip()
        if not normalized:
            raise ValueError("thread_id is required")
        with self._lock:
            runtime = self._threads.get(normalized)
            if runtime is None:
                runtime = self._build_runtime(thread_id=normalized)
                self._threads[normalized] = runtime
            return runtime

    def route(self, message: str, **kwargs: Any) -> CapabilityRoute:
        """Compatibility routing view for callers without an active state."""

        return self._base_router.route(message, **kwargs)

    def prepare_request(
        self,
        message: str,
        *,
        requested_tool_name: str | None = None,
    ) -> tuple[str | None, dict[str, Any], bool]:
        """Prepare explicit natural-language tool arguments for graph state."""

        return self._base_router.prepare_request(
            message,
            requested_tool_name=requested_tool_name,
        )

    def route_for_state(
        self,
        *,
        state: Mapping[str, Any],
        message: str,
        **kwargs: Any,
    ) -> CapabilityRoute:
        return self.for_thread(str(state.get("thread_id", ""))).router.route(
            message,
            **kwargs,
        )

    def execute_route(
        self,
        *,
        route: CapabilityRoute,
        arguments: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> ToolResult:
        """Resolve to an opaque authorization, then execute through the service."""

        checked_route = CapabilityRoute.model_validate(route.model_dump(mode="python"))
        capability_id = checked_route.selected_capability_id
        executor_name = checked_route.selected_executor_name
        operation_id = (
            f"direct:{state.get('root_run_id') or state.get('run_id')}:"
            f"{executor_name or 'unselected'}"
        )
        if not capability_id or not executor_name:
            return ToolResult(
                status="unavailable",
                error=ToolError(
                    code="unavailable",
                    message="direct route has no selected capability/executor",
                    recoverability="not_applicable",
                ),
                audit_id=hashlib.sha256(operation_id.encode("utf-8")).hexdigest(),
                operation_id=operation_id,
                recovery_class="read_only",
            )
        thread_id = str(state.get("thread_id", "") or "")
        runtime = self.for_thread(thread_id)
        request = ExecutionRequest(
            kind="tool",
            capability_id=capability_id,
            executor_name=executor_name,
            operation_id=operation_id,
            arguments=dict(arguments),
            confirmation_id=str(state.get("confirmation_id", "") or "") or None,
            origin="direct",
        )
        context = ExecutionContext(
            project_id=str(state.get("project_id", "") or ""),
            thread_id=thread_id,
            root_run_id=str(state.get("root_run_id") or state.get("run_id") or ""),
            workspace_root=str(self.workspace_root),
            capability_snapshot=runtime.registry.snapshot(capability_id),
        )
        authorized = runtime.registry.resolve(request, context)
        if isinstance(authorized, ToolResult):
            return authorized
        if not isinstance(authorized, AuthorizedExecutor):
            raise TypeError("ExecutionRegistry returned an invalid authorization result")
        return runtime.execution_service.execute(authorized)

    def audit_records(self, *, thread_id: str | None = None) -> list[dict[str, Any]]:
        records = self.execution_registry.audit_records()
        if thread_id is None:
            return records
        return [item for item in records if item.get("thread_id") == thread_id]

    @classmethod
    def _extract_memories(cls, user_message: str) -> list[tuple[str, str, str]]:
        """Extract only explicitly labelled user statements or assumptions."""

        extracted: list[tuple[str, str, str]] = []
        labels = {
            "约束": ("constraint", "user_stated"),
            "constraint": ("constraint", "user_stated"),
            "决定": ("decision", "user_stated"),
            "decision": ("decision", "user_stated"),
            "记住": ("preference", "user_stated"),
            "remember": ("preference", "user_stated"),
            "假设": ("fact", "assumption"),
            "assumption": ("fact", "assumption"),
            "assume": ("fact", "assumption"),
        }
        for line in str(user_message).splitlines():
            match = cls._MEMORY_LINE.fullmatch(line)
            if match is not None:
                label = match.group("label").casefold()
                content = match.group("content").strip()
                if content:
                    kind, truth_status = labels[label]
                    extracted.append((kind, truth_status, content))
                continue
            for pattern in cls._PREFERENCE_PATTERNS:
                preference = pattern.fullmatch(line)
                if preference is None:
                    continue
                content = line.strip()
                value = preference.group("value").strip().casefold()
                if value not in {"什么", "what", "the"}:
                    extracted.append(("preference", "user_stated", content))
                break
        return extracted

    def persist_after_checkpoint(
        self,
        *,
        thread_id: str,
        user_message: str,
        checkpoint_id: str,
        turn_count: int,
        context_ratio: float,
        task_state_changed: bool,
        user_corrected: bool = False,
    ) -> dict[str, Any]:
        """Persist traceable explicit memory after a successful graph checkpoint.

        The 6-turn/70%-context/state-change policy is evaluated on every call.
        Automatic semantic summaries remain deliberately disabled: without a
        provenance-preserving summarizer, generating one would create claims
        that cannot be traced back to an explicit user statement.
        """

        runtime = self.for_thread(thread_id)
        session = runtime.services.memory
        assert isinstance(session, SessionMemoryService)
        checkpoint = CheckpointRef(
            project_id=self.project_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
        )
        existing = {
            (item.kind, item.truth_status, item.content, item.source_content_hash)
            for item in session.list(include_history=False, limit=500)
        }
        saved: list[str] = []
        for kind, truth_status, content in self._extract_memories(user_message):
            source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            identity = (kind, truth_status, content, source_hash)
            if identity in existing:
                continue
            memory = session.remember(
                kind=kind,
                content=content,
                source_checkpoints=[checkpoint],
                source_content_hash=source_hash,
                truth_status=truth_status,
                confidence=1.0 if truth_status == "user_stated" else 0.5,
            )
            saved.append(memory.memory_id)
            existing.add(identity)
        summary_due = SessionMemoryService.summary_due(
            turn_count=turn_count,
            context_ratio=context_ratio,
            task_state_changed=task_state_changed,
            user_corrected=user_corrected,
        )
        summary_generated = False
        summary_policy = "deterministic_explicit_memory_snapshot"
        if summary_due:
            active = session.list(include_history=False, limit=500)
            if active:
                grouped = {
                    kind: [item.content for item in active if item.kind == kind][:20]
                    for kind in ("preference", "constraint", "decision", "artifact", "fact")
                }
                session.save_summary(
                    current_goal="Deterministic snapshot of explicit session memory",
                    preferences=grouped["preference"],
                    confirmed_constraints=grouped["constraint"],
                    decisions=grouped["decision"],
                    completed_items=[],
                    open_items=[],
                    artifacts=grouped["artifact"],
                    assumptions=grouped["fact"],
                    source_checkpoints=[checkpoint],
                )
                summary_generated = True
        result = {
            "saved_memory_ids": saved,
            "summary_due": summary_due,
            "summary_generated": summary_generated,
            "summary_policy": summary_policy,
        }
        with self._lock:
            self._memory_results[thread_id] = dict(result)
        return result

    def memory_persistence(self, thread_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(
                self._memory_results.get(
                    thread_id,
                    {
                        "saved_memory_ids": [],
                        "summary_due": False,
                        "summary_generated": False,
                        "summary_policy": "deterministic_explicit_memory_snapshot",
                    },
                )
            )

    def close(self) -> None:
        client = getattr(self.web, "_client", None)
        closer = getattr(client, "close", None)
        if callable(closer):
            closer()


__all__ = [
    "AgentCoreRuntime",
    "ThreadExecutionRuntime",
    "TrustedCodeRoot",
    "WorkspaceExecutionRegistry",
]
