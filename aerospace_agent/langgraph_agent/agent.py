"""Lifecycle-safe facade around the deterministic aerospace LangGraph."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Generator, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from .checkpointer import DEFAULT_CHECKPOINT_DB, create_memory_checkpointer, create_sqlite_checkpointer, list_saved_threads
from .evolution import create_evolution_engine
from .services.evolution_policy import parse_llm_proposal
from .graph import ServiceBundle, build_aerospace_graph, build_simple_graph
from .schema import AgentInput, AgentOutput, IntentType
from .safety import ApprovalRequired, SafetyValidator
from .state import create_initial_state
from .turns import AgentLoop, CommandDecision, TurnContext


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    return {}


class CheckpointSummary(dict):
    """Dict summary that also forwards graph-native snapshot attributes."""
    @property
    def values(self) -> Any:  # type: ignore[override]
        # Snapshot-like attribute; mapping callers can still use [] access.
        return dict.__getitem__(self, "values")

    @property
    def config(self) -> Any:
        snapshot = dict.__getitem__(self, "snapshot")
        return getattr(snapshot, "config", None)

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        snapshot = self.get("snapshot")
        if snapshot is not None and hasattr(snapshot, name):
            return getattr(snapshot, name)
        raise AttributeError(name)


class LangGraphAerospaceAgent:
    """Own a graph, its services, and its checkpointer for one agent lifetime."""

    def __init__(self, llm_endpoint: Optional[str] = None, model_name: str = "qwen3-vl", rag: Any = None,
                 available_tools: Optional[Dict[str, Any]] = None, checkpoint_backend: str = "sqlite",
                 checkpoint_db_path: Optional[str | Path] = None, evolution_db_path: Optional[str | Path] = None,
                 max_steps: int = 15, max_recursion_depth: int = 40, cycle_max_repeats: int = 3,
                 use_llm_intent: bool = False, mode: str = "full", *, settings: Any = None,
                 services: ServiceBundle | None = None, interrupt_before: Iterable[str] | None = None,
                 checkpointer: Any = None, evolution_service: Any | None = None):
        self.settings = settings
        runtime = getattr(settings, "runtime", None)
        checkpoint_settings = getattr(settings, "checkpoint", None)
        llm_settings = getattr(settings, "llm", None)
        if settings is not None:
            if llm_endpoint is None:
                llm_endpoint = getattr(llm_settings, "endpoint", None)
            if model_name == "qwen3-vl":
                model_name = str(getattr(llm_settings, "model", model_name))
            # Explicit constructor values are test/deployment overrides; only
            # replace the documented defaults from the settings file.
            if max_steps == 15:
                max_steps = int(getattr(runtime, "max_steps", max_steps))
            if max_recursion_depth == 40:
                max_recursion_depth = int(getattr(runtime, "recursion_limit", max_recursion_depth))
            if cycle_max_repeats == 3:
                cycle_max_repeats = int(getattr(runtime, "cycle_max_repeats", cycle_max_repeats))
            if checkpoint_db_path is None:
                checkpoint_db_path = getattr(checkpoint_settings, "path", None)
            # A non-default explicit backend (notably ``memory`` in tests) wins
            # over the settings file; otherwise use the configured backend.
            if checkpoint_backend == "sqlite":
                checkpoint_backend = str(getattr(checkpoint_settings, "backend", checkpoint_backend))
        self.model_name = model_name
        self.llm_endpoint = llm_endpoint
        self.rag = rag
        self.available_tools = available_tools or {}
        self.max_steps = int(max_steps)
        self.recursion_limit = int(max_recursion_depth)
        self.max_recursion_depth = self.recursion_limit
        self.cycle_max_repeats = int(cycle_max_repeats)
        self.use_llm_intent = bool(use_llm_intent)
        self.mode = mode
        self.interrupt_before = tuple(interrupt_before or ())
        self.turn_loop = AgentLoop()
        self.checkpoint_backend = checkpoint_backend
        if checkpointer is not None and str(checkpoint_backend).lower() == "sqlite":
            # A directly injected memory saver is an explicit backend override.
            if checkpointer.__class__.__name__ in {"InMemorySaver", "MemorySaver", "FailingCheckpointer"}:
                self.checkpoint_backend = "memory"
        self._checkpoint_db_path = str(checkpoint_db_path or DEFAULT_CHECKPOINT_DB)
        self._checkpoint_ctx = None
        self._closed = False
        self._runtime_memory_warnings: list[str] = []
        self._config: Dict[str, Any] = {"configurable": {"thread_id": "default"}}
        # Runtime composition is the production path.  The direct arguments
        # below remain only for legacy callers and narrowly injected tests.
        self.llm = services.llm if services is not None else (
            self._create_llm(llm_endpoint, model_name) if llm_endpoint else None
        )
        if services is None:
            from .services.planner import LLMPlanner

            tool_names = available_tools.keys() if isinstance(available_tools, Mapping) else ()
            planner = LLMPlanner(self.llm, tool_names=tool_names) if self.llm is not None else None
            services = ServiceBundle(
                knowledge=rag,
                planner=planner,
                mcp_gateway=available_tools,
                llm=self.llm,
                model_name=model_name,
                endpoint=llm_endpoint or "",
            )
        self.services = services
        self.checkpointer = checkpointer
        self._init_checkpointer()
        self._configure_agent_core()
        self._configure_memory_context()
        self.evolution = create_evolution_engine(db_path=evolution_db_path)
        evolution_settings = getattr(settings, "evolution", None)
        evolution_workspace = getattr(getattr(settings, "knowledge", None), "workspace", None) or Path.cwd()
        if settings is None:
            evolution_workspace = getattr(self.services.knowledge, "workspace", evolution_workspace)
        evolution_workspace = Path(evolution_workspace)
        if evolution_workspace.name.lower() == "knowledge":
            evolution_workspace = evolution_workspace.parent
        evolution_data_dir = getattr(evolution_settings, "data_dir", None) if evolution_settings is not None else None
        evolution_roots = getattr(evolution_settings, "allowed_roots", None) if evolution_settings is not None else None
        injected_evolution = evolution_service or getattr(self.services, "evolution", None)
        # Concrete evolution services belong to RuntimeServicesFactory.  The
        # facade accepts an injected service for compatibility, but never
        # constructs one itself; an uncomposed facade can still answer safe
        # no-op evolution queries.
        self.evolution_service = injected_evolution
        self.graph = self._build_graph()

    def _create_llm(self, endpoint: str | None, model_name: str):
        return SimpleLLMClient(endpoint=endpoint, model=model_name) if endpoint else None

    def _init_checkpointer(self) -> None:
        if self.checkpointer is not None:
            return
        if str(self.checkpoint_backend).lower() == "sqlite":
            self._checkpoint_ctx = create_sqlite_checkpointer(self._checkpoint_db_path)
            self.checkpointer = self._checkpoint_ctx.__enter__()
        else:
            self.checkpointer = create_memory_checkpointer()

    def _configure_memory_context(self) -> None:
        """Enable durable memory only for an explicitly initialized project."""

        context = getattr(self.services, "context", None)
        setter = getattr(context, "set_memory_context", None)
        workspace = getattr(context, "workspace", None)
        if not callable(setter) or workspace is None:
            return
        from .agent_core.context_assembler import MemoryContextAssembler
        from .agent_core.project_memory import ProjectIdentityService

        project = ProjectIdentityService(workspace)
        status = project.status()
        if status.state != "ready" or status.project_id is None:
            return

        def checkpoint_exists(checkpoint: Any, _source_hash: str) -> bool:
            if checkpoint.project_id != status.project_id:
                return False
            getter = getattr(self.checkpointer, "get_tuple", None)
            if not callable(getter):
                return False
            try:
                stored = getter(
                    {
                        "configurable": {
                            "thread_id": checkpoint.thread_id,
                            "checkpoint_id": checkpoint.checkpoint_id,
                        }
                    }
                )
            except Exception:
                return False
            return stored is not None

        setter(
            MemoryContextAssembler(
                project_memory=project,
                session_database_path=project.session_db_path,
                project_id=status.project_id,
                checkpoint_validator=checkpoint_exists,
            )
        )

    def _configure_agent_core(self) -> None:
        """Activate Agent Core only for a validated, initialized project."""

        core_settings = getattr(self.settings, "agent_core", None)
        workspace = getattr(self.settings, "workspace_root", None)
        if workspace is None or not bool(getattr(core_settings, "enabled", True)):
            self.services = replace(self.services, agent_core_enabled=False)
            return

        from .agent_core.project_memory import ProjectIdentityService
        from .agent_core.rag_gate import ExecutionRunStore, RagGateService
        from .agent_core.runtime import AgentCoreRuntime

        project = ProjectIdentityService(workspace)
        status = project.status()
        if status.state != "ready" or status.project_id is None:
            # The pre-Agent-Core graph remains the compatibility contract for
            # projects that have not been explicitly initialized.
            self.services = replace(self.services, agent_core_enabled=False)
            return
        configured_project_id = str(getattr(self.services, "project_id", "") or "")
        if configured_project_id and configured_project_id != status.project_id:
            raise ValueError("injected Agent Core project_id does not match initialized project")

        component = "execution_run_store"
        try:
            run_store = getattr(self.services, "execution_run_store", None)
            if run_store is None:
                configured_path = getattr(core_settings, "execution_runs_path", None)
                if configured_path is None:
                    configured_path = Path(workspace) / "data" / "langgraph" / "execution_runs.sqlite"
                run_store = ExecutionRunStore(configured_path)
            rag_gate = getattr(self.services, "rag_gate", None) or RagGateService(run_store)
        except (RuntimeError, sqlite3.DatabaseError, OSError) as exc:
            if not self._is_agent_core_storage_failure(exc):
                raise
            self._disable_agent_core_after_migration_failure(component, exc)
            return

        def checkpoint_exists(checkpoint: Any, _source_hash: str) -> bool:
            if checkpoint.project_id != status.project_id:
                return False
            getter = getattr(self.checkpointer, "get_tuple", None)
            if not callable(getter):
                return False
            try:
                return getter(
                    {
                        "configurable": {
                            "thread_id": checkpoint.thread_id,
                            "checkpoint_id": checkpoint.checkpoint_id,
                        }
                    }
                ) is not None
            except Exception:
                return False

        core_runtime = getattr(self.services, "agent_core_runtime", None)
        if core_runtime is None:
            component = "agent_core_runtime"
            try:
                web_settings = getattr(self.settings, "web", None)
                browser_settings = getattr(self.settings, "browser", None)
                core_runtime = AgentCoreRuntime(
                    workspace,
                    project_id=status.project_id,
                    session_database_path=project.session_db_path,
                    execution_run_store=run_store,
                    checkpoint_validator=checkpoint_exists,
                    llm=self.llm,
                    direct_execution_confidence_threshold=float(
                        getattr(core_settings, "direct_execution_confidence_threshold", 0.75)
                    ),
                    web_search_providers=[
                        item.model_dump(mode="python")
                        if hasattr(item, "model_dump")
                        else dict(item)
                        for item in getattr(web_settings, "search_providers", [])
                    ],
                    web_default_search_provider=getattr(
                        web_settings, "default_search_provider", None
                    ),
                    browser_playwright_enabled=bool(
                        getattr(browser_settings, "playwright_enabled", True)
                    ),
                )
            except (RuntimeError, sqlite3.DatabaseError, OSError) as exc:
                if not self._is_agent_core_storage_failure(exc):
                    raise
                self._disable_agent_core_after_migration_failure(component, exc)
                return

        context = getattr(self.services, "context", None)
        if context is None:
            context_settings = getattr(self.settings, "context", None)
            from .services.runtime import create_context_service

            context = create_context_service(workspace, context_settings)
        self.services = replace(
            self.services,
            agent_core_enabled=True,
            project_id=status.project_id,
            context=context,
            capability_router=(
                getattr(self.services, "capability_router", None) or core_runtime
            ),
            execution_run_store=run_store,
            rag_gate=rag_gate,
            direct_executor=(
                getattr(self.services, "direct_executor", None) or core_runtime
            ),
            agent_core_runtime=core_runtime,
            core_tool_services=core_runtime.core_tool_services,
            core_tool_catalog=core_runtime.core_tool_catalog,
            capability_registry=core_runtime.capability_registry,
            execution_registry=core_runtime.execution_registry,
            execution_service=core_runtime.execution_service,
            plan_execution_verifier=core_runtime.plan_execution_verifier,
            task_plan_service=(
                getattr(self.services, "task_plan_service", None)
                or getattr(core_runtime, "task_plan_service", None)
            ),
            dag_executor=(
                getattr(self.services, "dag_executor", None)
                or getattr(core_runtime, "dag_executor", None)
            ),
            review_service=(
                getattr(self.services, "review_service", None)
                or getattr(core_runtime, "review_service", None)
            ),
            review_assessor=(
                getattr(self.services, "review_assessor", None)
                or getattr(core_runtime, "review_assessor", None)
            ),
            workflow_registry=core_runtime.workflows,
            scheduler_service=core_runtime.scheduler,
            git_service=core_runtime.git,
            evolution_candidate_service=core_runtime.evolution_candidates,
            capability_acquisition_service=core_runtime.acquisition,
            integration_trust_service=core_runtime.integration_trust,
        )

    def _disable_agent_core_after_migration_failure(
        self,
        component: str,
        exc: BaseException,
    ) -> None:
        warning = {
            "code": "migration_failed",
            "component": str(component),
            "message": str(exc),
        }
        self.services = replace(
            self.services,
            agent_core_enabled=False,
            runtime_warnings=(*tuple(getattr(self.services, "runtime_warnings", ())), warning),
        )

    @staticmethod
    def _is_agent_core_storage_failure(exc: BaseException) -> bool:
        if isinstance(exc, (sqlite3.DatabaseError, OSError)):
            return True
        message = str(exc).casefold()
        return any(
            token in message
            for token in (
                "schema",
                "migration",
                "user_version",
                "database",
                "sqlite",
            )
        )

    def _close_checkpointer(self) -> None:
        ctx, self._checkpoint_ctx = self._checkpoint_ctx, None
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            return
        closer = getattr(self.checkpointer, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    def _close_gateway(self) -> None:
        closer = getattr(getattr(self.services, "mcp_gateway", None), "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    def _build_graph(self):
        if self.mode == "simple" and not bool(getattr(self.services, "agent_core_enabled", False)):
            return build_simple_graph(checkpointer=self.checkpointer, services=self.services, interrupt_before=self.interrupt_before)
        knowledge_settings = getattr(self.settings, "knowledge", None)
        retrieval_threshold = float(
            getattr(knowledge_settings, "retrieval_confidence_threshold", 0.60)
        )
        return build_aerospace_graph(checkpointer=self.checkpointer, max_steps=self.max_steps,
                                     max_repeats=self.cycle_max_repeats, max_recursion_depth=self.recursion_limit,
                                     use_llm_intent=self.use_llm_intent,
                                     retrieval_confidence_threshold=retrieval_threshold,
                                     services=self.services, interrupt_before=self.interrupt_before)

    def _config_for(self, thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
        configurable: dict[str, Any] = {"thread_id": str(thread_id)}
        if checkpoint_id:
            configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable, "recursion_limit": self.recursion_limit}

    @staticmethod
    def _checkpoint_id(snapshot: Any) -> str | None:
        config = getattr(snapshot, "config", None) or {}
        return _as_dict(config.get("configurable", {})).get("checkpoint_id")

    def _snapshot(self, thread_id: str, checkpoint_id: str | None = None):
        return self.graph.get_state(self._config_for(thread_id, checkpoint_id))

    def _input_for_run(
        self,
        user_message: str,
        thread_id: str,
        context: dict[str, Any] | None,
        *,
        turn: TurnContext | None = None,
    ) -> dict[str, Any]:
        messages: list[Any] = [HumanMessage(content=user_message)]
        if context:
            messages.append(
                AIMessage(
                    content=f"[context] {json.dumps(context, ensure_ascii=False)}",
                    additional_kwargs={
                        "agent_context_ephemeral": True,
                        "agent_context_run_id": "",
                    },
                )
            )
        # A checkpoint preserves conversation messages through its reducer, but
        # every other channel describes one run and must start clean.  Sending a
        # complete fresh delta prevents prior partial/error status, metrics, or
        # evidence from contaminating a later turn on the same thread.
        state = create_initial_state(
            thread_id=thread_id,
            run_id=uuid.uuid4().hex,
            max_steps=self.max_steps,
            max_cycles=self.max_steps,
        )
        state["messages"] = messages
        if context:
            messages[-1].additional_kwargs["agent_context_run_id"] = str(state["run_id"])
        if turn is not None:
            state["turn_state"] = turn.state.value
            state["turn_state_history"] = [item.value for item in turn.state_history]
            state["turn_restored_checkpoint_id"] = str(turn.restored_checkpoint_id or "")
        if bool(getattr(self.services, "agent_core_enabled", False)):
            state["project_id"] = str(getattr(self.services, "project_id", "") or "")
            state["root_run_id"] = str(state["run_id"])
            self.services.execution_run_store.create_user_run(
                root_run_id=str(state["root_run_id"]),
                project_id=str(state["project_id"]),
                thread_id=thread_id,
            )
            supplied = dict(context or {})
            requested = supplied.get("requested_tool_name")
            arguments = supplied.get("parsed_arguments")
            if requested is not None:
                state["requested_tool_name"] = str(requested)
            if isinstance(arguments, Mapping):
                state["parsed_arguments"] = dict(arguments)
            state["arguments_validated"] = bool(
                supplied.get("arguments_validated", False)
                and isinstance(arguments, Mapping)
            )
            confirmation_id = supplied.get("confirmation_id")
            if confirmation_id is not None:
                state["confirmation_id"] = str(confirmation_id)
        return state

    def _persist_session_memory(self, user_message: str, thread_id: str, snapshot: Any) -> None:
        """Persist only rule-extracted, checkpoint-traceable session memory."""

        runtime = getattr(self.services, "agent_core_runtime", None)
        if runtime is None or not bool(getattr(self.services, "agent_core_enabled", False)):
            return
        checkpoint_id = self._checkpoint_id(snapshot)
        if not checkpoint_id:
            return
        values = _as_dict(getattr(snapshot, "values", {}))
        messages = list(values.get("messages", []) or [])
        turn_count = sum(1 for item in messages if isinstance(item, HumanMessage))
        context_settings = getattr(self.settings, "context", None)
        max_tokens = max(1, int(getattr(context_settings, "max_tokens", 8192)))
        characters = sum(len(str(getattr(item, "content", "") or "")) for item in messages)
        context_ratio = min(1.0, characters / float(max_tokens * 4))
        normalized = user_message.strip().casefold()
        try:
            runtime.persist_after_checkpoint(
                thread_id=thread_id,
                user_message=user_message,
                checkpoint_id=checkpoint_id,
                turn_count=turn_count,
                context_ratio=context_ratio,
                task_state_changed=bool(
                    values.get("task_plan")
                    or values.get("plan_execution")
                    or values.get("review_result")
                ),
                user_corrected=normalized.startswith(("更正：", "correction:")),
            )
        except Exception as exc:
            self._runtime_memory_warnings.append(
                f"Session memory persistence unavailable: {exc}"
            )

    def _persist_assistant_message(self, thread_id: str, snapshot: Any) -> Any:
        """Store the synthesized answer in the message reducer for replay."""
        answer = str((_as_dict(getattr(snapshot, "values", {}))).get("final_answer", "") or "")
        if not answer or getattr(snapshot, "next", ()):
            return snapshot
        messages = list((_as_dict(getattr(snapshot, "values", {}))).get("messages", []) or [])
        if messages and getattr(messages[-1], "type", "") == "ai" and getattr(messages[-1], "content", "") == answer:
            return snapshot
        self.graph.update_state(self._config_for(thread_id), {"messages": [AIMessage(content=answer)]}, as_node="synthesize")
        return self._snapshot(thread_id)

    def _output_from_snapshot(self, snapshot: Any, *, thread_id: str, started: float,
                              forced_status: str | None = None, error_category: str | None = None,
                              error_message: str | None = None,
                              answer_override: str | None = None) -> AgentOutput:
        values = _as_dict(getattr(snapshot, "values", {}) if snapshot is not None else {})
        checkpoint_id = self._checkpoint_id(snapshot) if snapshot is not None else None
        status = forced_status or str(values.get("status") or ("interrupted" if getattr(snapshot, "next", ()) else "partial"))
        if error_category:
            status = "limit_reached" if error_category == "graph_recursion_limit" else "error"
        raw_runtime_warnings = list(getattr(self.services, "runtime_warnings", ()))
        structured_runtime_warnings = [
            dict(item)
            if isinstance(item, Mapping)
            else {"code": "runtime_warning", "message": str(item)}
            for item in raw_runtime_warnings
        ]
        warnings = [str(item) for item in raw_runtime_warnings]
        warnings.extend(self._runtime_memory_warnings)
        warnings.extend(str(item) for item in values.get("warnings", []) or [])
        metrics = dict(values.get("metrics", {}) or {})
        if structured_runtime_warnings:
            metrics["runtime_warnings"] = structured_runtime_warnings
        metrics.update({"run_id": str(values.get("run_id", "")), "thread_id": str(thread_id),
                        "checkpoint_id": checkpoint_id,
                        "model_name": str(getattr(self.services, "model_name", "") or self.model_name),
                        "endpoint": str(getattr(self.services, "endpoint", "") or self.llm_endpoint or ""),
                        "duration_ms": max(0.0, (time.perf_counter() - started) * 1000.0),
                        "total_duration_ms": float(metrics.get("total_duration_ms", 0.0) or 0.0),
                        "rag_hits": int(metrics.get("rag_hits", len(values.get("evidence", []) or [])) or 0),
                        "cycles": int(values.get("cycle_count", 0) or 0), "warnings": warnings,
                        "errors": list(values.get("errors", []) or [])})
        if error_category:
            metrics["error_category"] = error_category
        errors = list(values.get("errors", []) or [])
        if error_message:
            errors.append({"category": error_category or "agent_error", "message": error_message})
        try:
            intent = IntentType(str(values.get("intent", "general")))
        except Exception:
            intent = IntentType.GENERAL
        answer = str(values.get("final_answer", "") or "") if answer_override is None else str(answer_override)
        return AgentOutput(status=status, answer="" if status == "interrupted" else answer, intent=intent,
                           intent_confidence=float(values.get("intent_confidence", 0.0) or 0.0),
                           citations=list(values.get("citations", values.get("evidence", [])) or []),
                           tool_results=list(values.get("tool_results", []) or []), steps=int(values.get("step_count", values.get("recursion_depth", 0)) or 0),
                           cycle_triggers=int(values.get("cycle_count", 0) or 0), checkpoint_id=checkpoint_id,
                           warnings=warnings, errors=errors, metrics=metrics)

    def run(self, user_message: str, thread_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        thread_id = str(thread_id or uuid.uuid4().hex[:8])
        self._config = self._config_for(thread_id)
        started = time.perf_counter()
        turn = TurnContext(
            project_id=str(getattr(self.services, "project_id", "") or "uninitialized"),
            thread_id=thread_id,
            run_id=uuid.uuid4().hex,
            user_message=user_message,
            context=dict(context or {}),
        )
        holder: dict[str, Any] = {}

        def restore(item: TurnContext) -> TurnContext:
            try:
                previous = self._snapshot(thread_id)
                checkpoint_id = self._checkpoint_id(previous) if previous is not None else None
            except Exception:
                checkpoint_id = None
            if checkpoint_id:
                return item.model_copy(update={"restored_checkpoint_id": checkpoint_id})
            return item

        def compact(item: TurnContext) -> TurnContext:
            # The graph's first node performs the actual bounded assembly. The
            # explicit outer state makes that phase observable before RUN.
            try:
                previous = self._snapshot(thread_id)
                context_service = getattr(self.services, "context", None)
                if previous is not None and context_service is not None:
                    values = _as_dict(getattr(previous, "values", {}))
                    assembly = context_service.assemble(
                        messages=list(values.get("messages", []) or []),
                        tool_results=list(values.get("tool_results", []) or []),
                        thread_id=thread_id,
                        current_request=user_message,
                    )
                    holder["precompact_tokens"] = int(
                        getattr(assembly, "estimated_tokens", 0) or 0
                    )
                    holder["precompact_summary"] = str(
                        getattr(assembly, "summary", "") or ""
                    )
            except Exception as exc:
                holder["precompact_warning"] = str(exc)
            return item

        def build(item: TurnContext) -> TurnContext:
            return item

        def execute(item: TurnContext) -> TurnContext:
            self.graph.invoke(
                self._input_for_run(user_message, thread_id, context, turn=item),
                config=self._config,
            )
            holder["snapshot"] = self._snapshot(thread_id)
            return item

        def save(item: TurnContext) -> TurnContext:
            snapshot = holder.get("snapshot")
            if snapshot is None:
                return item
            snapshot = self._persist_assistant_message(thread_id, snapshot)
            self._persist_session_memory(user_message, thread_id, snapshot)
            holder["snapshot"] = snapshot
            return item

        def respond(item: TurnContext) -> TurnContext:
            snapshot = holder.get("snapshot")
            values = _as_dict(getattr(snapshot, "values", {})) if snapshot is not None else {}
            return item.with_response(values.get("final_answer", ""))

        def command_handler(_item: TurnContext, decision: CommandDecision) -> str | None:
            if decision.command != "model" or not decision.arguments:
                return None
            selector = getattr(getattr(self.services, "llm", None), "select", None)
            if not callable(selector):
                return "当前模型运行时未启用可切换的 provider registry。"
            try:
                selector(decision.arguments[0])
            except KeyError:
                return f"未配置模型 provider：{decision.arguments[0]}。"
            selected = getattr(self.services.llm, "model", decision.arguments[0])
            self.services = replace(
                self.services,
                model_name=str(selected),
                llm=self.services.llm,
            )
            # Graph node closures capture a ServiceBundle. Rebuild the small
            # graph so subsequent turns observe the selected provider name.
            self.graph = self._build_graph()
            return f"已切换到模型 provider：{decision.arguments[0]}。"

        try:
            AgentInput(user_message=user_message, thread_id=thread_id, max_steps=self.max_steps,
                       recursion_limit=max(self.recursion_limit, self.max_steps + 1), context=context or {})
            completed = self.turn_loop.run(
                turn,
                restore=restore,
                compact=compact,
                build=build,
                execute=execute,
                save=save,
                respond=respond,
                command_handler=command_handler,
            )
            if completed.shortcut:
                output = self._output_from_snapshot(
                    None,
                    thread_id=thread_id,
                    started=started,
                    forced_status="success",
                    answer_override=completed.response,
                )
            else:
                # Persist the complete outer lifecycle after RESPOND.  The
                # graph's synthesized checkpoint remains the source of truth;
                # this adds only primitive audit metadata to that checkpoint.
                snapshot = holder.get("snapshot")
                snapshot = self._snapshot(thread_id)
                # An interrupt-before checkpoint is deliberately left at its
                # pending node.  Updating it as ``synthesize`` would advance
                # the graph to END and erase the resumable ``next`` marker.
                if not getattr(snapshot, "next", ()):
                    self.graph.update_state(
                        self._config_for(thread_id),
                        {
                            "turn_state": completed.state.value,
                            "turn_state_history": [state.value for state in completed.state_history],
                            "turn_restored_checkpoint_id": str(
                                completed.restored_checkpoint_id or ""
                            ),
                        },
                        as_node="synthesize",
                    )
                    snapshot = self._snapshot(thread_id)
                output = self._output_from_snapshot(
                    snapshot,
                    thread_id=thread_id,
                    started=started,
                    forced_status="interrupted" if getattr(snapshot, "next", ()) else None,
                )
            output = output.model_copy(
                update={
                    "metrics": {
                        **dict(output.metrics),
                        "turn_state": completed.state.value,
                        "turn_state_history": [state.value for state in completed.state_history],
                        "turn_command": completed.command,
                        "turn_shortcut": completed.shortcut,
                        "precompact_tokens": holder.get("precompact_tokens", 0),
                        "precompact_summary": holder.get("precompact_summary", ""),
                        "precompact_warning": holder.get("precompact_warning", ""),
                    }
                }
            )
            return output
        except GraphRecursionError as exc:
            try: snapshot = self._snapshot(thread_id)
            except Exception: snapshot = None
            return self._output_from_snapshot(snapshot, thread_id=thread_id, started=started,
                                              error_category="graph_recursion_limit", error_message=str(exc))
        except OSError as exc:
            return self._output_from_snapshot(None, thread_id=thread_id, started=started,
                                              error_category="checkpoint_write_error", error_message=str(exc))
        except Exception as exc:
            try: snapshot = self._snapshot(thread_id)
            except Exception: snapshot = None
            return self._output_from_snapshot(snapshot, thread_id=thread_id, started=started,
                                              error_category="agent_error", error_message=str(exc))

    def resume_execution(self, thread_id: str) -> AgentOutput:
        started = time.perf_counter()
        try:
            snapshot = self._snapshot(thread_id)
            if snapshot is None or not getattr(snapshot, "next", ()):
                return self._output_from_snapshot(snapshot, thread_id=thread_id, started=started,
                                                  error_category="resume_unavailable", error_message="no pending graph execution")
            self.graph.invoke(None, config=self._config_for(thread_id))
            snapshot = self._persist_assistant_message(thread_id, self._snapshot(thread_id))
            return self._output_from_snapshot(snapshot, thread_id=thread_id, started=started)
        except GraphRecursionError as exc:
            return self._output_from_snapshot(self._snapshot(thread_id), thread_id=thread_id, started=started,
                                              error_category="graph_recursion_limit", error_message=str(exc))
        except OSError as exc:
            return self._output_from_snapshot(None, thread_id=thread_id, started=started,
                                              error_category="checkpoint_write_error", error_message=str(exc))
        except Exception as exc:
            return self._output_from_snapshot(self._snapshot(thread_id), thread_id=thread_id, started=started,
                                              error_category="agent_error", error_message=str(exc))

    def resume(self, thread_id: str) -> bool:
        try: return bool(self._snapshot(thread_id))
        except Exception: return False

    def stream(self, user_message: str, thread_id: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        thread_id = str(thread_id or uuid.uuid4().hex[:8])
        yield from self.graph.stream(self._input_for_run(user_message, thread_id, None), config=self._config_for(thread_id),
                                     stream_mode="values")

    def stream_text(self, user_message: str, thread_id: Optional[str] = None) -> Generator[str, None, None]:
        """Return a verified answer through the streaming-compatible facade.

        Token streaming directly from the model would bypass retrieval,
        grounding, cycle detection and the aerospace safety checks.  Until the
        graph supports node-level incremental synthesis, stream the completed
        graph result as one verified chunk instead.
        """
        yield self.run(user_message, thread_id=thread_id).answer

    def list_conversations(self) -> List[str]:
        if str(self.checkpoint_backend).lower() == "sqlite":
            return list_saved_threads(self._checkpoint_db_path)
        found: set[str] = set()
        try:
            for item in self.checkpointer.list(None):
                tid = _as_dict(getattr(item, "config", {})).get("configurable", {}).get("thread_id")
                if tid: found.add(str(tid))
        except Exception: pass
        return sorted(found)

    def _resolve_state_config(self, thread_or_config: Any, checkpoint_id: str | None = None):
        if isinstance(thread_or_config, Mapping):
            return thread_or_config
        return self._config_for(str(thread_or_config), checkpoint_id)

    def get_conversation_state(self, thread_id: str | Mapping[str, Any], checkpoint_id: str | None = None):
        if isinstance(thread_id, Mapping):
            return self.graph.get_state(thread_id)
        return self._snapshot(thread_id, checkpoint_id)

    def get_state(self, thread_id: str | Mapping[str, Any], checkpoint_id: str | None = None):
        return self.get_conversation_state(thread_id, checkpoint_id)

    def get_checkpoint_history(self, thread_id: str | Mapping[str, Any]) -> list[dict[str, Any]]:
        config = self._resolve_state_config(thread_id)
        snapshots = list(self.graph.get_state_history(config))
        result = []
        for index, snapshot in enumerate(snapshots):
            cp_id = self._checkpoint_id(snapshot)
            metadata = _as_dict(getattr(snapshot, "metadata", {}))
            result.append(CheckpointSummary({"checkpoint_id": cp_id, "created_at": getattr(snapshot, "created_at", None) or metadata.get("created_at") or cp_id or index,
                           "parent_checkpoint_id": self._checkpoint_id(getattr(snapshot, "parent_config", None)),
                           "values": getattr(snapshot, "values", {}), "next": tuple(getattr(snapshot, "next", ()) or ()),
                           "metadata": metadata, "snapshot": snapshot}))
        return result

    def get_state_history(self, thread_id: str | Mapping[str, Any]):
        return self.get_checkpoint_history(thread_id)

    def replay_checkpoint(self, thread_id: str, checkpoint_id: str | None = None):
        return self.get_conversation_state(thread_id, checkpoint_id)

    def replay_from_checkpoint(self, thread_id: str, checkpoint_id: str | None = None):
        return self.replay_checkpoint(thread_id, checkpoint_id)

    replay = replay_checkpoint

    def fork_from_checkpoint(self, checkpoint_id: str, *, new_thread_id: str, source_thread_id: str | None = None):
        source = source_thread_id
        if source is None:
            for candidate in self.list_conversations():
                if any(item.get("checkpoint_id") == checkpoint_id for item in self.get_checkpoint_history(candidate)):
                    source = candidate; break
        if source is None: raise ValueError(f"checkpoint not found: {checkpoint_id}")
        checkpoint_values = dict(self.get_conversation_state(source, checkpoint_id).values)
        values = dict(create_initial_state(thread_id=new_thread_id, run_id=uuid.uuid4().hex,
                                           max_steps=self.max_steps, max_cycles=self.max_steps))
        values.update(checkpoint_values)
        values.update(thread_id=new_thread_id, run_id=uuid.uuid4().hex, status="", final_answer="", is_complete=False)
        self.graph.invoke(values, config=self._config_for(new_thread_id))
        return self.get_conversation_state(new_thread_id)

    def _evolution_no_op(self, thread_id: str | None = None, reason: str = "no verifiable proposal") -> dict[str, Any]:
        return {"status": "no_op", "no_op": True, "thread_id": thread_id, "reason": reason}

    def _validate_evolution_approval(self, proposal: Any) -> None:
        """Require an explicit approval gate before file mutations."""
        validator = getattr(self.services, "safety", None)
        if validator is None:
            validator = SafetyValidator(approval_gate=getattr(self.services, "approval_gate", None))
        validator.validate_evolution_write(proposal)

    @staticmethod
    def _reject_boolean_evolution_approval(kwargs: Mapping[str, Any]) -> None:
        if "human_approved" in kwargs or "approved" in kwargs:
            raise ApprovalRequired(
                "boolean approval flags are not accepted for evolution writes"
            )

    def _proposal_from_checkpoint(self, thread_id: str) -> Any:
        if self.llm is None:
            return None
        history = self.get_checkpoint_history(thread_id)
        if not history:
            return None
        latest = history[0]
        checkpoint_id = latest.get("checkpoint_id")
        values = latest.get("values", {})
        # Checkpoint values may contain message objects; default=str keeps the
        # request bounded and serializable without inventing a change.
        context = json.dumps(_as_dict(values), ensure_ascii=False, default=str, sort_keys=True)
        prompt = (
            "Return exactly one JSON object, with no Markdown or commentary. "
            "The object must match EvolutionProposal fields: thread_id, run_id, "
            "checkpoint_id, rationale, changes, source, unfinished_items, "
            "required_validations. changes may only contain operation/path/content "
            "and paths must stay under knowledge, memory, evolved_skills, or "
            "workflows/evolved. If no safe change is justified, return changes [].\n"
            f"thread_id={thread_id}\ncheckpoint_id={checkpoint_id}\ncheckpoint={context}"
        )
        try:
            if hasattr(self.llm, "chat"):
                try:
                    response = self.llm.chat(
                        prompt,
                        system_prompt="You produce strict JSON proposals only.",
                        temperature=0.0,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                except TypeError:
                    response = self.llm.chat(
                        prompt,
                        system_prompt="You produce strict JSON proposals only.",
                        temperature=0.0,
                    )
            elif hasattr(self.llm, "invoke"):
                response = self.llm.invoke(prompt)
            elif callable(self.llm):
                response = self.llm(prompt)
            else:
                return None
        except Exception:
            return None
        if hasattr(response, "content"):
            response = response.content
        elif isinstance(response, Mapping) and "content" in response:
            response = response["content"]
        return parse_llm_proposal(response)

    def evolve(self, proposal: Any = None, **kwargs: Any):
        """Apply an explicit proposal, or derive one from a thread checkpoint."""
        self._reject_boolean_evolution_approval(kwargs)
        if isinstance(proposal, str) and "thread_id" not in kwargs:
            explicit = parse_llm_proposal(proposal)
            if explicit is not None:
                self._validate_evolution_approval(explicit)
                return self.evolution_service.apply(explicit, **kwargs)
            thread_id = proposal
            parsed = self._proposal_from_checkpoint(thread_id)
            if parsed is None or not parsed.changes:
                return self._evolution_no_op(thread_id)
            self._validate_evolution_approval(parsed)
            return self.evolution_service.apply(parsed, **kwargs)
        if proposal is None:
            thread_id = kwargs.pop("thread_id", None)
            parsed = self._proposal_from_checkpoint(str(thread_id)) if thread_id else None
            if parsed is None or not parsed.changes:
                return self._evolution_no_op(thread_id)
            proposal = parsed
        self._validate_evolution_approval(proposal)
        return self.evolution_service.apply(proposal, **kwargs)

    def evolve_due(self, proposal: Any = None, **kwargs: Any):
        """Apply one eligible, explicitly approved proposal at most once.

        A string proposal is interpreted as a checkpoint thread identifier.  It
        is first converted into a strict proposal by the local model, then
        routed through the same approval gate used by ``evolve``.  This keeps
        scheduler entry points from bypassing the human-write boundary.
        """
        self._reject_boolean_evolution_approval(kwargs)
        thread_id = kwargs.pop("thread_id", None)
        if isinstance(proposal, str):
            thread_id = proposal
            proposal = self._proposal_from_checkpoint(thread_id)
        elif proposal is None and thread_id:
            proposal = self._proposal_from_checkpoint(str(thread_id))
        if proposal is None:
            return self._evolution_no_op(str(thread_id) if thread_id else None)
        self._validate_evolution_approval(proposal)
        return self.evolution_service.evolve_due(proposal, **kwargs)

    def get_evolution_summary(self) -> Dict[str, Any]: return self.evolution.get_evolution_summary()
    def get_available_tools(self) -> List[str]:
        catalog = getattr(self.services, "core_tool_catalog", None)
        if catalog is not None and hasattr(catalog, "executable_names"):
            return list(catalog.executable_names())
        return list(self.available_tools.keys())
    def reset(self) -> None: self._config = self._config_for(uuid.uuid4().hex[:8])

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            closer = getattr(getattr(self.services, "agent_core_runtime", None), "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
            self._close_gateway()
            self._close_checkpointer()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()
    def __del__(self):
        try: self.close()
        except Exception: pass
    def __repr__(self) -> str: return f"<LangGraphAerospaceAgent model={self.model_name} mode={self.mode} checkpointer={type(self.checkpointer).__name__}>"


class SimpleLLMClient:
    """Tiny OpenAI-compatible HTTP client retained for existing callers."""
    def __init__(self, endpoint: str = "http://127.0.0.1:8000/v1", model: str = "qwen3-vl", api_key: str = "not-needed", timeout: float = 60.0):
        self.endpoint, self.model, self.api_key, self.timeout = endpoint.rstrip("/"), model, api_key, timeout
    def chat(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024, temperature: float = 0.7,
             chat_template_kwargs: dict[str, Any] | None = None) -> str:
        import urllib.request
        payload = {"model": self.model, "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature, "stream": False}
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        req = urllib.request.Request(f"{self.endpoint}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response: return json.loads(response.read().decode())["choices"][0]["message"]["content"]
    def stream_chat(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024, temperature: float = 0.7,
                    chat_template_kwargs: dict[str, Any] | None = None):
        import urllib.request
        payload = {"model": self.model, "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature, "stream": True}
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        req = urllib.request.Request(f"{self.endpoint}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            for line in response:
                text = line.decode().strip()
                if text.startswith("data: ") and text != "data: [DONE]":
                    try: content = json.loads(text[6:]).get("choices", [{}])[0].get("delta", {}).get("content", "")
                    except Exception: content = ""
                    if content: yield content
    def is_available(self) -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.endpoint}/models", timeout=5) as response: return response.status == 200
        except Exception: return False


def create_agent(llm_endpoint: str = "http://127.0.0.1:8000/v1", model_name: str = "qwen3-vl", checkpoint_backend: str = "sqlite", mode: str = "full", **kwargs) -> LangGraphAerospaceAgent:
    return LangGraphAerospaceAgent(llm_endpoint=llm_endpoint, model_name=model_name, checkpoint_backend=checkpoint_backend, mode=mode, **kwargs)

__all__ = ["LangGraphAerospaceAgent", "SimpleLLMClient", "create_agent"]
