#!/usr/bin/env python3
"""Command line boundary for the local-first LangGraph aerospace agent.

The command intentionally has no side effects outside the selected workspace
and never starts an LLM process.  Integrations can use ``--json`` to receive
one JSON document on stdout; human diagnostics are written to stderr.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_LLM_ENDPOINT = "http://127.0.0.1:8000/v1"
# Historical names retained for scripts that imported the launcher constants.
LLM_ENDPOINT = DEFAULT_LLM_ENDPOINT
LLM_MODEL = "qwythos"


def _jsonable(value: Any) -> Any:
    """Convert Pydantic models, Paths and LangGraph values to JSON data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except TypeError:
            dumped = value.model_dump()
        return _jsonable(dumped)
    if hasattr(value, "content") and not isinstance(value, (bytes, bytearray)):
        return _jsonable(getattr(value, "content"))
    return str(value)


def _emit(payload: Any, *, as_json: bool, exit_code: int = 0) -> int:
    """Emit exactly one document in JSON mode and return the process code."""
    data = _jsonable(payload)
    if exit_code and isinstance(data, dict) and data.get("error_code"):
        print(f"{data['error_code']}: {data.get('message', '')}", file=sys.stderr)
    if as_json:
        print(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2))
    elif isinstance(data, str):
        print(data)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return int(exit_code)


def _error(code: str, message: str, *, details: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error_code": code, "message": message}
    if details is not None:
        payload["details"] = _jsonable(details)
    return payload


def _load_settings(workspace: Path, config: str | None):
    """Load settings once, anchoring every writable path to *workspace*."""
    from aerospace_agent.langgraph_agent.config import AgentSettings, load_settings

    if not config:
        return load_settings(workspace=workspace)
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = workspace / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    import yaml

    mapping = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(mapping, dict):
        raise ValueError("LangGraph configuration must be a mapping")
    return AgentSettings.from_mapping(mapping, workspace=workspace)


def _knowledge_service(settings: Any, workspace: Path):
    from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

    return RuntimeServicesFactory(settings, project_root=workspace).create_knowledge_service()


def _seed_if_needed(service: Any) -> None:
    """Ensure the deterministic six-page seed corpus exists before querying."""
    pages = service.wiki_root / "orbital-dynamics"
    if len(list(pages.glob("*.md"))) < len(service.pages):
        service.initialize_seed_wiki()


def _create_agent(settings: Any, workspace: Path, *, mock: bool, check_endpoint: bool = True):
    from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent
    from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

    factory_parameters = inspect.signature(RuntimeServicesFactory).parameters
    factory_options: dict[str, Any] = {"project_root": workspace}
    optional = {
        "allow_degraded_mcp": bool(mock),
        "mock_llm": bool(mock),
        "check_llm_endpoint": bool(check_endpoint),
    }
    factory_options.update({key: value for key, value in optional.items() if key in factory_parameters})
    runtime = RuntimeServicesFactory(settings, **factory_options).create()
    _seed_if_needed(runtime.knowledge)
    return LangGraphAerospaceAgent(
        llm_endpoint="" if mock else None,
        services=runtime.bundle,
        checkpoint_backend=str(settings.checkpoint.backend),
        checkpoint_db_path=settings.checkpoint.path,
        evolution_db_path=settings.evolution.data_dir / "engine.json",
        max_steps=int(settings.runtime.max_steps),
        max_recursion_depth=int(settings.runtime.recursion_limit),
        cycle_max_repeats=int(settings.runtime.cycle_max_repeats),
        mode="full",
        settings=settings,
    )


def _schema_export(workspace: Path) -> dict[str, Any]:
    from aerospace_agent.langgraph_agent.schema import export_json_schemas

    # Filenames are versioned protocol identifiers.  Keep the model name
    # separate from the on-disk name so schema class refactors do not silently
    # change the integration contract.
    names = (
        ("AgentInput", "agent-input-v1.json"),
        ("AgentOutput", "agent-output-v1.json"),
        ("Decision", "decision-v1.json"),
        ("EvidenceItem", "evidence-item-v1.json"),
        ("ToolCallRequest", "tool-call-request-v1.json"),
        ("ToolCallResponse", "tool-call-response-v1.json"),
        ("EvolutionProposal", "evolution-proposal-v1.json"),
        ("EvolutionRecord", "evolution-record-v1.json"),
        ("ValidationResult", "validation-result-v1.json"),
    )
    destination = workspace / "schemas" / "langgraph_agent"
    destination.mkdir(parents=True, exist_ok=True)
    schemas = export_json_schemas()
    files: list[str] = []
    tool_catalog_filename = "core-tool-catalog-v1.json"
    expected_files = {filename for _, filename in names} | {tool_catalog_filename}
    # Migrate the pre-v1 CamelCase names and the placeholder marker when a
    # workspace was initialized by an earlier launcher build.
    stale_names = {
        ".gitkeep",
        "AgentInput.json", "AgentOutput.json", "Decision.json", "EvidenceItem.json",
        "ToolCallRequest.json", "ToolCallResponse.json", "EvolutionProposal.json",
        "EvolutionRecord.json", "ValidationResult.json",
    }
    for stale in stale_names:
        stale_path = destination / stale
        if stale_path.name not in expected_files and stale_path.exists():
            stale_path.unlink()
    for name, filename in names:
        if name not in schemas:
            raise KeyError(f"schema is not exported: {name}")
        path = destination / filename
        path.write_text(
            json.dumps(schemas[name], ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        files.append(str(path.resolve()))
    from aerospace_agent.langgraph_agent.agent_core.tools.system import (
        CoreToolServices,
        build_core_tool_catalog,
    )

    tool_catalog_path = destination / tool_catalog_filename
    tool_catalog_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tools": build_core_tool_catalog(
                    workspace,
                    CoreToolServices(),
                ).definitions(),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    files.append(str(tool_catalog_path.resolve()))
    return {"files": files, "count": len(files)}


def _graph_payload(service: Any, output: str | None, settings: Any, workspace: Path) -> dict[str, Any]:
    from aerospace_agent.langgraph_agent.services.graph_export import export_graph

    _seed_if_needed(service)
    target = Path(output) if output else Path(settings.knowledge.graph_output)
    if not target.is_absolute():
        target = workspace / target
    target = target.resolve()
    if not target.is_relative_to(workspace):
        raise ValueError("knowledge graph output must stay inside workspace")
    result = export_graph(service, target)
    html_path = str(Path(result.html_path).resolve())
    json_path = str(Path(result.json_path).resolve())
    # Keep both explicit path names and short aliases for older integrations.
    return {"html_path": html_path, "json_path": json_path, "html": html_path, "json": json_path}


def _checkpoint_payload(agent: Any, thread_id: str) -> dict[str, Any]:
    history = []
    for item in agent.get_checkpoint_history(thread_id):
        history.append(
            {
                "checkpoint_id": item.get("checkpoint_id"),
                "created_at": item.get("created_at"),
                "parent_checkpoint_id": item.get("parent_checkpoint_id"),
                "next": list(item.get("next", ())),
                "metadata": _jsonable(item.get("metadata", {})),
            }
        )
    return {"thread_id": thread_id, "checkpoints": history, "count": len(history)}


def _evolve_payload(
    service: Any,
    thread_id: str,
    *,
    agent: Any = None,
    human_approved: bool = False,
) -> dict[str, Any]:
    """Derive a proposal from the latest checkpoint; never submit an empty one."""
    if agent is None:
        return {"thread_id": str(thread_id), "status": "no_op", "no_op": True,
                "reason": "no verifiable proposal"}
    try:
        approval_kwargs = {"human_approved": True} if human_approved else {}
        result = agent.evolve(str(thread_id), **approval_kwargs)
    except Exception as exc:
        if exc.__class__.__name__ == "ApprovalRequired":
            return {
                "thread_id": str(thread_id),
                "status": "approval_required",
                "no_op": True,
                "reason": str(exc),
            }
        raise
    payload = {"thread_id": str(thread_id), "result": _jsonable(result)}
    if isinstance(result, dict):
        payload.update({key: result[key] for key in ("status", "no_op", "reason") if key in result})
        if result.get("evolution_id"):
            payload["evolution_id"] = result["evolution_id"]
    else:
        payload.update({"status": getattr(result, "status", "committed"),
                        "evolution_id": getattr(result, "evolution_id", None)})
    return payload


def _created_at_minutes_ago(value: Any) -> float:
    """Return elapsed wall time for a LangGraph checkpoint timestamp."""
    if not value:
        return 0.0
    try:
        text = str(value).replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(text)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds() / 60.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _evolve_due_payload(
    settings: Any,
    workspace: Path,
    *,
    thread_id: str | None,
    mock: bool,
    human_approved: bool,
) -> dict[str, Any]:
    """Run an eligibility check against a real checkpoint, never a dummy one."""
    agent = _create_agent(settings, workspace, mock=mock, check_endpoint=False)
    try:
        selected = str(thread_id) if thread_id else ""
        conversations = agent.list_conversations()
        if not selected:
            if not conversations:
                return {"eligible": False, "changed": [], "failed": [], "no_op": True,
                        "status": "no_op", "reason": "no_checkpoint_thread"}
            # A deterministic fallback is preferable to silently fabricating a
            # scheduler context.  Operators should pass --evolve-due THREAD
            # where multiple active conversations exist.
            selected = conversations[-1]
        history = agent.get_checkpoint_history(selected)
        if not history:
            return {"thread_id": selected, "eligible": False, "changed": [], "failed": [],
                    "no_op": True, "status": "no_op", "reason": "checkpoint_not_found"}
        latest = history[0]
        idle_minutes = _created_at_minutes_ago(latest.get("created_at"))
        turn_count = sum(
            1
            for message in (latest.get("values", {}) or {}).get("messages", []) or []
            if getattr(message, "type", None) in {"human", "user"}
            or (isinstance(message, dict) and str(message.get("type", message.get("role", ""))).lower() in {"human", "user"})
        )
        eligibility = agent.evolution_service.policy.is_due(
            idle_minutes=idle_minutes,
            turn_count=turn_count,
        )
        base = {
            "thread_id": selected,
            "eligible": bool(eligibility.due),
            "idle_minutes": round(idle_minutes, 3),
            "turn_count": turn_count,
            "changed": [],
            "failed": [],
        }
        if not eligibility.due:
            return {**base, "no_op": True, "status": "no_op", "reason": eligibility.reason}
        try:
            approval_kwargs = {"human_approved": True} if human_approved else {}
            record = agent.evolve_due(
                selected,
                idle_minutes=idle_minutes,
                turn_count=turn_count,
                **approval_kwargs,
            )
        except Exception as exc:
            if exc.__class__.__name__ == "ApprovalRequired":
                return {**base, "no_op": True, "status": "approval_required", "reason": str(exc)}
            raise
        if isinstance(record, dict):
            return {**base, "no_op": True, "status": record.get("status", "no_op"),
                    "reason": record.get("reason", "no verifiable proposal")}
        return {**base, "changed": [record.evolution_id] if record else [],
                "no_op": record is None, "status": "no_op" if record is None else record.status}
    finally:
        agent.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first LangGraph aerospace agent")
    parser.add_argument("--workspace", default=None, help="workspace root (default: current directory)")
    parser.add_argument("--config", default=None, help="settings YAML path")
    parser.add_argument("--json", action="store_true", help="emit one JSON document")
    parser.add_argument("--mock", action="store_true", help="run without a Qwen endpoint")
    parser.add_argument("--thread", default="default", help="conversation thread id")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--repl", action="store_true")
    actions.add_argument("--task", metavar="TEXT")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="stream the verified result of --task as deterministic chunks",
    )
    actions.add_argument("--tools", action="store_true")
    actions.add_argument("--init-knowledge", action="store_true")
    actions.add_argument("--knowledge-status", action="store_true")
    actions.add_argument("--knowledge-graph", metavar="PATH")
    actions.add_argument("--checkpoint-history", metavar="THREAD")
    actions.add_argument("--evolve", metavar="THREAD")
    parser.add_argument(
        "--approve-evolution",
        action="store_true",
        help="explicitly approve the selected evolution write",
    )
    actions.add_argument(
        "--evolve-due", metavar="THREAD", nargs="?", const="__auto__",
        help="evaluate one checkpoint thread for an approved, due evolution",
    )
    actions.add_argument("--rollback", metavar="ID")
    actions.add_argument("--export-schemas", action="store_true")
    actions.add_argument("--init-project", action="store_true")
    actions.add_argument("--project-memory-status", action="store_true")
    actions.add_argument("--project-memory-reindex", action="store_true")
    actions.add_argument(
        "--schedule-reminder",
        nargs=2,
        metavar=("DUE_AT", "MESSAGE"),
        help="create an internal SQLite reminder",
    )
    actions.add_argument("--schedule-list", action="store_true")
    actions.add_argument("--schedule-cancel", metavar="JOB_ID")
    actions.add_argument("--schedule-run-due", action="store_true")
    parser.add_argument(
        "--schedule-version",
        type=int,
        default=None,
        help="optimistic version required by --schedule-cancel",
    )
    return parser


def _scheduler_service(workspace: Path):
    """Build the project-local internal queue without any OS scheduler."""

    from aerospace_agent.langgraph_agent.agent_core.approval import (
        CapabilityApprovalLedger,
        CapabilityApprovalVerifier,
    )
    from aerospace_agent.langgraph_agent.agent_core.project_memory import (
        ProjectIdentityService,
    )
    from aerospace_agent.langgraph_agent.agent_core.rag_gate import ExecutionRunStore
    from aerospace_agent.langgraph_agent.agent_core.scheduler import SchedulerService
    from aerospace_agent.langgraph_agent.agent_core.workflows import WorkflowRegistry

    project = ProjectIdentityService(workspace)
    status = project.status()
    if status.state != "ready" or status.project_id is None:
        raise RuntimeError("project_not_initialized")
    data = workspace / "data" / "langgraph"
    ledger = CapabilityApprovalLedger(
        data / "capability_approvals.sqlite3",
        trusted_public_keys={},
    )
    workflows = WorkflowRegistry(
        data / "workflows.sqlite3",
        approval_verifier=CapabilityApprovalVerifier(ledger),
    )
    runs = ExecutionRunStore(data / "execution_runs.sqlite3")
    return status.project_id, SchedulerService(
        data / "scheduler.sqlite3",
        workflow_registry=workflows,
        execution_run_store=runs,
    )


def _schedule_action_payload(args: Any, workspace: Path, action: str) -> dict[str, Any]:
    project_id, scheduler = _scheduler_service(workspace)
    if action == "schedule_reminder":
        due_at, message = args.schedule_reminder
        job = scheduler.create_reminder(
            project_id=project_id,
            thread_id=str(args.thread),
            due_at=due_at,
            message=message,
        )
        return {"status": "scheduled", "job": job}
    if action == "schedule_list":
        return {
            "status": "ok",
            "jobs": scheduler.list_jobs(
                project_id=project_id,
                thread_id=str(args.thread),
            ),
        }
    if action == "schedule_cancel":
        if args.schedule_version is None:
            raise ValueError("--schedule-version is required by --schedule-cancel")
        job = scheduler.cancel(
            str(args.schedule_cancel),
            expected_version=int(args.schedule_version),
        )
        if job is None:
            raise ValueError("scheduled job/version is not cancellable")
        return {"status": job.status, "job": job}

    worker_id = f"cli:{os.getpid()}"
    job = scheduler.claim_due(worker_id=worker_id)
    if job is None:
        return {"status": "idle", "job": None}
    running = scheduler.mark_running(
        job.job_id,
        expected_version=job.version,
        worker_id=worker_id,
    )
    payload = scheduler.get_payload(running.payload_id)
    if payload.body.get("kind") != "reminder":
        failed = scheduler.mark_failed(
            running.job_id,
            expected_version=running.version,
            worker_id=worker_id,
            retryable=False,
        )
        return {
            "status": failed.status,
            "job": failed,
            "reason": "no approved workflow executor is bound to the CLI worker",
        }
    succeeded = scheduler.mark_succeeded(
        running.job_id,
        expected_version=running.version,
        worker_id=worker_id,
    )
    return {
        "status": succeeded.status,
        "job": succeeded,
        "message": payload.body["message"],
    }


# Compatibility helpers retained for callers of the original launcher.  They
# deliberately delegate to the same no-auto-spawn implementation used by the
# CLI and do not print diagnostics.
def check_llm(endpoint: str = DEFAULT_LLM_ENDPOINT, model: str = "qwythos") -> bool:
    from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

    return RuntimeServicesFactory.check_llm_endpoint(endpoint, model)


def create_agent(mock: bool = False, *, workspace: str | Path | None = None):
    root = Path(workspace or Path.cwd()).resolve()
    settings = _load_settings(root, None)
    return _create_agent(settings, root, mock=mock)


def list_tools() -> list[dict[str, Any]]:
    from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

    return RuntimeServicesFactory.list_tool_definitions()


def run_task(agent: Any, task: str, *, thread_id: str = "default") -> Any:
    """Compatibility wrapper for the former launcher helper."""
    return agent.run(task, thread_id=thread_id)


def run_repl(agent: Any, *, thread_id: str = "default") -> None:
    """Small compatibility REPL used by older scripts."""
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if line.lower() in {"/q", "/quit", "/exit"}:
            return
        if line:
            print(agent.run(line, thread_id=thread_id).answer)


def run_stream_repl(agent: Any, *, thread_id: str = "default") -> None:
    """Compatibility streaming REPL."""
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if line.lower() in {"/q", "/quit", "/exit"}:
            return
        if line:
            print("".join(agent.stream_text(line, thread_id=thread_id)))


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    as_json_hint = "--json" in raw_argv
    try:
        args = parser.parse_args(raw_argv)
    except SystemExit as exc:
        # Argparse has already sent its usage diagnostics to stderr.  Keep
        # the stdout contract for machine callers that requested JSON.
        code = int(exc.code or 0)
        if code and as_json_hint:
            return _emit(_error("CLI_USAGE", "invalid command-line arguments"), as_json=True, exit_code=code)
        return code
    workspace = Path(args.workspace or Path.cwd()).resolve()
    as_json = bool(args.json)
    action = "stream" if args.stream and args.task is not None else next((name for name in (
        "task", "stream", "tools", "init_knowledge", "knowledge_status", "knowledge_graph",
        "checkpoint_history", "evolve", "evolve_due", "rollback", "export_schemas", "repl",
        "init_project", "project_memory_status", "project_memory_reindex",
        "schedule_reminder", "schedule_list", "schedule_cancel", "schedule_run_due",
    ) if getattr(args, name, False)), "repl")
    try:
        if action in {"init_project", "project_memory_status", "project_memory_reindex"}:
            from aerospace_agent.langgraph_agent.agent_core.project_memory import (
                ProjectIdentityService,
            )

            project_service = ProjectIdentityService(workspace)
            if action == "init_project":
                return _emit(project_service.initialize(), as_json=as_json)
            if action == "project_memory_status":
                return _emit(project_service.status(), as_json=as_json)
            status = project_service.status()
            if status.state != "ready":
                return _emit(
                    _error(
                        "PROJECT_NOT_INITIALIZED",
                        "project memory is not ready",
                        details=status,
                    ),
                    as_json=as_json,
                    exit_code=2,
                )
            return _emit(project_service.reindex(), as_json=as_json)

        if action in {
            "schedule_reminder",
            "schedule_list",
            "schedule_cancel",
            "schedule_run_due",
        }:
            try:
                payload = _schedule_action_payload(args, workspace, action)
            except RuntimeError as exc:
                if str(exc) == "project_not_initialized":
                    return _emit(
                        _error(
                            "PROJECT_NOT_INITIALIZED",
                            "initialize project memory before using the internal scheduler",
                        ),
                        as_json=as_json,
                        exit_code=2,
                    )
                raise
            except (KeyError, ValueError) as exc:
                return _emit(
                    _error("SCHEDULER_INVALID_REQUEST", str(exc)),
                    as_json=as_json,
                    exit_code=2,
                )
            return _emit(payload, as_json=as_json)

        settings = _load_settings(workspace, args.config)

        if action == "export_schemas":
            return _emit(_schema_export(workspace), as_json=as_json)
        if action == "tools":
            from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

            tools = RuntimeServicesFactory.list_tool_definitions()
            return _emit({"tools": tools, "count": len(tools)}, as_json=as_json)
        if action in {"init_knowledge", "knowledge_status", "knowledge_graph"}:
            service = _knowledge_service(settings, workspace)
            if action == "init_knowledge":
                summary = service.initialize_seed_wiki()
                return _emit(summary, as_json=as_json)
            if action == "knowledge_status":
                root = service.wiki_root / "orbital-dynamics"
                count = len(list(root.glob("*.md"))) if root.exists() else 0
                return _emit({"status": "ok", "wiki_pages": count, "pages": count,
                              "seed_pages": len(service.pages), "workspace": str(workspace)}, as_json=as_json)
            return _emit(_graph_payload(service, args.knowledge_graph, settings, workspace), as_json=as_json)
        if action == "evolve_due":
            return _emit(
                _evolve_due_payload(
                    settings,
                    workspace,
                    thread_id=None if args.evolve_due == "__auto__" else args.evolve_due,
                    mock=bool(args.mock),
                    human_approved=bool(args.approve_evolution),
                ),
                as_json=as_json,
            )
        if action == "rollback":
            from aerospace_agent.langgraph_agent.services.runtime import RuntimeServicesFactory

            service = RuntimeServicesFactory(settings, project_root=workspace).create_evolution_service()
            try:
                record = service.rollback(str(args.rollback))
            except (FileNotFoundError, OSError, ValueError) as exc:
                return _emit(_error("EVOLUTION_NOT_FOUND", str(exc)), as_json=as_json, exit_code=2)
            if record.status == "conflict":
                return _emit(
                    _error("EVOLUTION_CONFLICT", "rollback refused because a target changed", details=record),
                    as_json=as_json,
                    exit_code=2,
                )
            return _emit({"evolution_id": record.evolution_id, "status": record.status, "record": record}, as_json=as_json)
        if action == "evolve":
            # Proposal derivation treats an unavailable endpoint as a safe
            # no-op; unlike task execution it must not turn into a CLI error.
            agent = _create_agent(settings, workspace, mock=args.mock, check_endpoint=False)
            try:
                return _emit(
                    _evolve_payload(
                        agent.evolution_service,
                        str(args.evolve),
                        agent=agent,
                        human_approved=bool(args.approve_evolution),
                    ),
                    as_json=as_json,
                )
            finally:
                agent.close()
        if action == "checkpoint_history":
            # History is a storage query and must remain usable when Qwen is
            # offline; no model call is made while opening the checkpointer.
            agent = _create_agent(settings, workspace, mock=True)
            try:
                return _emit(_checkpoint_payload(agent, str(args.checkpoint_history)), as_json=as_json)
            finally:
                agent.close()
        if action in {"task", "stream", "repl"}:
            if action == "repl" and as_json:
                return _emit({"status": "ready", "thread_id": args.thread}, as_json=True)
            agent = _create_agent(settings, workspace, mock=args.mock)
            try:
                if action == "task":
                    result = agent.run(str(args.task), thread_id=args.thread)
                    status_value = getattr(result.status, "value", result.status)
                    return _emit(result, as_json=as_json, exit_code=0 if status_value == "success" else 1)
                if action == "stream" and args.task:
                    chunks = list(agent.stream_text(str(args.task), thread_id=args.thread))
                    return _emit({"thread_id": args.thread, "chunks": chunks, "answer": "".join(chunks)}, as_json=as_json)
                if as_json:
                    return _emit({"status": "ready", "thread_id": args.thread}, as_json=True)
                while True:
                    try:
                        line = input("> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if line.lower() in {"/q", "/quit", "/exit"}:
                        break
                    if line:
                        print(agent.run(line, thread_id=args.thread).answer)
                return 0
            finally:
                agent.close()
    except RuntimeError as exc:
        if "Qwen endpoint is unavailable" in str(exc):
            return _emit(_error("LLM_UNAVAILABLE", str(exc)), as_json=as_json, exit_code=2)
        return _emit(_error("CLI_ERROR", str(exc)), as_json=as_json, exit_code=2)
    except Exception as exc:
        if not as_json:
            print(f"error: {exc}", file=sys.stderr)
        return _emit(_error("CLI_ERROR", str(exc)), as_json=as_json, exit_code=2)


if __name__ == "__main__":
    raise SystemExit(main())
