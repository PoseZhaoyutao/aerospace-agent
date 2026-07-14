"""Preflighted, checkpointed execution of immutable :class:`TaskPlan` DAGs.

The executor deliberately has no handler/callable API.  A step is resolved by
``ExecutionRegistry`` and can only be invoked by ``ExecutionService``.  Plan
validation finishes before this module writes a run or step checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, ValidationError

from .artifacts import ArtifactCheckpointBinding, ArtifactStore

from .execution import (
    AuthorizedExecutor,
    ExecutionContext,
    ExecutionRegistry,
    ExecutionRequest,
    ExecutionService,
)
from .models import (
    ContractModel,
    CheckpointRef,
    CheckResult,
    DomainExecutionOutput,
    DomainReview,
    HandoffRecord,
    PlanExecutionState,
    PlanStep,
    PlanStepExecutionState,
    TaskPlan,
    ToolError,
    ToolResult,
)
from .planning import PlanExecutionVerifier, compute_task_plan_sha256


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class CanonicalMetadataVocabulary(ContractModel):
    """Exact identifiers published by the active SpaceBasicTools manifests."""

    quantity_units: set[str] = Field(default_factory=set)
    frame_ids: set[str] = Field(default_factory=set)
    time_systems: set[str] = Field(default_factory=set)


class DAGCheckpoint(ContractModel):
    checkpoint_id: str
    project_id: str
    thread_id: str
    root_run_id: str
    plan_id: str
    plan_sha256: str
    step_id: str
    phase: Literal["before", "after", "inspection_required", "inspection"]
    input_hash: str
    idempotency_key: str
    attempt: int
    status: str
    result: ToolResult | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    handoff_ids: list[str] = Field(default_factory=list)
    created_at: str


class DAGExecutionOutcome(ContractModel):
    status: Literal["completed", "partial", "blocked", "failed", "interrupted", "invalid_plan"]
    plan_id: str | None = None
    plan_sha256: str | None = None
    state: PlanExecutionState | None = None
    step_results: dict[str, ToolResult] = Field(default_factory=dict)
    reused_step_ids: list[str] = Field(default_factory=list)
    error: str | None = None


def _step_binding(step: PlanStep) -> tuple[str, str]:
    if step.executor_type in {"basic_tool", "space_basic_tool"}:
        return "tool", str(step.tool_name)
    if step.executor_type == "workflow":
        return "workflow", str(step.workflow_id)
    if step.executor_type == "domain_subgraph":
        return "domain", str(step.domain_subgraph)
    if step.executor_type == "capability_builder":
        return "capability_builder", str(step.capability_gap_id)
    return "human", step.capability


class CheckpointedDAGExecutor:
    """Execute an immutable plan with durable step state and exact resume keys."""

    _SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        database_path: str | Path,
        workspace_root: str | Path,
        registry: ExecutionRegistry,
        execution_service: ExecutionService,
        plan_verifier: PlanExecutionVerifier,
        metadata_vocabulary: CanonicalMetadataVocabulary,
        artifact_store: ArtifactStore | None = None,
        handoff_validation_checks: Mapping[str, Callable[[dict[str, Any]], bool]] | None = None,
    ) -> None:
        if not isinstance(registry, ExecutionRegistry):
            raise TypeError("registry must be ExecutionRegistry")
        if not isinstance(execution_service, ExecutionService):
            raise TypeError("execution_service must be ExecutionService")
        if execution_service._registry is not registry:
            raise ValueError("execution service and registry must share one trust boundary")
        if not isinstance(plan_verifier, PlanExecutionVerifier):
            raise TypeError("plan_verifier must be PlanExecutionVerifier")
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._workspace_root = Path(workspace_root).resolve()
        self._registry = registry
        self._execution_service = execution_service
        self._plan_verifier = plan_verifier
        self._vocabulary = CanonicalMetadataVocabulary.model_validate(
            metadata_vocabulary.model_dump(mode="python")
        )
        if artifact_store is not None and not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must be ArtifactStore")
        self._artifact_store = artifact_store
        self._handoff_validation_checks = dict(handoff_validation_checks or {})
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported DAG schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.executescript(
                    """
                    CREATE TABLE dag_runs (
                        plan_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        plan_sha256 TEXT NOT NULL,
                        status TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE dag_steps (
                        project_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        plan_id TEXT NOT NULL,
                        plan_sha256 TEXT NOT NULL,
                        step_id TEXT NOT NULL,
                        input_hash TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        status TEXT NOT NULL,
                        attempts INTEGER NOT NULL,
                        result_json TEXT,
                        artifact_ids_json TEXT NOT NULL DEFAULT '[]',
                        handoff_ids_json TEXT NOT NULL DEFAULT '[]',
                        last_checkpoint_id TEXT,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(project_id, thread_id, root_run_id, plan_id, step_id)
                    );
                    CREATE UNIQUE INDEX dag_step_idempotency_idx
                    ON dag_steps(project_id, thread_id, root_run_id, idempotency_key);
                    CREATE TABLE dag_checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        plan_id TEXT NOT NULL,
                        plan_sha256 TEXT NOT NULL,
                        step_id TEXT NOT NULL,
                        phase TEXT NOT NULL,
                        input_hash TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        attempt INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        result_json TEXT,
                        artifact_ids_json TEXT NOT NULL DEFAULT '[]',
                        handoff_ids_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX dag_checkpoint_plan_idx
                    ON dag_checkpoints(project_id, thread_id, root_run_id, plan_id, created_at);
                    CREATE TABLE dag_interruption_inspections (
                        project_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        plan_id TEXT NOT NULL,
                        step_id TEXT NOT NULL,
                        input_hash TEXT NOT NULL,
                        inspection_audit_id TEXT NOT NULL UNIQUE,
                        decision TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(project_id, thread_id, root_run_id, plan_id, step_id, input_hash)
                    );
                    """
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()
            elif version == 1:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "ALTER TABLE dag_steps ADD COLUMN artifact_ids_json TEXT NOT NULL DEFAULT '[]'"
                )
                connection.execute(
                    "ALTER TABLE dag_steps ADD COLUMN handoff_ids_json TEXT NOT NULL DEFAULT '[]'"
                )
                connection.execute(
                    "ALTER TABLE dag_checkpoints ADD COLUMN artifact_ids_json TEXT NOT NULL DEFAULT '[]'"
                )
                connection.execute(
                    "ALTER TABLE dag_checkpoints ADD COLUMN handoff_ids_json TEXT NOT NULL DEFAULT '[]'"
                )
                connection.execute("PRAGMA user_version = 2")
                connection.commit()

    def execute(
        self,
        plan: TaskPlan | Mapping[str, Any],
        *,
        project_id: str,
        thread_id: str,
        root_run_id: str,
        confirmation_ids: Mapping[str, str] | None = None,
    ) -> DAGExecutionOutcome:
        """Validate the whole plan, then execute its ready steps in topological order."""

        try:
            checked = self._preflight(
                plan,
                project_id=project_id,
                thread_id=thread_id,
                root_run_id=root_run_id,
            )
        except (ValueError, TypeError, ValidationError, KeyError, RuntimeError) as exc:
            raw_plan_id = plan.plan_id if isinstance(plan, TaskPlan) else plan.get("plan_id")
            raw_hash = plan.plan_sha256 if isinstance(plan, TaskPlan) else plan.get("plan_sha256")
            return DAGExecutionOutcome(
                status="invalid_plan",
                plan_id=str(raw_plan_id) if raw_plan_id is not None else None,
                plan_sha256=str(raw_hash) if raw_hash is not None else None,
                error=str(exc),
            )

        try:
            self._plan_verifier.register_plan(checked)
            self._ensure_run(checked)
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            return DAGExecutionOutcome(
                status="invalid_plan",
                plan_id=checked.plan_id,
                plan_sha256=checked.plan_sha256,
                error=str(exc),
            )
        confirmations = dict(confirmation_ids or {})
        step_results: dict[str, ToolResult] = {}
        reused: list[str] = []
        ordered = self._topological_steps(checked)

        for step in ordered:
            if any(
                dependency not in step_results
                or step_results[dependency].status != "success"
                for dependency in step.dependencies
            ):
                continue
            input_hash = self._input_hash(step, step_results)
            idempotency_key = f"{checked.plan_id}:{step.step_id}:{input_hash}"
            existing = self._load_step(checked, step.step_id)
            if (
                existing is not None
                and existing["input_hash"] == input_hash
                and existing["status"] == "completed"
                and existing["result_json"] is not None
            ):
                result = ToolResult.model_validate(json.loads(existing["result_json"]))
                if result.status == "success":
                    try:
                        self._validate_result_references(checked, step, result)
                        self._validate_target_handoffs(checked, step)
                    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                        step_results[step.step_id] = self._domain_failure(
                            step,
                            f"persisted domain reference validation failed: {exc}",
                        )
                        return self._outcome(checked, step_results, reused, "failed")
                    step_results[step.step_id] = result
                    reused.append(step.step_id)
                    continue

            if existing is not None and existing["status"] == "interrupted":
                prior = ToolResult.model_validate(json.loads(existing["result_json"]))
                if prior.recovery_class != "read_only":
                    inspection = self._load_inspection(checked, step.step_id, input_hash)
                    if inspection is None:
                        self._record_checkpoint(
                            checked,
                            step,
                            phase="inspection_required",
                            input_hash=input_hash,
                            idempotency_key=idempotency_key,
                            attempt=int(existing["attempts"]),
                            status="interrupted",
                            result=prior,
                        )
                        return self._outcome(checked, step_results, reused, "interrupted")
                    if inspection["decision"] != "retry":
                        step_results[step.step_id] = prior
                        return self._outcome(checked, step_results, reused, "blocked")

            attempt = 1 if existing is None else int(existing["attempts"]) + 1
            if attempt > step.max_attempts:
                if existing is not None and existing["result_json"]:
                    step_results[step.step_id] = ToolResult.model_validate(
                        json.loads(existing["result_json"])
                    )
                return self._outcome(checked, step_results, reused, "failed")

            before_id = self._record_checkpoint(
                checked,
                step,
                phase="before",
                input_hash=input_hash,
                idempotency_key=idempotency_key,
                attempt=attempt,
                status="running",
                result=None,
            )
            self._save_step(
                checked,
                step,
                input_hash=input_hash,
                idempotency_key=idempotency_key,
                status="running",
                attempts=attempt,
                result=None,
                checkpoint_id=before_id,
            )
            reserved_after_id = f"dag:{uuid4().hex}"
            try:
                if (
                    step.executor_type == "domain_subgraph"
                    and existing is not None
                    and existing["status"] == "running"
                    and existing["input_hash"] == input_hash
                    and existing["result_json"] is None
                    and self._artifact_store is not None
                ):
                    recovered = self._artifact_store.recover_exchange(
                        project_id=checked.project_id,
                        thread_id=checked.thread_id,
                        root_run_id=checked.root_run_id,
                        plan_id=checked.plan_id,
                        plan_sha256=checked.plan_sha256,
                        step_id=step.step_id,
                        source_snapshot=self._snapshot_for_capability(
                            checked, step.capability
                        ),
                        expected_target_step_ids=(
                            item.target_step_id
                            for item in checked.handoffs
                            if item.source_step_id == step.step_id
                        ),
                    )
                    if recovered is not None:
                        # The domain execution checkpoint is already durable in
                        # ExecutionService and the artifact transaction is
                        # append-only.  Reusing its exact reserved identity lets
                        # the idempotent execution result complete the missing
                        # DAG checkpoint without re-running the domain handler.
                        reserved_after_id = (
                            recovered.checkpoint_binding.checkpoint_id
                        )
                self._validate_target_handoffs(checked, step)
                request, context = self._execution_request(
                    checked,
                    step,
                    input_hash=input_hash,
                    attempt=attempt,
                    confirmation_id=confirmations.get(step.step_id),
                    domain_checkpoint_id=(
                        reserved_after_id
                        if step.executor_type == "domain_subgraph"
                        else None
                    ),
                )
                authorized = self._registry.resolve(request, context)
                result = (
                    self._execution_service.execute(authorized)
                    if isinstance(authorized, AuthorizedExecutor)
                    else authorized
                )
                if result.status == "success" and step.executor_type == "domain_subgraph":
                    result = self._persist_domain_result(
                        checked,
                        step,
                        result,
                        checkpoint_id=reserved_after_id,
                    )
            except (ValueError, TypeError, ValidationError, KeyError, RuntimeError) as exc:
                result = self._domain_failure(step, f"domain handoff validation failed: {exc}")
            status = self._step_status(result)
            after_id = self._record_checkpoint(
                checked,
                step,
                phase="after",
                input_hash=input_hash,
                idempotency_key=idempotency_key,
                attempt=attempt,
                status=status,
                result=result,
                checkpoint_id=reserved_after_id,
            )
            self._save_step(
                checked,
                step,
                input_hash=input_hash,
                idempotency_key=idempotency_key,
                status=status,
                attempts=attempt,
                result=result,
                checkpoint_id=after_id,
            )
            step_results[step.step_id] = result
            if result.status != "success":
                break

        status: Literal["completed", "partial", "blocked", "failed", "interrupted"]
        if len(step_results) == len(checked.steps) and all(
            result.status == "success" for result in step_results.values()
        ):
            status = "completed"
        elif any(result.status == "blocked" for result in step_results.values()):
            status = "blocked"
        elif any(result.status == "interrupted" for result in step_results.values()):
            status = "interrupted"
        elif any(result.status in {"failed", "timeout", "unavailable", "invalid_arguments"} for result in step_results.values()):
            status = "failed"
        else:
            status = "partial"
        return self._outcome(checked, step_results, reused, status)

    def record_interrupted_inspection(
        self,
        *,
        plan: TaskPlan,
        step_id: str,
        inspection_result: ToolResult,
        decision: Literal["retry", "blocked"],
    ) -> DAGCheckpoint:
        """Accept a write-state inspection only if it has a read-only execution audit."""

        checked = self._preflight(
            plan,
            project_id=plan.project_id,
            thread_id=plan.thread_id,
            root_run_id=plan.root_run_id,
        )
        step = next((item for item in checked.steps if item.step_id == step_id), None)
        if step is None:
            raise ValueError("inspection step is not in the plan")
        row = self._load_step(checked, step_id)
        if row is None or row["status"] != "interrupted" or not row["result_json"]:
            raise ValueError("step has no persisted interruption to inspect")
        prior = ToolResult.model_validate(json.loads(row["result_json"]))
        if prior.recovery_class == "read_only":
            raise ValueError("read-only interruption does not need write-state inspection")
        matching_audit = next(
            (
                item
                for item in self._registry.audit_records()
                if item["audit_id"] == inspection_result.audit_id
                and item["project_id"] == checked.project_id
                and item["thread_id"] == checked.thread_id
                and item["root_run_id"] == checked.root_run_id
                and item["status"] == "success"
                and item["recovery_class"] == "read_only"
            ),
            None,
        )
        if inspection_result.status != "success" or inspection_result.recovery_class != "read_only" or matching_audit is None:
            raise ValueError("inspection must be a successful read-only ExecutionService result")

        input_hash = str(row["input_hash"])
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO dag_interruption_inspections(
                    project_id, thread_id, root_run_id, plan_id, step_id,
                    input_hash, inspection_audit_id, decision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, thread_id, root_run_id, plan_id, step_id, input_hash)
                DO UPDATE SET inspection_audit_id=excluded.inspection_audit_id,
                              decision=excluded.decision, created_at=excluded.created_at
                """,
                (
                    checked.project_id,
                    checked.thread_id,
                    checked.root_run_id,
                    checked.plan_id,
                    step_id,
                    input_hash,
                    inspection_result.audit_id,
                    decision,
                    _now(),
                ),
            )
            connection.commit()
        checkpoint_id = self._record_checkpoint(
            checked,
            step,
            phase="inspection",
            input_hash=input_hash,
            idempotency_key=str(row["idempotency_key"]),
            attempt=int(row["attempts"]),
            status=decision,
            result=inspection_result,
        )
        return next(item for item in self.list_checkpoints(checked.plan_id) if item.checkpoint_id == checkpoint_id)

    def list_checkpoints(self, plan_id: str) -> list[DAGCheckpoint]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM dag_checkpoints WHERE plan_id = ? ORDER BY rowid",
                (plan_id,),
            ).fetchall()
        return [self._checkpoint_from_row(row) for row in rows]

    def verify_state(
        self,
        plan: TaskPlan,
        state: PlanExecutionState,
        step_results: Mapping[str, ToolResult],
    ) -> bool:
        """Verify review inputs against this executor's durable rows."""

        try:
            checked = self._preflight(
                plan,
                project_id=plan.project_id,
                thread_id=plan.thread_id,
                root_run_id=plan.root_run_id,
            )
            state_checked = PlanExecutionState.model_validate(state.model_dump(mode="python"))
            if (
                state_checked.project_id,
                state_checked.thread_id,
                state_checked.root_run_id,
                state_checked.plan_id,
                state_checked.plan_sha256,
            ) != (
                checked.project_id,
                checked.thread_id,
                checked.root_run_id,
                checked.plan_id,
                checked.plan_sha256,
            ):
                return False
            if {item.step_id for item in state_checked.step_states} != {
                item.step_id for item in checked.steps
            }:
                return False
            for item in state_checked.step_states:
                row = self._load_step(checked, item.step_id)
                if row is None:
                    if item.status != "pending" or item.attempts != 0:
                        return False
                    continue
                if (
                    item.status != row["status"]
                    or item.attempts != int(row["attempts"])
                    or item.last_input_hash != row["input_hash"]
                    or item.last_checkpoint_id != row["last_checkpoint_id"]
                    or item.last_output_refs
                    != [
                        *[
                            f"artifact:{artifact_id}"
                            for artifact_id in json.loads(row["artifact_ids_json"])
                        ],
                        *[
                            f"handoff:{handoff_id}"
                            for handoff_id in json.loads(row["handoff_ids_json"])
                        ],
                    ]
                ):
                    return False
                with closing(self._connect()) as connection:
                    checkpoint = connection.execute(
                        """
                        SELECT * FROM dag_checkpoints
                        WHERE checkpoint_id=? AND project_id=? AND thread_id=?
                          AND root_run_id=? AND plan_id=? AND plan_sha256=? AND step_id=?
                        """,
                        (
                            item.last_checkpoint_id,
                            checked.project_id,
                            checked.thread_id,
                            checked.root_run_id,
                            checked.plan_id,
                            checked.plan_sha256,
                            item.step_id,
                        ),
                    ).fetchone()
                if checkpoint is None or checkpoint["phase"] not in {"after", "inspection"}:
                    return False
                result = step_results.get(item.step_id)
                if result is None:
                    if row["result_json"] is not None:
                        return False
                elif row["result_json"] != _canonical_json(result):
                    return False
                elif result.status == "success":
                    self._validate_result_references(checked, next(
                        step for step in checked.steps if step.step_id == item.step_id
                    ), result)
            return set(step_results).issubset({item.step_id for item in checked.steps})
        except (ValueError, TypeError, ValidationError, KeyError, RuntimeError, sqlite3.Error):
            return False

    def derive_domain_reviews(
        self,
        plan: TaskPlan,
        state: PlanExecutionState,
        step_results: Mapping[str, ToolResult],
    ) -> list[DomainReview]:
        """Derive domain review status only from durable artifact/handoff evidence."""

        state_by_step = {item.step_id: item for item in state.step_states}
        reviews: list[DomainReview] = []
        for step in plan.steps:
            if step.executor_type != "domain_subgraph":
                continue
            try:
                step_state = state_by_step[step.step_id]
                result = step_results[step.step_id]
                if step_state.status != "completed" or result.status != "success":
                    raise ValueError("domain step is not durably completed")
                self._validate_result_references(plan, step, result)
                self._validate_target_handoffs(plan, step)
                refs = list(step_state.last_output_refs)
                if not refs or not any(item.startswith("artifact:") for item in refs):
                    raise ValueError("domain step has no artifact evidence reference")
                reviews.append(
                    DomainReview(
                        domain=str(step.domain_subgraph),
                        status="passed",
                        validator="artifact-store-persistent-validation",
                        checks=[
                            CheckResult(
                                check_id=f"domain-artifacts:{step.step_id}",
                                passed=True,
                                severity="error",
                                message="artifact and handoff records revalidated from persistence",
                                evidence_refs=refs,
                            )
                        ],
                    )
                )
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                reviews.append(
                    DomainReview(
                        domain=str(step.domain_subgraph),
                        status="failed",
                        validator="artifact-store-persistent-validation",
                        checks=[
                            CheckResult(
                                check_id=f"domain-artifacts:{step.step_id}",
                                passed=False,
                                severity="critical",
                                message=str(exc),
                                evidence_refs=[],
                            )
                        ],
                    )
                )
        return reviews

    @staticmethod
    def _domain_failure(step: PlanStep, message: str) -> ToolResult:
        return ToolResult(
            status="failed",
            error=ToolError(
                code="failed",
                message=message,
                recoverability="manual_recovery",
            ),
            audit_id=uuid4().hex,
            operation_id=f"dag-domain:{step.step_id}",
            recovery_class="manual_recovery",
        )

    @staticmethod
    def _result_output_refs(result: ToolResult) -> list[str]:
        if not isinstance(result.result, dict):
            return []
        artifact_ids = result.result.get("artifact_ids", [])
        handoff_ids = result.result.get("handoff_ids", [])
        if not isinstance(artifact_ids, list) or not isinstance(handoff_ids, list):
            raise ValueError("domain output references must be lists")
        return [
            *[f"artifact:{item}" for item in artifact_ids if isinstance(item, str)],
            *[f"handoff:{item}" for item in handoff_ids if isinstance(item, str)],
        ]

    def _snapshot_for_capability(self, plan: TaskPlan, capability_id: str):
        return next(
            item
            for item in plan.execution_snapshot.capability_snapshots
            if item.capability_id == capability_id
        )

    def _execute_handoff_conversion(
        self,
        plan: TaskPlan,
        planned: Any,
        *,
        source_artifacts: Mapping[str, Any],
        checkpoint_binding: ArtifactCheckpointBinding,
    ) -> DomainExecutionOutput:
        if self._artifact_store is None or planned.conversion is None:
            raise ValueError("handoff conversion has no artifact trust boundary")
        conversion = planned.conversion
        registrations = [
            item
            for item in self._registry._registrations.values()
            if item.kind == "domain"
            and item.manifest.capability_id == conversion.converter_capability
            and item.manifest.status == "available"
        ]
        if len(registrations) != 1:
            raise ValueError(
                "conversion capability must have exactly one available domain executor"
            )
        registration = registrations[0]
        resolved_sources = [
            self._artifact_store.resolve(
                artifact,
                project_id=plan.project_id,
                thread_id=plan.thread_id,
            )
            for artifact in source_artifacts.values()
        ]
        arguments: dict[str, Any] = {}
        for source_field, converter_input in conversion.input_mapping.items():
            values = [
                item.payload[source_field]
                for item in resolved_sources
                if isinstance(item.payload, dict) and source_field in item.payload
            ]
            if len(values) != 1 or converter_input in arguments:
                raise ValueError(
                    f"conversion input mapping is ambiguous or missing: {source_field}"
                )
            arguments[converter_input] = values[0]

        snapshot = self._snapshot_for_capability(
            plan, conversion.converter_capability
        )
        step_id = (
            f"handoff-conversion:{planned.source_step_id}:{planned.target_step_id}"
        )
        checkpoint_ref = {
            "project_id": plan.project_id,
            "thread_id": plan.thread_id,
            "checkpoint_id": checkpoint_binding.checkpoint_id,
        }
        domain_state = {
            "execution_role": "handoff_conversion",
            "project_id": plan.project_id,
            "thread_id": plan.thread_id,
            "root_run_id": plan.root_run_id,
            "plan_id": plan.plan_id,
            "plan_sha256": plan.plan_sha256,
            "step_id": step_id,
            "source_step_id": planned.source_step_id,
            "target_step_id": planned.target_step_id,
            "phase": "after",
            "checkpoint_ref": checkpoint_ref,
            "capability_snapshot": snapshot.model_dump(mode="json"),
            "handoff": planned.model_dump(mode="json"),
        }
        request = ExecutionRequest(
            kind="domain",
            capability_id=conversion.converter_capability,
            executor_name=registration.executor_name,
            operation_id=(
                f"dag-conversion:{plan.plan_id}:{planned.source_step_id}:"
                f"{planned.target_step_id}:{checkpoint_binding.checkpoint_id}"
            ),
            arguments=arguments,
            origin="planned",
            step_id=step_id,
            domain_state=domain_state,
        )
        context = ExecutionContext(
            project_id=plan.project_id,
            thread_id=plan.thread_id,
            root_run_id=plan.root_run_id,
            workspace_root=str(self._workspace_root),
            capability_snapshot=snapshot,
            plan_id=plan.plan_id,
            plan_sha256=plan.plan_sha256,
            registry_snapshot_sha256=plan.execution_snapshot.registry_snapshot_sha256,
        )
        authorized = self._registry.resolve(request, context)
        if not isinstance(authorized, AuthorizedExecutor):
            message = authorized.error.message if authorized.error is not None else authorized.status
            raise ValueError(f"conversion authorization failed: {message}")
        result = self._execution_service.execute(authorized)
        if result.status != "success":
            message = result.error.message if result.error is not None else result.status
            raise ValueError(f"conversion execution failed: {message}")
        if not isinstance(result.result, dict) or "domain_output" not in result.result:
            raise ValueError("conversion execution lacks validated domain output")
        converted = DomainExecutionOutput.model_validate(result.result["domain_output"])
        if not converted.artifacts:
            raise ValueError("handoff conversion produced no concrete artifact")
        return converted

    def _persist_domain_result(
        self,
        plan: TaskPlan,
        step: PlanStep,
        result: ToolResult,
        *,
        checkpoint_id: str,
    ) -> ToolResult:
        if self._artifact_store is None:
            raise ValueError("domain execution has no ArtifactStore trust boundary")
        if not isinstance(result.result, dict) or "domain_output" not in result.result:
            raise ValueError("domain result lacks validated DomainExecutionOutput")
        output = DomainExecutionOutput.model_validate(result.result["domain_output"])
        if not output.artifacts:
            raise ValueError("domain result has no concrete artifacts")
        source_snapshot = self._snapshot_for_capability(plan, step.capability)
        binding = ArtifactCheckpointBinding(
            project_id=plan.project_id,
            thread_id=plan.thread_id,
            root_run_id=plan.root_run_id,
            plan_id=plan.plan_id,
            plan_sha256=plan.plan_sha256,
            step_id=step.step_id,
            checkpoint_id=checkpoint_id,
            phase="after",
        )
        expected_checkpoint = CheckpointRef(
            project_id=plan.project_id,
            thread_id=plan.thread_id,
            checkpoint_id=checkpoint_id,
        )
        source_artifacts = {item.artifact_id: item for item in output.artifacts}
        for artifact in output.artifacts:
            if artifact.source_capability != source_snapshot:
                raise ValueError("domain artifact source snapshot mismatch")
            if expected_checkpoint not in artifact.source_checkpoints:
                raise ValueError("domain artifact source checkpoint mismatch")

        resolved_handoffs = []
        all_artifacts = list(output.artifacts)
        by_step = {item.step_id: item for item in plan.steps}
        for planned in (
            item for item in plan.handoffs if item.source_step_id == step.step_id
        ):
            target_step = by_step[planned.target_step_id]
            target_artifacts: dict[str, Any] = {}
            target_snapshot = self._snapshot_for_capability(plan, target_step.capability)
            if planned.conversion is not None:
                converted = self._execute_handoff_conversion(
                    plan,
                    planned,
                    source_artifacts=source_artifacts,
                    checkpoint_binding=binding,
                )
                if not converted.artifacts:
                    raise ValueError("handoff conversion produced no concrete artifact")
                target_artifacts = {
                    item.artifact_id: item for item in converted.artifacts
                }
                all_artifacts.extend(converted.artifacts)
                target_snapshot = self._snapshot_for_capability(
                    plan, planned.conversion.converter_capability
                )

            check_ids = (
                [planned.conversion.validation_check_id]
                if planned.conversion is not None
                else [
                    check.check_id
                    for check in target_step.verification
                    if check.required
                    and check.method in {"domain_validator", "cross_validation"}
                ]
            )
            if not check_ids:
                raise ValueError("cross-domain handoff has no trusted validation check")
            validation_results: dict[str, bool] = {}
            for check_id in check_ids:
                validator = self._handoff_validation_checks.get(check_id)
                if validator is None:
                    raise ValueError(f"trusted handoff validation check is unavailable: {check_id}")
                validation_results[check_id] = validator(
                    {
                        "plan_id": plan.plan_id,
                        "plan_sha256": plan.plan_sha256,
                        "source_step_id": step.step_id,
                        "target_step_id": target_step.step_id,
                        "source_artifact_ids": sorted(source_artifacts),
                        "target_artifact_ids": sorted(target_artifacts),
                        "checkpoint_binding": binding.model_dump(mode="json"),
                    }
                ) is True
            handoff_id = "handoff:" + _sha256(
                {
                    "plan_id": plan.plan_id,
                    "source_step_id": step.step_id,
                    "target_step_id": target_step.step_id,
                    "source_artifact_ids": sorted(source_artifacts),
                    "target_artifact_ids": sorted(target_artifacts),
                }
            )
            record = HandoffRecord(
                handoff_id=handoff_id,
                plan_id=plan.plan_id,
                source_step_id=step.step_id,
                target_step_id=target_step.step_id,
                source_artifact_ids=sorted(source_artifacts),
                target_artifact_ids=sorted(target_artifacts),
                conversion_capability=(
                    planned.conversion.converter_capability
                    if planned.conversion is not None
                    else None
                ),
                validation_check_ids=check_ids,
                checkpoint=expected_checkpoint,
            )
            resolved_handoffs.append(
                self._artifact_store.validate_handoff(
                    planned,
                    record,
                    source_artifacts=source_artifacts,
                    target_artifacts=target_artifacts,
                    project_id=plan.project_id,
                    thread_id=plan.thread_id,
                    plan_id=plan.plan_id,
                    plan_sha256=plan.plan_sha256,
                    source_snapshot=source_snapshot,
                    target_snapshot=target_snapshot,
                    source_checkpoint_binding=binding,
                    validation_results=validation_results,
                )
            )

        artifact_records, handoff_records = self._artifact_store.persist_exchange(
            artifacts=all_artifacts,
            handoffs=resolved_handoffs,
            project_id=plan.project_id,
            thread_id=plan.thread_id,
            root_run_id=plan.root_run_id,
            plan_id=plan.plan_id,
            plan_sha256=plan.plan_sha256,
            step_id=step.step_id,
            checkpoint_binding=binding,
            source_snapshot=source_snapshot,
        )
        data = result.model_dump(mode="python")
        payload = dict(result.result)
        payload["artifact_ids"] = [item.artifact_id for item in artifact_records]
        payload["handoff_ids"] = [item.handoff_id for item in handoff_records]
        data["result"] = payload
        return ToolResult.model_validate(data)

    def _validate_target_handoffs(self, plan: TaskPlan, step: PlanStep) -> None:
        incoming = [item for item in plan.handoffs if item.target_step_id == step.step_id]
        if not incoming:
            return
        if self._artifact_store is None:
            raise ValueError("target domain step has no ArtifactStore")
        persisted_ids = self._artifact_store.handoff_ids_for_target(
            project_id=plan.project_id,
            plan_id=plan.plan_id,
            target_step_id=step.step_id,
        )
        if len(persisted_ids) != len(incoming):
            raise ValueError("persisted handoff count does not match the immutable plan")
        by_pair = {
            (item.source_step_id, item.target_step_id): item for item in incoming
        }
        for handoff_id in persisted_ids:
            # Read the stored envelope first to bind it to one exact planned pair.
            matched = None
            for planned in incoming:
                source_step = next(
                    item for item in plan.steps if item.step_id == planned.source_step_id
                )
                source_snapshot = self._snapshot_for_capability(
                    plan, source_step.capability
                )
                target_snapshot = self._snapshot_for_capability(plan, step.capability)
                if planned.conversion is not None:
                    target_snapshot = self._snapshot_for_capability(
                        plan, planned.conversion.converter_capability
                    )
                try:
                    candidate = self._artifact_store.load_handoff(
                        planned,
                        handoff_id,
                        project_id=plan.project_id,
                        thread_id=plan.thread_id,
                        root_run_id=plan.root_run_id,
                        plan_id=plan.plan_id,
                        plan_sha256=plan.plan_sha256,
                        source_snapshot=source_snapshot,
                        target_snapshot=target_snapshot,
                    )
                except ValueError:
                    continue
                if (candidate.record.source_step_id, candidate.record.target_step_id) in by_pair:
                    matched = candidate
                    break
            if matched is None:
                raise ValueError("persisted handoff does not match any planned source/target pair")
            planned_inputs = {
                name: step.inputs.get(name) for name in matched.target_inputs
            }
            if planned_inputs != matched.target_inputs:
                raise ValueError("persisted handoff mapping does not match target plan inputs")

    def _validate_result_references(
        self,
        plan: TaskPlan,
        step: PlanStep,
        result: ToolResult,
    ) -> None:
        if step.executor_type != "domain_subgraph":
            return
        if self._artifact_store is None:
            raise ValueError("domain result cannot be resumed without ArtifactStore")
        refs = self._result_output_refs(result)
        if not refs or not any(item.startswith("artifact:") for item in refs):
            raise ValueError("domain result lacks durable artifact references")
        for ref in refs:
            if ref.startswith("artifact:"):
                self._artifact_store.load_artifact_record(
                    ref.removeprefix("artifact:"),
                    project_id=plan.project_id,
                    thread_id=plan.thread_id,
                    root_run_id=plan.root_run_id,
                    plan_id=plan.plan_id,
                    plan_sha256=plan.plan_sha256,
                )
        expected_handoffs: set[str] = set()
        for handoff in plan.handoffs:
            if handoff.source_step_id == step.step_id:
                expected_handoffs.update(
                    self._artifact_store.handoff_ids_for_target(
                        project_id=plan.project_id,
                        plan_id=plan.plan_id,
                        target_step_id=handoff.target_step_id,
                    )
                )
        actual_handoffs = {
            item.removeprefix("handoff:")
            for item in refs
            if item.startswith("handoff:")
        }
        if actual_handoffs != expected_handoffs:
            raise ValueError("domain result handoff references do not match persistence")

    def _preflight(
        self,
        plan: TaskPlan | Mapping[str, Any],
        *,
        project_id: str,
        thread_id: str,
        root_run_id: str,
    ) -> TaskPlan:
        checked = TaskPlan.model_validate(
            plan.model_dump(mode="python") if isinstance(plan, TaskPlan) else dict(plan)
        )
        if compute_task_plan_sha256(checked) != checked.plan_sha256:
            raise ValueError("TaskPlan canonical hash mismatch")
        if (checked.project_id, checked.thread_id, checked.root_run_id) != (
            project_id,
            thread_id,
            root_run_id,
        ):
            raise ValueError("TaskPlan project/thread/root-run identity mismatch")
        if not checked.steps:
            raise ValueError("TaskPlan must contain at least one step")

        self._validate_handoffs(checked)
        snapshots = {
            snapshot.capability_id: snapshot
            for snapshot in checked.execution_snapshot.capability_snapshots
        }
        if len(snapshots) != len(checked.execution_snapshot.capability_snapshots):
            raise ValueError("TaskPlan contains duplicate capability snapshots")
        required_capabilities = {step.capability for step in checked.steps} | {
            handoff.conversion.converter_capability
            for handoff in checked.handoffs
            if handoff.conversion is not None
        }
        if checked.selected_capabilities and set(checked.selected_capabilities) != required_capabilities:
            raise ValueError("selected capabilities do not exactly match plan and conversion requirements")

        for step in checked.steps:
            kind, executor_name = _step_binding(step)
            registration = self._registry._registrations.get(
                (kind, step.capability, executor_name)
            )
            if registration is None:
                raise ValueError(
                    f"unavailable planned executor: {kind}/{step.capability}/{executor_name}"
                )
            if registration.manifest.status != "available":
                raise ValueError(f"capability is unavailable: {step.capability}")
            current_snapshot = self._registry.snapshot(step.capability)
            if snapshots.get(step.capability) != current_snapshot:
                raise ValueError(f"capability snapshot mismatch: {step.capability}")
            risk_rank = {"read_only": 0, "project_write": 1, "high_risk": 2}
            if risk_rank[step.risk_level] < risk_rank[registration.manifest.risk_level]:
                raise ValueError(f"step understates capability risk: {step.step_id}")
            if registration.requires_confirmation and not step.requires_confirmation:
                raise ValueError(f"step omits required confirmation: {step.step_id}")
            if step.executor_type == "workflow":
                workflow = next(
                    (
                        item
                        for item in checked.execution_snapshot.workflow_snapshots
                        if item.workflow_id == step.workflow_id
                        and item.version == step.workflow_version
                    ),
                    None,
                )
                if workflow is None or workflow != registration.workflow_snapshot:
                    raise ValueError(f"workflow snapshot mismatch: {step.step_id}")
        return checked

    def _validate_handoffs(self, plan: TaskPlan) -> None:
        by_step = {step.step_id: step for step in plan.steps}
        verification = {
            check.check_id: check
            for step in plan.steps
            for check in step.verification
        }
        for handoff in plan.handoffs:
            source = by_step[handoff.source_step_id]
            target = by_step[handoff.target_step_id]
            if source.executor_type != "domain_subgraph" or target.executor_type != "domain_subgraph":
                raise ValueError("cross-domain handoff endpoints must be domain steps")
            if source.domain_subgraph != handoff.source_domain or target.domain_subgraph != handoff.target_domain:
                raise ValueError("cross-domain handoff domain identity mismatch")
            if source.step_id not in target.dependencies:
                raise ValueError("cross-domain target must depend on its source step")
            if not handoff.required_inputs or not handoff.expected_outputs:
                raise ValueError("cross-domain handoff requires explicit input/output fields")
            if set(handoff.source_output_mapping) != set(handoff.expected_outputs):
                raise ValueError("cross-domain source mapping is incomplete or overfilled")
            if set(handoff.target_input_mapping.values()) != set(handoff.required_inputs):
                raise ValueError("cross-domain target mapping is incomplete or overfilled")
            for metadata in (handoff.source_metadata, handoff.target_metadata):
                unknown_units = set(metadata.quantity_units.values()) - self._vocabulary.quantity_units
                if unknown_units:
                    raise ValueError(
                        "non-canonical quantity unit(s): " + ", ".join(sorted(unknown_units))
                    )
                if metadata.frame_id is not None and metadata.frame_id not in self._vocabulary.frame_ids:
                    raise ValueError(f"non-canonical frame identifier: {metadata.frame_id}")
                if metadata.time_system is not None and metadata.time_system not in self._vocabulary.time_systems:
                    raise ValueError(f"non-canonical time system: {metadata.time_system}")
                if metadata.epoch_field is not None and not metadata.epoch_field.strip():
                    raise ValueError("epoch field cannot be blank")
            conversion = handoff.conversion
            if handoff.source_metadata != handoff.target_metadata and conversion is None:
                raise ValueError("metadata difference requires a conversion")
            if conversion is not None:
                if set(conversion.input_mapping) != set(handoff.expected_outputs):
                    raise ValueError("conversion input mapping is incomplete or overfilled")
                if set(conversion.output_mapping) != set(handoff.required_inputs):
                    raise ValueError("conversion output mapping is incomplete or overfilled")
                check = verification.get(conversion.validation_check_id)
                if check is None or not check.required or check.method not in {
                    "tool",
                    "domain_validator",
                    "cross_validation",
                }:
                    raise ValueError("conversion requires an explicit executable verification check")
                planned_snapshot = next(
                    (
                        item
                        for item in plan.execution_snapshot.capability_snapshots
                        if item.capability_id == conversion.converter_capability
                    ),
                    None,
                )
                try:
                    current_snapshot = self._registry.snapshot(conversion.converter_capability)
                except (KeyError, RuntimeError) as exc:
                    raise ValueError("conversion capability is unavailable") from exc
                available_registrations = [
                    registration
                    for registration in self._registry._registrations.values()
                    if registration.manifest.capability_id == conversion.converter_capability
                    and registration.manifest.status == "available"
                ]
                if planned_snapshot != current_snapshot or len(available_registrations) != 1:
                    raise ValueError("conversion capability snapshot is unavailable or mismatched")

    @staticmethod
    def _topological_steps(plan: TaskPlan) -> list[PlanStep]:
        by_id = {step.step_id: step for step in plan.steps}
        indegree = {step.step_id: len(step.dependencies) for step in plan.steps}
        dependents: dict[str, list[str]] = {step.step_id: [] for step in plan.steps}
        for step in plan.steps:
            for dependency in step.dependencies:
                dependents[dependency].append(step.step_id)
        ready = deque(step.step_id for step in plan.steps if indegree[step.step_id] == 0)
        ordered: list[PlanStep] = []
        while ready:
            step_id = ready.popleft()
            ordered.append(by_id[step_id])
            for dependent in dependents[step_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
        if len(ordered) != len(plan.steps):
            raise ValueError("step dependencies must be acyclic")
        return ordered

    @staticmethod
    def _input_hash(step: PlanStep, results: Mapping[str, ToolResult]) -> str:
        dependency_results = {
            dependency: results[dependency].model_dump(mode="json")
            for dependency in sorted(step.dependencies)
        }
        return _sha256(
            {
                "inputs": step.inputs,
                "dependency_results": dependency_results,
            }
        )

    def _execution_request(
        self,
        plan: TaskPlan,
        step: PlanStep,
        *,
        input_hash: str,
        attempt: int,
        confirmation_id: str | None,
        domain_checkpoint_id: str | None = None,
    ) -> tuple[ExecutionRequest, ExecutionContext]:
        kind, executor_name = _step_binding(step)
        capability_snapshot = next(
            item
            for item in plan.execution_snapshot.capability_snapshots
            if item.capability_id == step.capability
        )
        workflow_snapshot = None
        if step.executor_type == "workflow":
            workflow_snapshot = next(
                item
                for item in plan.execution_snapshot.workflow_snapshots
                if item.workflow_id == step.workflow_id and item.version == step.workflow_version
            )
        operation_id = (
            f"dag:{plan.plan_id}:{step.step_id}:{input_hash[:16]}:{attempt}"
        )
        request_data: dict[str, Any] = {
            "kind": kind,
            "capability_id": step.capability,
            "executor_name": executor_name,
            "operation_id": operation_id,
            "arguments": dict(step.inputs),
            "confirmation_id": confirmation_id,
            "origin": "planned",
            "step_id": step.step_id,
        }
        if kind == "domain":
            if not domain_checkpoint_id:
                raise ValueError("domain execution requires a reserved after-checkpoint identity")
            request_data["domain_state"] = {
                "project_id": plan.project_id,
                "thread_id": plan.thread_id,
                "root_run_id": plan.root_run_id,
                "plan_id": plan.plan_id,
                "plan_sha256": plan.plan_sha256,
                "step_id": step.step_id,
                "phase": "after",
                "checkpoint_ref": {
                    "project_id": plan.project_id,
                    "thread_id": plan.thread_id,
                    "checkpoint_id": domain_checkpoint_id,
                },
                "capability_snapshot": capability_snapshot.model_dump(mode="json"),
            }
        return (
            ExecutionRequest.model_validate(request_data),
            ExecutionContext(
                project_id=plan.project_id,
                thread_id=plan.thread_id,
                root_run_id=plan.root_run_id,
                workspace_root=str(self._workspace_root),
                capability_snapshot=capability_snapshot,
                workflow_snapshot=workflow_snapshot,
                plan_id=plan.plan_id,
                plan_sha256=plan.plan_sha256,
                registry_snapshot_sha256=plan.execution_snapshot.registry_snapshot_sha256,
            ),
        )

    def _ensure_run(self, plan: TaskPlan) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM dag_runs WHERE plan_id = ?",
                (plan.plan_id,),
            ).fetchone()
            if row is not None and (
                row["project_id"], row["thread_id"], row["root_run_id"], row["plan_sha256"]
            ) != (plan.project_id, plan.thread_id, plan.root_run_id, plan.plan_sha256):
                connection.rollback()
                raise ValueError("persisted DAG run identity mismatch")
            connection.execute(
                """
                INSERT INTO dag_runs(plan_id, project_id, thread_id, root_run_id, plan_sha256, status, updated_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?)
                ON CONFLICT(plan_id) DO UPDATE SET status='running', updated_at=excluded.updated_at
                """,
                (
                    plan.plan_id,
                    plan.project_id,
                    plan.thread_id,
                    plan.root_run_id,
                    plan.plan_sha256,
                    _now(),
                ),
            )
            connection.commit()

    def _record_checkpoint(
        self,
        plan: TaskPlan,
        step: PlanStep,
        *,
        phase: str,
        input_hash: str,
        idempotency_key: str,
        attempt: int,
        status: str,
        result: ToolResult | None,
        checkpoint_id: str | None = None,
    ) -> str:
        checkpoint_id = checkpoint_id or f"dag:{uuid4().hex}"
        result_json = _canonical_json(result) if result is not None else None
        refs = self._result_output_refs(result) if result is not None else []
        artifact_ids_json = _canonical_json(
            [item.removeprefix("artifact:") for item in refs if item.startswith("artifact:")]
        )
        handoff_ids_json = _canonical_json(
            [item.removeprefix("handoff:") for item in refs if item.startswith("handoff:")]
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO dag_checkpoints(
                    checkpoint_id, project_id, thread_id, root_run_id, plan_id,
                    plan_sha256, step_id, phase, input_hash, idempotency_key,
                    attempt, status, result_json, artifact_ids_json,
                    handoff_ids_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    plan.project_id,
                    plan.thread_id,
                    plan.root_run_id,
                    plan.plan_id,
                    plan.plan_sha256,
                    step.step_id,
                    phase,
                    input_hash,
                    idempotency_key,
                    attempt,
                    status,
                    result_json,
                    artifact_ids_json,
                    handoff_ids_json,
                    _now(),
                ),
            )
            connection.commit()
        return checkpoint_id

    def _save_step(
        self,
        plan: TaskPlan,
        step: PlanStep,
        *,
        input_hash: str,
        idempotency_key: str,
        status: str,
        attempts: int,
        result: ToolResult | None,
        checkpoint_id: str,
    ) -> None:
        result_json = _canonical_json(result) if result is not None else None
        refs = self._result_output_refs(result) if result is not None else []
        artifact_ids_json = _canonical_json(
            [item.removeprefix("artifact:") for item in refs if item.startswith("artifact:")]
        )
        handoff_ids_json = _canonical_json(
            [item.removeprefix("handoff:") for item in refs if item.startswith("handoff:")]
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO dag_steps(
                    project_id, thread_id, root_run_id, plan_id, plan_sha256,
                    step_id, input_hash, idempotency_key, status, attempts,
                    result_json, artifact_ids_json, handoff_ids_json,
                    last_checkpoint_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, thread_id, root_run_id, plan_id, step_id)
                DO UPDATE SET plan_sha256=excluded.plan_sha256,
                              input_hash=excluded.input_hash,
                              idempotency_key=excluded.idempotency_key,
                              status=excluded.status,
                              attempts=excluded.attempts,
                              result_json=excluded.result_json,
                              artifact_ids_json=excluded.artifact_ids_json,
                              handoff_ids_json=excluded.handoff_ids_json,
                              last_checkpoint_id=excluded.last_checkpoint_id,
                              updated_at=excluded.updated_at
                """,
                (
                    plan.project_id,
                    plan.thread_id,
                    plan.root_run_id,
                    plan.plan_id,
                    plan.plan_sha256,
                    step.step_id,
                    input_hash,
                    idempotency_key,
                    status,
                    attempts,
                    result_json,
                    artifact_ids_json,
                    handoff_ids_json,
                    checkpoint_id,
                    _now(),
                ),
            )
            connection.commit()

    def _load_step(self, plan: TaskPlan, step_id: str) -> sqlite3.Row | None:
        with closing(self._connect()) as connection:
            return connection.execute(
                """
                SELECT * FROM dag_steps
                WHERE project_id=? AND thread_id=? AND root_run_id=?
                  AND plan_id=? AND plan_sha256=? AND step_id=?
                """,
                (
                    plan.project_id,
                    plan.thread_id,
                    plan.root_run_id,
                    plan.plan_id,
                    plan.plan_sha256,
                    step_id,
                ),
            ).fetchone()

    def _load_inspection(self, plan: TaskPlan, step_id: str, input_hash: str) -> sqlite3.Row | None:
        with closing(self._connect()) as connection:
            return connection.execute(
                """
                SELECT * FROM dag_interruption_inspections
                WHERE project_id=? AND thread_id=? AND root_run_id=?
                  AND plan_id=? AND step_id=? AND input_hash=?
                """,
                (
                    plan.project_id,
                    plan.thread_id,
                    plan.root_run_id,
                    plan.plan_id,
                    step_id,
                    input_hash,
                ),
            ).fetchone()

    @staticmethod
    def _step_status(result: ToolResult) -> str:
        return {
            "success": "completed",
            "blocked": "blocked",
            "interrupted": "interrupted",
        }.get(result.status, "failed")

    def _state(self, plan: TaskPlan) -> PlanExecutionState:
        states: list[PlanStepExecutionState] = []
        for step in plan.steps:
            row = self._load_step(plan, step.step_id)
            if row is None:
                states.append(PlanStepExecutionState(step_id=step.step_id, status="pending"))
                continue
            states.append(
                PlanStepExecutionState(
                    step_id=step.step_id,
                    status=row["status"],
                    attempts=int(row["attempts"]),
                    last_input_hash=row["input_hash"],
                    last_output_refs=[
                        *[
                            f"artifact:{artifact_id}"
                            for artifact_id in json.loads(row["artifact_ids_json"])
                        ],
                        *[
                            f"handoff:{handoff_id}"
                            for handoff_id in json.loads(row["handoff_ids_json"])
                        ],
                    ],
                    last_checkpoint_id=row["last_checkpoint_id"],
                )
            )
        return PlanExecutionState(
            project_id=plan.project_id,
            thread_id=plan.thread_id,
            root_run_id=plan.root_run_id,
            plan_id=plan.plan_id,
            plan_sha256=plan.plan_sha256,
            step_states=states,
            updated_at=_now(),
        )

    def _outcome(
        self,
        plan: TaskPlan,
        results: dict[str, ToolResult],
        reused: list[str],
        status: Literal["completed", "partial", "blocked", "failed", "interrupted"],
    ) -> DAGExecutionOutcome:
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE dag_runs SET status=?, updated_at=? WHERE plan_id=?",
                (status, _now(), plan.plan_id),
            )
            connection.commit()
        return DAGExecutionOutcome(
            status=status,
            plan_id=plan.plan_id,
            plan_sha256=plan.plan_sha256,
            state=self._state(plan),
            step_results=results,
            reused_step_ids=reused,
        )

    @staticmethod
    def _checkpoint_from_row(row: sqlite3.Row) -> DAGCheckpoint:
        return DAGCheckpoint(
            checkpoint_id=row["checkpoint_id"],
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            root_run_id=row["root_run_id"],
            plan_id=row["plan_id"],
            plan_sha256=row["plan_sha256"],
            step_id=row["step_id"],
            phase=row["phase"],
            input_hash=row["input_hash"],
            idempotency_key=row["idempotency_key"],
            attempt=int(row["attempt"]),
            status=row["status"],
            result=(ToolResult.model_validate(json.loads(row["result_json"])) if row["result_json"] else None),
            artifact_ids=json.loads(row["artifact_ids_json"]),
            handoff_ids=json.loads(row["handoff_ids_json"]),
            created_at=row["created_at"],
        )


__all__ = [
    "CanonicalMetadataVocabulary",
    "CheckpointedDAGExecutor",
    "DAGCheckpoint",
    "DAGExecutionOutcome",
]
