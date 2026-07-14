"""Strict public data contracts for the Agent Core runtime.

This module contains data and cross-field validation only.  It deliberately
does not import graph nodes, tools, storage, or domain implementations.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, field_validator, model_validator


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class FrozenList(list[Any]):
    """JSON-serializable list whose mutation methods fail closed."""

    @staticmethod
    def _immutable(*_: Any, **__: Any) -> None:
        raise TypeError("immutable contract collection")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


class FrozenDict(dict[str, Any]):
    """JSON-serializable mapping whose mutation methods fail closed."""

    @staticmethod
    def _immutable(*_: Any, **__: Any) -> None:
        raise TypeError("immutable contract mapping")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


class ContractModel(BaseModel):
    """Base for public contracts; undeclared data is never accepted."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class FrozenContractModel(ContractModel):
    """Deeply immutable plan content and snapshot contract."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @model_validator(mode="after")
    def freeze_collections(self) -> "FrozenContractModel":
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, _deep_freeze(getattr(self, field_name)))
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return a validated, re-frozen copy, including updated fields.

        Pydantic's default ``model_copy(update=...)`` intentionally skips
        validation.  That behavior would break the immutable plan/snapshot
        guarantee, so public frozen contracts always round-trip through their
        validator.  ``deep`` is accepted for API compatibility; validation
        reconstructs nested values regardless.
        """

        del deep
        data = self.model_dump(mode="python", round_trip=True)
        if update:
            data.update(dict(update))
        return type(self).model_validate(data)


def _validate_sha256(value: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError("must be a 64-character SHA-256 hex digest")
    return value


class CapabilityManifest(ContractModel):
    capability_id: str
    version: str
    category: Literal["basic", "space_basic", "domain", "workflow", "memory", "project"]
    status: Literal["available", "interface_only", "disabled", "unavailable"]
    intents: list[str]
    tool_names: list[str] = Field(default_factory=list)
    risk_level: Literal["read_only", "project_write", "high_risk"]
    required_dependencies: list[str] = Field(default_factory=list)
    validators: list[str] = Field(default_factory=list)
    source: str


class ToolCall(ContractModel):
    tool_name: str
    arguments: dict[str, Any]
    run_id: str
    operation_id: str
    confirmation_id: str | None = None


class ToolError(ContractModel):
    code: Literal[
        "invalid_arguments",
        "path_outside_workspace",
        "confirmation_required",
        "confirmation_expired",
        "confirmation_replayed",
        "unavailable",
        "timeout",
        "interrupted",
        "conflict",
        "failed",
        "project_not_initialized",
        "project_memory_migration_failed",
    ]
    message: str
    recoverability: Literal["retryable", "reversible", "manual_recovery", "not_applicable"]


class ToolResult(ContractModel):
    status: Literal[
        "success", "blocked", "invalid_arguments", "unavailable", "failed", "timeout", "interrupted"
    ]
    result: dict[str, Any] = Field(default_factory=dict)
    error: ToolError | None = None
    audit_id: str
    operation_id: str
    recovery_class: Literal["read_only", "reversible", "compensatable", "manual_recovery"]

    @model_validator(mode="after")
    def validate_status_error(self) -> "ToolResult":
        if self.status == "success":
            if self.error is not None:
                raise ValueError("successful ToolResult cannot contain an error")
            return self
        if self.error is None:
            raise ValueError("non-success ToolResult requires a structured error")
        compatible = {
            "blocked": {
                "path_outside_workspace",
                "confirmation_required",
                "confirmation_expired",
                "confirmation_replayed",
                "conflict",
                "unavailable",
            },
            "invalid_arguments": {"invalid_arguments"},
            "unavailable": {
                "unavailable",
                "project_not_initialized",
                "project_memory_migration_failed",
            },
            "failed": {"failed"},
            "timeout": {"timeout"},
            "interrupted": {"interrupted"},
        }
        if self.error.code not in compatible[self.status]:
            raise ValueError("ToolResult status is incompatible with error code")
        return self


class ConfirmationGrant(ContractModel):
    confirmation_id: str
    project_id: str
    thread_id: str | None
    root_run_id: str
    operation_id: str
    action_hash: str
    issued_at: str
    expires_at: str
    used_at: str | None = None

    _action_hash = field_validator("action_hash")(_validate_sha256)


class GoalBoundary(FrozenContractModel):
    objective: str
    in_scope: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    termination_conditions: list[str] = Field(default_factory=list)


class VerificationCheck(FrozenContractModel):
    check_id: str
    description: str
    method: Literal["schema", "tool", "domain_validator", "cross_validation", "evidence", "human"]
    required: bool = True
    acceptance_rule: str


class DomainDataMetadata(FrozenContractModel):
    quantity_units: dict[str, str]
    frame_id: str | None = None
    time_system: str | None = None
    epoch_field: str | None = None


class HandoffConversion(FrozenContractModel):
    converter_capability: str
    input_mapping: dict[str, str]
    output_mapping: dict[str, str]
    validation_check_id: str


class CrossDomainHandoff(FrozenContractModel):
    source_step_id: str
    target_step_id: str
    source_domain: str
    target_domain: str
    reason: str
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    source_output_mapping: dict[str, str] = Field(default_factory=dict)
    target_input_mapping: dict[str, str] = Field(default_factory=dict)
    source_metadata: DomainDataMetadata
    target_metadata: DomainDataMetadata
    conversion: HandoffConversion | None = None

    @model_validator(mode="after")
    def validate_conversion(self) -> "CrossDomainHandoff":
        if self.source_metadata != self.target_metadata and self.conversion is None:
            raise ValueError("metadata changes require an explicit handoff conversion")
        return self


class PlanStep(FrozenContractModel):
    step_id: str
    title: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    executor_type: Literal[
        "basic_tool", "space_basic_tool", "domain_subgraph", "workflow", "capability_builder", "human"
    ]
    capability: str
    tool_name: str | None = None
    workflow_id: str | None = None
    workflow_version: str | None = None
    domain_subgraph: str | None = None
    capability_gap_id: str | None = None
    human_instruction: str | None = None
    inputs: dict[str, JsonValue] = Field(default_factory=dict)
    expected_outputs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    verification: list[VerificationCheck] = Field(default_factory=list)
    risk_level: Literal["read_only", "project_write", "high_risk"] = "read_only"
    requires_confirmation: bool = False
    checkpoint_required: bool = True
    max_attempts: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def validate_executor_reference(self) -> "PlanStep":
        references = {
            "tool_name": self.tool_name,
            "workflow_id": self.workflow_id,
            "workflow_version": self.workflow_version,
            "domain_subgraph": self.domain_subgraph,
            "capability_gap_id": self.capability_gap_id,
            "human_instruction": self.human_instruction,
        }
        required: dict[str, set[str]] = {
            "basic_tool": {"tool_name"},
            "space_basic_tool": {"tool_name"},
            "workflow": {"workflow_id", "workflow_version"},
            "domain_subgraph": {"domain_subgraph"},
            "capability_builder": {"capability_gap_id"},
            "human": {"human_instruction"},
        }
        expected = required[self.executor_type]
        missing = [name for name in expected if not references[name]]
        extra = [name for name, value in references.items() if name not in expected and value is not None]
        if missing:
            raise ValueError(f"missing executor reference(s): {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"unexpected executor reference(s): {', '.join(sorted(extra))}")
        if self.risk_level == "high_risk" and not self.requires_confirmation:
            raise ValueError("high-risk step requires confirmation")
        if not self.expected_outputs:
            raise ValueError("step requires at least one expected output")
        if not self.verification:
            raise ValueError("step requires at least one verification check")
        return self


class CapabilitySnapshot(FrozenContractModel):
    capability_id: str
    version: str
    manifest_sha256: str
    adapter_sha256: str | None = None
    trusted_code_roots_sha256: str | None = None

    _manifest_hash = field_validator("manifest_sha256")(_validate_sha256)
    _adapter_hash = field_validator("adapter_sha256")(
        lambda value: _validate_sha256(value) if value is not None else value
    )
    _trusted_code_roots_hash = field_validator("trusted_code_roots_sha256")(
        lambda value: _validate_sha256(value) if value is not None else value
    )


class WorkflowSnapshot(FrozenContractModel):
    workflow_id: str
    version: str
    workflow_sha256: str
    manifest_sha256: str
    approval_record_id: str

    _workflow_hash = field_validator("workflow_sha256")(_validate_sha256)
    _manifest_hash = field_validator("manifest_sha256")(_validate_sha256)


class PlanExecutionSnapshot(FrozenContractModel):
    capability_snapshots: list[CapabilitySnapshot]
    workflow_snapshots: list[WorkflowSnapshot] = Field(default_factory=list)
    registry_snapshot_sha256: str
    captured_at: str

    _registry_hash = field_validator("registry_snapshot_sha256")(_validate_sha256)


class TaskPlan(FrozenContractModel):
    schema_version: Literal["1.0"] = "1.0"
    plan_id: str
    project_id: str
    thread_id: str
    root_run_id: str
    goal: GoalBoundary
    complexity: Literal["complex"] = "complex"
    selected_capabilities: list[str] = Field(default_factory=list)
    steps: list[PlanStep]
    handoffs: list[CrossDomainHandoff] = Field(default_factory=list)
    execution_snapshot: PlanExecutionSnapshot
    retrieval_policy: Literal["conditional"] = "conditional"
    rollback_policy: Literal["step", "workflow", "manual"] = "step"
    created_at: str
    revision: int = Field(default=1, ge=1)
    plan_sha256: str
    supersedes_plan_id: str | None = None

    _plan_hash = field_validator("plan_sha256")(_validate_sha256)

    @model_validator(mode="after")
    def validate_graph(self) -> "TaskPlan":
        identifiers = [step.step_id for step in self.steps]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("step_id values must be unique")
        known = set(identifiers)
        dependencies = {step.step_id: set(step.dependencies) for step in self.steps}
        unknown = sorted({dep for values in dependencies.values() for dep in values if dep not in known})
        if unknown:
            raise ValueError(f"unknown step dependencies: {', '.join(unknown)}")

        dependents: dict[str, set[str]] = {identifier: set() for identifier in identifiers}
        indegree = {identifier: len(values) for identifier, values in dependencies.items()}
        for identifier, values in dependencies.items():
            for dependency in values:
                dependents[dependency].add(identifier)
        ready = deque(identifier for identifier, degree in indegree.items() if degree == 0)
        visited_count = 0
        while ready:
            identifier = ready.popleft()
            visited_count += 1
            for dependent in dependents[identifier]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
        if visited_count != len(identifiers):
            raise ValueError("step dependencies must be acyclic")

        capability_ids = {
            snapshot.capability_id for snapshot in self.execution_snapshot.capability_snapshots
        }
        missing_capabilities = sorted(
            {step.capability for step in self.steps if step.capability not in capability_ids}
        )
        if missing_capabilities:
            raise ValueError(
                "missing capability snapshot(s): " + ", ".join(missing_capabilities)
            )

        for handoff in self.handoffs:
            missing_steps = sorted(
                step_id
                for step_id in (handoff.source_step_id, handoff.target_step_id)
                if step_id not in known
            )
            if missing_steps:
                raise ValueError(
                    "handoff references unknown step(s): " + ", ".join(missing_steps)
                )
            if (
                handoff.conversion is not None
                and handoff.conversion.converter_capability not in capability_ids
            ):
                raise ValueError(
                    "missing capability snapshot(s): "
                    + handoff.conversion.converter_capability
                )

        workflow_keys = {
            (snapshot.workflow_id, snapshot.version)
            for snapshot in self.execution_snapshot.workflow_snapshots
        }
        missing_workflows = sorted(
            {
                (step.workflow_id, step.workflow_version)
                for step in self.steps
                if step.executor_type == "workflow"
                and (step.workflow_id, step.workflow_version) not in workflow_keys
            }
        )
        if missing_workflows:
            rendered = ", ".join(f"{workflow_id}@{version}" for workflow_id, version in missing_workflows)
            raise ValueError(f"missing workflow snapshot(s): {rendered}")
        return self


class PlanStepExecutionState(ContractModel):
    step_id: str
    status: Literal[
        "pending",
        "ready",
        "running",
        "completed",
        "failed",
        "blocked",
        "interrupted",
        "rolled_back",
        "blocked_snapshot_mismatch",
    ]
    attempts: StrictInt = Field(default=0, ge=0)
    last_input_hash: str | None = None
    last_output_refs: list[str] = Field(default_factory=list)
    last_checkpoint_id: str | None = None

    _input_hash = field_validator("last_input_hash")(
        lambda value: _validate_sha256(value) if value is not None else value
    )


class PlanExecutionState(ContractModel):
    project_id: str
    thread_id: str
    root_run_id: str
    plan_id: str
    plan_sha256: str
    step_states: list[PlanStepExecutionState]
    updated_at: str

    _plan_hash = field_validator("plan_sha256")(_validate_sha256)


class CheckResult(FrozenContractModel):
    check_id: str
    passed: bool
    severity: Literal["info", "warning", "error", "critical"]
    message: str
    evidence_refs: list[str] = Field(default_factory=list)


class DomainReview(FrozenContractModel):
    domain: str
    status: Literal["passed", "partial", "failed", "not_run"]
    validator: str
    checks: list[CheckResult] = Field(default_factory=list)


class ReviewResult(FrozenContractModel):
    schema_version: Literal["1.0"] = "1.0"
    review_id: str
    project_id: str
    thread_id: str
    plan_id: str
    root_run_id: str
    plan_sha256: str
    status: Literal["passed", "partial", "failed", "needs_confirmation"]
    goal_satisfied: bool
    boundary_compliant: bool
    constraints_satisfied: bool
    checkpoint_valid: bool
    evidence_sufficient: bool
    tool_execution_safe: bool
    checks: list[CheckResult] = Field(default_factory=list)
    domain_reviews: list[DomainReview] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    verified_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    recommended_action: Literal[
        "respond", "replan", "retry", "rollback", "request_confirmation", "stop"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reviewed_at: str

    _plan_hash = field_validator("plan_sha256")(_validate_sha256)

    @model_validator(mode="after")
    def validate_passed_claim(self) -> "ReviewResult":
        if self.status != "passed":
            return self
        required = (
            self.goal_satisfied,
            self.boundary_compliant,
            self.constraints_satisfied,
            self.checkpoint_valid,
            self.evidence_sufficient,
            self.tool_execution_safe,
        )
        if not all(required):
            raise ValueError("passed review requires all completion gates to pass")
        if any(not check.passed and check.severity == "critical" for check in self.checks):
            raise ValueError("passed review cannot contain a failed critical check")
        if any(review.status == "failed" for review in self.domain_reviews):
            raise ValueError("passed review cannot contain a failed domain review")
        if any(
            not check.passed and check.severity == "critical"
            for review in self.domain_reviews
            for check in review.checks
        ):
            raise ValueError("passed review cannot contain a failed critical domain check")
        if self.recommended_action != "respond":
            raise ValueError("passed review must recommend a response")
        return self


class ExecutionRun(ContractModel):
    root_run_id: str
    project_id: str
    thread_id: str | None
    kind: Literal["user", "scheduled"]
    continuation_of: str | None = None
    retrieval_budget: StrictInt = Field(ge=0)
    retrieval_state: Literal[
        "unavailable", "available", "claimed", "in_flight", "consumed", "consumed_unknown"
    ]
    retrieval_reason: str = ""
    retrieval_query_hash: str | None = None
    retrieval_claimed_at: str | None = None
    retrieval_lease_expires_at: str | None = None
    retrieval_attempt_started_at: str | None = None
    retrieval_claimer_id: str | None = None
    version: StrictInt = Field(ge=0)

    _query_hash = field_validator("retrieval_query_hash")(
        lambda value: _validate_sha256(value) if value is not None else value
    )

    @model_validator(mode="after")
    def validate_run_kind(self) -> "ExecutionRun":
        if self.kind == "scheduled":
            if self.retrieval_budget != 0 or self.retrieval_state != "unavailable":
                raise ValueError("scheduled execution cannot use private RAG")
            retrieval_trace = (
                self.retrieval_query_hash,
                self.retrieval_claimed_at,
                self.retrieval_lease_expires_at,
                self.retrieval_attempt_started_at,
                self.retrieval_claimer_id,
            )
            if self.retrieval_reason or any(value is not None for value in retrieval_trace):
                raise ValueError("scheduled execution cannot contain private RAG trace state")
        else:
            if self.thread_id is None or self.retrieval_budget != 1:
                raise ValueError("user execution requires a thread and one retrieval budget")
            claim_fields = (
                self.retrieval_query_hash,
                self.retrieval_claimed_at,
                self.retrieval_lease_expires_at,
                self.retrieval_claimer_id,
            )
            if self.retrieval_state in {"unavailable", "available"}:
                if any(value is not None for value in (*claim_fields, self.retrieval_attempt_started_at)):
                    raise ValueError("unclaimed retrieval state cannot contain claim trace fields")
            elif self.retrieval_state == "claimed":
                if any(value is None for value in claim_fields):
                    raise ValueError("claimed retrieval requires query, claim, lease, and claimer fields")
                if self.retrieval_attempt_started_at is not None:
                    raise ValueError("claimed retrieval cannot already have an attempt start")
            elif self.retrieval_state == "in_flight":
                if any(value is None for value in claim_fields) or self.retrieval_attempt_started_at is None:
                    raise ValueError("in-flight retrieval requires complete claim and attempt trace fields")
            elif self.retrieval_state in {"consumed", "consumed_unknown"}:
                if any(value is None for value in claim_fields) or self.retrieval_attempt_started_at is None:
                    raise ValueError("consumed retrieval requires complete claim and attempt trace fields")
        return self


class CheckpointRef(FrozenContractModel):
    project_id: str
    thread_id: str
    checkpoint_id: str


class SessionMemory(ContractModel):
    memory_id: str
    project_id: str
    thread_id: str
    kind: Literal["fact", "preference", "decision", "constraint", "task_state", "artifact", "open_item"]
    content: str
    source_checkpoints: list[CheckpointRef] = Field(min_length=1)
    source_content_hash: str
    truth_status: Literal["user_stated", "verified", "assumption", "superseded", "retracted"]
    confidence: float = Field(ge=0.0, le=1.0)
    supersedes: str | None = None
    created_at: str
    updated_at: str

    _content_hash = field_validator("source_content_hash")(_validate_sha256)

    @model_validator(mode="after")
    def validate_checkpoint_namespace(self) -> "SessionMemory":
        if any(
            checkpoint.project_id != self.project_id or checkpoint.thread_id != self.thread_id
            for checkpoint in self.source_checkpoints
        ):
            raise ValueError("memory checkpoint namespace mismatch")
        if self.supersedes == self.memory_id:
            raise ValueError("memory cannot supersede itself")
        return self


class ArtifactRef(FrozenContractModel):
    uri: str
    sha256: str
    media_type: str
    byte_length: StrictInt = Field(ge=0)

    _content_hash = field_validator("sha256")(_validate_sha256)


class ArtifactSchemaManifest(FrozenContractModel):
    schema_id: str
    schema_version: str
    payload_json_schema: dict[str, JsonValue]
    required_quantity_fields: list[str]
    requires_frame: bool
    requires_time_system: bool
    requires_epoch_field: bool


class DomainArtifact(FrozenContractModel):
    artifact_id: str
    payload_ref: ArtifactRef
    schema_id: str
    schema_version: str
    metadata: DomainDataMetadata
    source_capability: CapabilitySnapshot
    source_checkpoints: list[CheckpointRef] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)


class DomainExecutionOutput(FrozenContractModel):
    artifacts: list[DomainArtifact]
    observation: str = ""


class HandoffRecord(FrozenContractModel):
    handoff_id: str
    plan_id: str
    source_step_id: str
    target_step_id: str
    source_artifact_ids: list[str] = Field(min_length=1)
    target_artifact_ids: list[str] = Field(default_factory=list)
    conversion_capability: str | None = None
    validation_check_ids: list[str]
    checkpoint: CheckpointRef


class WorkflowStepPolicy(FrozenContractModel):
    step_id: str
    executor_type: Literal["tool", "domain", "human", "capability_builder"]
    capability_id: str
    risk_level: Literal["read_only", "project_write", "high_risk"]
    recovery_class: Literal["read_only", "reversible", "compensatable", "manual_recovery"]
    idempotent: bool


class WorkflowManifest(FrozenContractModel):
    workflow_id: str
    version: str
    workflow_schema_version: str
    input_schema: dict[str, JsonValue]
    steps: list[WorkflowStepPolicy]
    workflow_sha256: str
    manifest_sha256: str
    approval_record_id: str
    approval_scope: Literal["interactive_only", "scheduled_read_only"]
    automatable: bool = False

    _workflow_hash = field_validator("workflow_sha256")(_validate_sha256)
    _manifest_hash = field_validator("manifest_sha256")(_validate_sha256)

    @model_validator(mode="after")
    def validate_automation(self) -> "WorkflowManifest":
        if self.automatable:
            if self.approval_scope != "scheduled_read_only":
                raise ValueError("automatable workflow needs scheduled_read_only approval")
            if not self.steps or any(
                step.risk_level != "read_only"
                or step.recovery_class != "read_only"
                or not step.idempotent
                or step.executor_type in {"human", "capability_builder"}
                for step in self.steps
            ):
                raise ValueError("automatable workflow may contain only idempotent read-only steps")
        return self


class DependencyCandidate(FrozenContractModel):
    name: str
    source_type: Literal["installed", "package", "git", "local_code"]
    source_uri: str | None = None
    version_or_commit: str | None = None
    license: str | None = None
    compatibility: Literal["compatible", "uncertain", "incompatible"]
    maintenance_status: Literal["active", "unknown", "inactive"]
    risks: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)


class CapabilityGap(FrozenContractModel):
    capability_id: str
    requested_by_step_id: str
    description: str
    required_contract: dict[str, JsonValue]
    candidates: list[DependencyCandidate] = Field(default_factory=list)
    resolution: Literal[
        "use_installed",
        "install_package",
        "integrate_git",
        "implement_tool",
        "implement_subgraph",
        "defer",
    ] | None = None


__all__ = [
    "ArtifactRef",
    "ArtifactSchemaManifest",
    "CapabilityGap",
    "CapabilityManifest",
    "CapabilitySnapshot",
    "CheckpointRef",
    "CheckResult",
    "ConfirmationGrant",
    "ContractModel",
    "CrossDomainHandoff",
    "DependencyCandidate",
    "DomainArtifact",
    "DomainDataMetadata",
    "DomainExecutionOutput",
    "DomainReview",
    "ExecutionRun",
    "FrozenContractModel",
    "GoalBoundary",
    "HandoffRecord",
    "HandoffConversion",
    "PlanExecutionSnapshot",
    "PlanExecutionState",
    "PlanStep",
    "PlanStepExecutionState",
    "ReviewResult",
    "SessionMemory",
    "TaskPlan",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "VerificationCheck",
    "WorkflowSnapshot",
    "WorkflowManifest",
    "WorkflowStepPolicy",
]
