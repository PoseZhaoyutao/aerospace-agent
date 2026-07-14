"""Deterministic completion review bound to one immutable plan and DAG state."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import Field

from .models import (
    CheckResult,
    ContractModel,
    DomainReview,
    PlanExecutionState,
    ReviewResult,
    TaskPlan,
    ToolResult,
)
from .planning import compute_task_plan_sha256

if False:  # pragma: no cover - imported lazily to avoid an eager module cycle
    from .dag import CheckpointedDAGExecutor


class ReviewAssessment(ContractModel):
    """Evidence supplied by explicit validators, never inferred from prose."""

    goal_satisfied: bool
    boundary_compliant: bool
    constraints_satisfied: bool
    evidence_sufficient: bool
    tool_execution_safe: bool
    checks: list[CheckResult] = Field(default_factory=list)
    domain_reviews: list[DomainReview] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    verified_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewService:
    """Apply the completion gates and produce one exact ``ReviewResult``."""

    def __init__(self, checkpoint_verifier: "CheckpointedDAGExecutor | None" = None) -> None:
        if checkpoint_verifier is not None:
            from .dag import CheckpointedDAGExecutor

            if not isinstance(checkpoint_verifier, CheckpointedDAGExecutor):
                raise TypeError("checkpoint_verifier must be CheckpointedDAGExecutor")
        self._checkpoint_verifier = checkpoint_verifier

    def review(
        self,
        *,
        plan: TaskPlan,
        state: PlanExecutionState,
        step_results: Mapping[str, ToolResult],
        assessment: ReviewAssessment,
    ) -> ReviewResult:
        checked_plan = TaskPlan.model_validate(plan.model_dump(mode="python"))
        checked_state = PlanExecutionState.model_validate(state.model_dump(mode="python"))
        checked_assessment = ReviewAssessment.model_validate(
            assessment.model_dump(mode="python")
        )
        results = {
            step_id: ToolResult.model_validate(result.model_dump(mode="python"))
            for step_id, result in step_results.items()
        }
        expected_identity = (
            checked_plan.project_id,
            checked_plan.thread_id,
            checked_plan.root_run_id,
            checked_plan.plan_id,
            checked_plan.plan_sha256,
        )
        actual_identity = (
            checked_state.project_id,
            checked_state.thread_id,
            checked_state.root_run_id,
            checked_state.plan_id,
            checked_state.plan_sha256,
        )
        identity_valid = (
            actual_identity == expected_identity
            and compute_task_plan_sha256(checked_plan) == checked_plan.plan_sha256
        )
        expected_step_ids = {step.step_id for step in checked_plan.steps}
        state_by_step = {step.step_id: step for step in checked_state.step_states}
        state_shape_valid = (
            len(state_by_step) == len(checked_state.step_states)
            and set(state_by_step) == expected_step_ids
            and set(results).issubset(expected_step_ids)
        )
        checkpoint_valid = (
            identity_valid
            and state_shape_valid
            and self._checkpoint_verifier is not None
            and self._checkpoint_verifier.verify_state(checked_plan, checked_state, results)
        )

        completed = (
            checkpoint_valid
            and set(results) == expected_step_ids
            and all(state_by_step[step_id].status == "completed" for step_id in expected_step_ids)
            and all(results[step_id].status == "success" for step_id in expected_step_ids)
        )
        confirmation_needed = any(
            result.status == "blocked"
            and result.error is not None
            and result.error.code == "confirmation_required"
            for result in results.values()
        )
        required_check_ids = {
            check.check_id
            for step in checked_plan.steps
            for check in step.verification
            if check.required
        }
        checks_by_id = {check.check_id: check for check in checked_assessment.checks}
        known_evidence_refs = {
            result.audit_id for result in results.values()
        } | {
            step.last_checkpoint_id
            for step in state_by_step.values()
            if step.last_checkpoint_id is not None
        }
        checks_complete = all(
            check_id in checks_by_id
            and checks_by_id[check_id].passed
            and bool(set(checks_by_id[check_id].evidence_refs) & known_evidence_refs)
            for check_id in required_check_ids
        )
        failed_critical = any(
            not check.passed and check.severity == "critical"
            for check in checked_assessment.checks
        )

        required_domains = {
            str(step.domain_subgraph)
            for step in checked_plan.steps
            if step.executor_type == "domain_subgraph"
        }
        derived_domain_reviews = (
            self._checkpoint_verifier.derive_domain_reviews(
                checked_plan, checked_state, results
            )
            if self._checkpoint_verifier is not None
            else []
        )
        derived_by_name = {review.domain: review for review in derived_domain_reviews}
        domain_reviews = [
            *derived_domain_reviews,
            *[
                review
                for review in checked_assessment.domain_reviews
                if review.domain not in required_domains
            ],
        ]
        domain_complete = all(
            domain in derived_by_name and derived_by_name[domain].status == "passed"
            for domain in required_domains
        )
        failed_domain = any(
            review.status == "failed"
            or any(not check.passed and check.severity == "critical" for check in review.checks)
            for review in domain_reviews
        )
        unsupported = list(checked_assessment.unsupported_claims)
        unresolved = list(checked_assessment.unresolved_items)
        if not identity_valid:
            unresolved.insert(0, "plan/state identity mismatch")
        elif not state_shape_valid:
            unresolved.insert(0, "plan/state step identity mismatch")
        if not checks_complete:
            unresolved.append("required verification checks did not all pass")
        if not domain_complete:
            unresolved.append("required domain reviews did not all pass")

        boundary = checked_assessment.boundary_compliant and identity_valid
        constraints = checked_assessment.constraints_satisfied and identity_valid
        evidence = (
            checked_assessment.evidence_sufficient
            and checks_complete
            and not unsupported
        )
        tool_safe = (
            checked_assessment.tool_execution_safe
            and identity_valid
            and not failed_critical
            and not failed_domain
        )
        completion_gates = (
            completed
            and checked_assessment.goal_satisfied
            and boundary
            and constraints
            and checkpoint_valid
            and evidence
            and tool_safe
            and domain_complete
            and not unresolved
            and not unsupported
        )
        goal_satisfied = checked_assessment.goal_satisfied and completed

        if confirmation_needed:
            status = "needs_confirmation"
            action = "request_confirmation"
        elif not identity_valid:
            status = "failed"
            action = "stop"
        elif failed_critical or failed_domain or not boundary or not tool_safe:
            status = "failed"
            action = "rollback"
        elif any(result.status == "failed" for result in results.values()):
            status = "failed"
            action = "rollback"
        elif completion_gates:
            status = "passed"
            action = "respond"
        else:
            status = "partial"
            action = (
                "retry"
                if any(
                    step.status in {"interrupted", "failed"}
                    for step in state_by_step.values()
                )
                else "replan"
            )

        return ReviewResult(
            review_id=f"review:{uuid4().hex}",
            project_id=checked_plan.project_id,
            thread_id=checked_plan.thread_id,
            plan_id=checked_plan.plan_id,
            root_run_id=checked_plan.root_run_id,
            plan_sha256=checked_plan.plan_sha256,
            status=status,
            goal_satisfied=goal_satisfied,
            boundary_compliant=boundary,
            constraints_satisfied=constraints,
            checkpoint_valid=checkpoint_valid,
            evidence_sufficient=evidence,
            tool_execution_safe=tool_safe,
            checks=checked_assessment.checks,
            domain_reviews=domain_reviews,
            unresolved_items=unresolved,
            verified_claims=checked_assessment.verified_claims,
            unsupported_claims=unsupported,
            recommended_action=action,
            confidence=checked_assessment.confidence,
            reviewed_at=datetime.now(UTC).isoformat(),
        )


__all__ = ["ReviewAssessment", "ReviewService"]
