"""Self-contained Agent Core contracts and services."""

from .models import *  # noqa: F401,F403
from .models import __all__ as _MODEL_EXPORTS
from .approval import CapabilityApprovalLedger, CapabilityApprovalVerifier
from .artifacts import ArtifactStore
from .capabilities import CapabilityRegistry
from .confirmation import ConfirmationService, compute_action_hash
from .execution import (
    AuthorizedExecutor,
    ExecutionContext,
    ExecutionRegistry,
    ExecutionRequest,
    ExecutionService,
)
from .routing import CapabilityRoute, CapabilityRouter
from .project_memory import ProjectIdentityService, ProjectStatus
from .session_memory import SessionMemoryService, SessionSummary
from .rag_gate import ExecutionRunStore, RagGateDecision, RagGateService, decide_private_rag
from .context_assembler import MemoryContextAssembler, MemoryContextSections
from .dag import CheckpointedDAGExecutor, DAGExecutionOutcome
from .execution_checkpoints import ExecutionCheckpointStore
from .git_service import GitService
from .planning import PlanExecutionVerifier, build_task_plan, compute_task_plan_sha256
from .review import ReviewAssessment, ReviewService
from .scheduler import SchedulerService
from .workflows import WorkflowRegistry

__all__ = [
    *_MODEL_EXPORTS,
    "ArtifactStore",
    "CapabilityApprovalLedger",
    "CapabilityApprovalVerifier",
    "CapabilityRegistry",
    "CapabilityRoute",
    "CapabilityRouter",
    "ConfirmationService",
    "AuthorizedExecutor",
    "ExecutionContext",
    "ExecutionRegistry",
    "ExecutionRequest",
    "ExecutionService",
    "ProjectIdentityService",
    "ProjectStatus",
    "SessionMemoryService",
    "SessionSummary",
    "MemoryContextAssembler",
    "MemoryContextSections",
    "CheckpointedDAGExecutor",
    "DAGExecutionOutcome",
    "ExecutionCheckpointStore",
    "GitService",
    "PlanExecutionVerifier",
    "ReviewAssessment",
    "ReviewService",
    "SchedulerService",
    "WorkflowRegistry",
    "build_task_plan",
    "compute_task_plan_sha256",
    "ExecutionRunStore",
    "RagGateDecision",
    "RagGateService",
    "decide_private_rag",
    "compute_action_hash",
]
