"""Serializable LangGraph state contract.

Only values that can be persisted by LangGraph are kept here.  Runtime
services (database handles, clients, or indexes) belong in graph context and
must never be inserted into this TypedDict.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AerospaceAgentState(TypedDict, total=False):
    """State channels shared by all graph nodes and checkpoints."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    run_id: str
    root_run_id: str
    project_id: str
    thread_id: str
    intent: str
    intent_confidence: float
    decision: Optional[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    retrieval_required: bool
    retrieval_reason: str
    retrieval_attempted: bool
    retrieval_query_hash: str
    tool_requests: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    artifact_refs: List[Dict[str, Any]]
    state_fingerprints: List[str]
    step_count: int
    intervention_count: int
    termination_reason: str
    max_steps: int

    # Existing serialized channels retained for graph compatibility.
    tool_calls: List[Dict[str, Any]]
    rag_documents: List[Dict[str, str]]
    rag_query: str
    context_summary: str
    context_strategy: str
    schema_errors: List[str]
    input_valid: bool
    output_valid: bool
    recursion_depth: int
    cycle_hash_history: List[str]
    cycle_count: int
    max_cycles: int
    final_answer: str
    is_complete: bool
    skill_metrics: Dict[str, Any]

    # Stable runtime/output protocol channels.  Values are deliberately
    # primitive containers so checkpoints can be serialized and replayed.
    status: str
    metrics: Dict[str, Any]
    errors: List[Dict[str, Any]]
    warnings: List[Any]
    node_timings_ms: Dict[str, float]
    citations: List[Dict[str, Any]]
    observation: Any

    # Agent Core orchestration channels.  These contain only validated JSON
    # snapshots; service handles and executors remain in ``ServiceBundle``.
    capability_route: Dict[str, Any]
    requested_tool_name: str
    parsed_arguments: Dict[str, Any]
    arguments_validated: bool
    confirmation_id: str
    planner_retrieval_request: str
    rag_gate_decision: Dict[str, Any]
    task_plan: Dict[str, Any]
    plan_execution: Dict[str, Any]
    review_result: Dict[str, Any]
    # Explicit outer AgentLoop lifecycle metadata.  Values are primitive so
    # they survive SQLite checkpoint serialization and can be audited after a
    # process restart.
    turn_state: str
    turn_state_history: List[str]
    turn_restored_checkpoint_id: str


def create_initial_state(
    thread_id: str = "default",
    run_id: Optional[str] = None,
    max_steps: int = 15,
    max_cycles: int = 15,
    **legacy: Any,
) -> AerospaceAgentState:
    """Create a checkpoint-safe initial state.

    ``max_recursion_depth`` was an implementation detail of the old graph. It
    is accepted as a legacy keyword so existing callers do not crash, but is
    intentionally not persisted in the state.  ``recursion_limit`` remains a
    top-level LangGraph invocation option and is not state.
    """
    # Accepted for compatibility with the old constructor, but intentionally
    # not persisted in checkpoint state (LangGraph recursion_limit is an
    # invocation-level bound).
    legacy.pop("max_recursion_depth", None)
    del legacy
    selected_run_id = run_id or uuid4().hex
    return AerospaceAgentState(
        messages=[],
        run_id=selected_run_id,
        root_run_id=selected_run_id,
        project_id="",
        thread_id=thread_id,
        intent="general",
        intent_confidence=0.0,
        decision=None,
        evidence=[],
        retrieval_required=False,
        retrieval_reason="",
        retrieval_attempted=False,
        retrieval_query_hash="",
        tool_requests=[],
        tool_results=[],
        artifact_refs=[],
        state_fingerprints=[],
        step_count=0,
        intervention_count=0,
        termination_reason="",
        max_steps=max_steps,
        tool_calls=[],
        rag_documents=[],
        rag_query="",
        context_summary="",
        context_strategy="essential",
        schema_errors=[],
        input_valid=True,
        output_valid=True,
        recursion_depth=0,
        cycle_hash_history=[],
        cycle_count=0,
        max_cycles=max_cycles,
        final_answer="",
        is_complete=False,
        skill_metrics={},
        status="",
        metrics={},
        errors=[],
        warnings=[],
        node_timings_ms={},
        citations=[],
        observation=None,
        capability_route={},
        requested_tool_name="",
        parsed_arguments={},
        arguments_validated=False,
        confirmation_id="",
        planner_retrieval_request="",
        rag_gate_decision={},
        task_plan={},
        plan_execution={},
        review_result={},
        turn_state="restore",
        turn_state_history=[],
        turn_restored_checkpoint_id="",
    )
