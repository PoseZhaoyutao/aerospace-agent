"""Deterministic LangGraph runtime for the aerospace agent.

The graph keeps runtime clients in a closure (``ServiceBundle``) and exposes
only JSON/checkpoint-safe values through :class:`AerospaceAgentState`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from .nodes import (
    capability_route_node,
    classify_intent_node,
    core_direct_execute_node,
    core_execute_review_node,
    core_plan_node,
    core_rag_gate_node,
    context_compress_node,
    evidence_gate_node,
    evaluate_node,
    observe_node,
    planner_node,
    rag_retrieve_node,
    synthesize_node,
    tool_execute_node,
    tool_select_node,
    validate_output_node,
)
from .state import AerospaceAgentState, create_initial_state


@dataclass(frozen=True)
class ServiceBundle:
    """Runtime-only dependencies injected into graph node closures.

    None of these objects is written to graph state or checkpoints.  The
    planner and gateway protocols are intentionally duck-typed so deterministic
    test doubles and production services can share this boundary.
    """

    knowledge: Any = None
    context: Any = None
    evolution: Any = None
    planner: Any = None
    mcp_gateway: Any = None
    llm: Any = None
    model_name: str = ""
    endpoint: str = ""
    runtime_warnings: tuple[Any, ...] = ()
    # Optional safety policy and human-confirmation gate.  Kept out of the
    # serialized graph state; these are runtime-only dependencies like the
    # planner and MCP gateway.
    safety: Any = None
    approval_gate: Any = None
    # Agent Core services.  ``agent_core_enabled`` is set by the agent only
    # after the project identity manifest and databases validate as ready.
    agent_core_enabled: bool = False
    project_id: str = ""
    capability_router: Any = None
    route_classifier: Any = None
    execution_run_store: Any = None
    rag_gate: Any = None
    direct_executor: Any = None
    task_plan_service: Any = None
    dag_executor: Any = None
    review_service: Any = None
    review_assessor: Any = None
    # Concrete production composition.  The catalog/registry/service startup
    # view is workspace-bound; direct execution creates a thread-bound view so
    # SessionMemoryService is never shared across threads.
    agent_core_runtime: Any = None
    core_tool_services: Any = None
    core_tool_catalog: Any = None
    capability_registry: Any = None
    execution_registry: Any = None
    execution_service: Any = None
    plan_execution_verifier: Any = None
    workflow_registry: Any = None
    scheduler_service: Any = None
    git_service: Any = None
    evolution_candidate_service: Any = None
    capability_acquisition_service: Any = None
    integration_trust_service: Any = None

    def initial_input(
        self,
        message: str,
        max_steps: int = 15,
        thread_id: str = "test",
    ) -> AerospaceAgentState:
        """Return a fresh graph input without serializing this bundle."""

        return initial_input(message, max_steps=max_steps, thread_id=thread_id)


def _bundle(
    services: ServiceBundle | None,
    *,
    llm: Any,
    rag: Any,
    available_tools: Any,
) -> ServiceBundle:
    """Normalize modern service injection and legacy graph arguments."""

    if services is not None:
        return services
    # ``rag`` and ``available_tools`` are retained for callers of the original
    # graph API.  A mapping/callable is accepted directly by tool execution.
    return ServiceBundle(knowledge=rag, mcp_gateway=available_tools, llm=llm)


def initial_input(
    message: str,
    max_steps: int = 15,
    *,
    thread_id: str = "test",
) -> AerospaceAgentState:
    """Create a checkpoint-safe graph input with one human message."""

    state = create_initial_state(thread_id=thread_id, max_steps=max_steps)
    state["messages"] = [HumanMessage(content=message)]
    return state


def _route_after_retrieve(state: AerospaceAgentState) -> str:
    """Return retrieved evidence to the planner for the next decision."""

    return "planner"


_WORK_INTENTS = {
    "orbit_design",
    "orbit_propagation",
    "launch_window",
    "lunar_transfer",
    "maneuver_planning",
    "tool_discovery",
}


def _route_after_evidence_gate(
    state: AerospaceAgentState,
) -> str:
    """Bypass RAG unless this turn explicitly requires private evidence."""

    if state.get("retrieval_required"):
        return "retrieve"
    if state.get("intent") in _WORK_INTENTS:
        return "plan"
    return "respond"


def _route_after_planner(state: AerospaceAgentState) -> str:
    """Route the planner's versioned decision without implicit tool hops."""

    if state.get("status") in {"partial", "cycle_detected", "error", "interrupted"}:
        return "synthesize"
    if state.get("termination_reason"):
        return "synthesize"
    decision = state.get("decision") or {}
    action = decision.get("action") if isinstance(decision, dict) else None
    if action == "retrieve":
        return "retrieve"
    if action == "call_tool":
        return "tool"
    return "synthesize"


def _route_after_evaluate(state: AerospaceAgentState) -> str:
    """Route terminal outcomes to synthesis and continue live plans."""

    if state.get("status") in {"partial", "cycle_detected", "error", "interrupted"}:
        return "synthesize"
    if state.get("termination_reason"):
        return "synthesize"
    decision = state.get("decision") or {}
    action = decision.get("action") if isinstance(decision, dict) else None
    if action in {"respond", "stop"}:
        return "synthesize"
    return "planner"


def _route_after_capability(state: AerospaceAgentState) -> str:
    route = (state.get("capability_route") or {}).get("route")
    return {
        "conversation": "synthesize",
        "knowledge_qa": "rag_gate",
        "direct_execution": "direct",
        "complex_task": "plan",
        "memory_operation": "synthesize",
        "project_operation": "synthesize",
        "clarify": "synthesize",
    }.get(str(route), "synthesize")


def _route_after_core_plan(state: AerospaceAgentState) -> str:
    if state.get("status") in {"partial", "error", "interrupted"}:
        return "synthesize"
    return "rag_gate"


def _route_after_core_rag_gate(state: AerospaceAgentState) -> str:
    if state.get("retrieval_required"):
        return "retrieve"
    route = (state.get("capability_route") or {}).get("route")
    return "execute_review" if route == "complex_task" else "synthesize"


def _route_after_core_retrieve(state: AerospaceAgentState) -> str:
    route = (state.get("capability_route") or {}).get("route")
    return "execute_review" if route == "complex_task" else "synthesize"


def _build_agent_core_graph(
    *,
    bundle: ServiceBundle,
    checkpointer: Optional[BaseCheckpointSaver],
    retrieval_confidence_threshold: float,
    interrupt_before: tuple[str, ...] | list[str] | None,
) -> Any:
    """Build the project-initialized Agent Core route graph."""

    graph = StateGraph(AerospaceAgentState)
    graph.add_node("context_compress", lambda state: context_compress_node(state, max_tokens=4096, services=bundle))
    graph.add_node("capability_route", lambda state: capability_route_node(state, services=bundle))
    graph.add_node(
        "core_rag_gate",
        lambda state: core_rag_gate_node(
            state,
            confidence_threshold=retrieval_confidence_threshold,
        ),
    )
    graph.add_node("rag_retrieve", lambda state: rag_retrieve_node(state, services=bundle))
    graph.add_node("core_direct_execute", lambda state: core_direct_execute_node(state, services=bundle))
    graph.add_node("core_plan", lambda state: core_plan_node(state, services=bundle))
    graph.add_node("core_execute_review", lambda state: core_execute_review_node(state, services=bundle))
    graph.add_node("synthesize", lambda state: synthesize_node(state, llm=bundle.llm, services=bundle))
    graph.set_entry_point("context_compress")
    graph.add_edge("context_compress", "capability_route")
    graph.add_conditional_edges(
        "capability_route",
        _route_after_capability,
        {
            "synthesize": "synthesize",
            "rag_gate": "core_rag_gate",
            "direct": "core_direct_execute",
            "plan": "core_plan",
        },
    )
    graph.add_edge("core_direct_execute", "synthesize")
    graph.add_conditional_edges(
        "core_plan",
        _route_after_core_plan,
        {"rag_gate": "core_rag_gate", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges(
        "core_rag_gate",
        _route_after_core_rag_gate,
        {"retrieve": "rag_retrieve", "execute_review": "core_execute_review", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges(
        "rag_retrieve",
        _route_after_core_retrieve,
        {"execute_review": "core_execute_review", "synthesize": "synthesize"},
    )
    graph.add_edge("core_execute_review", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer, interrupt_before=list(interrupt_before or ()))


def build_aerospace_graph(
    llm: Any = None,
    rag: Any = None,
    available_tools: Optional[Dict[str, Any]] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    max_steps: int = 15,
    max_repeats: int = 3,
    max_recursion_depth: int = 10,
    use_llm_intent: bool = False,
    retrieval_confidence_threshold: float = 0.60,
    cycle_max_repeats: int | None = None,
    *,
    services: ServiceBundle | None = None,
    interrupt_before: tuple[str, ...] | list[str] | None = None,
) -> Any:
    """Build and compile the deterministic aerospace graph.

    ``max_recursion_depth`` is retained as a compatibility argument; the
    caller's LangGraph ``recursion_limit`` remains the outer safety bound.
    ``max_steps`` is copied into initial state by the caller and enforced by
    :func:`evaluate_node`.
    """

    if cycle_max_repeats is not None:
        max_repeats = int(cycle_max_repeats)
    bundle = _bundle(services, llm=llm, rag=rag, available_tools=available_tools)
    if bundle.agent_core_enabled:
        return _build_agent_core_graph(
            bundle=bundle,
            checkpointer=checkpointer,
            retrieval_confidence_threshold=retrieval_confidence_threshold,
            interrupt_before=interrupt_before,
        )
    graph = StateGraph(AerospaceAgentState)

    graph.add_node(
        "classify_intent",
        lambda state: classify_intent_node(
            state,
            llm=bundle.llm if bundle.llm is not None else llm,
            use_llm=use_llm_intent,
        ),
    )
    graph.add_node(
        "context_compress",
        lambda state: context_compress_node(state, max_tokens=4096, services=bundle),
    )
    graph.add_node(
        "rag_retrieve",
        lambda state: rag_retrieve_node(state, rag=bundle.knowledge, services=bundle),
    )
    graph.add_node(
        "evidence_gate",
        lambda state: evidence_gate_node(
            state,
            confidence_threshold=retrieval_confidence_threshold,
        ),
    )
    graph.add_node(
        "tool_select",
        lambda state: tool_select_node(state, services=bundle),
    )
    graph.add_node("planner", lambda state: planner_node(state, services=bundle))
    graph.add_node(
        "tool_execute",
        lambda state: tool_execute_node(state, services=bundle, available_tools=available_tools),
    )
    graph.add_node("observe", observe_node)
    graph.add_node("validate_output", lambda state: validate_output_node(state, services=bundle))
    graph.add_node(
        "evaluate",
        lambda state: evaluate_node(state, max_repeats=max_repeats, max_steps=max_steps),
    )
    graph.add_node(
        "synthesize",
        lambda state: synthesize_node(
            state,
            llm=bundle.llm if bundle.llm is not None else llm,
            services=bundle,
        ),
    )

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "context_compress")
    graph.add_edge("context_compress", "evidence_gate")
    graph.add_conditional_edges(
        "evidence_gate",
        _route_after_evidence_gate,
        {"retrieve": "rag_retrieve", "plan": "planner", "respond": "synthesize"},
    )
    graph.add_edge("rag_retrieve", "planner")
    # Keep the historical selection/execute/validate topology.  ``observe``
    # remains registered for stream consumers, but the executor already emits
    # a serialized observation and bypasses that no-op hop in the bounded loop.
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"retrieve": "rag_retrieve", "tool": "tool_select", "synthesize": "synthesize"},
    )
    graph.add_edge("tool_select", "tool_execute")
    graph.add_edge("observe", "validate_output")
    graph.add_edge("tool_execute", "validate_output")
    graph.add_edge("validate_output", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {"planner": "planner", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer, interrupt_before=list(interrupt_before or ()))


def build_simple_graph(
    llm: Any = None,
    available_tools: Optional[Dict[str, Any]] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    *,
    services: ServiceBundle | None = None,
    interrupt_before: tuple[str, ...] | list[str] | None = None,
) -> Any:
    """Build the legacy short graph used by lightweight callers."""

    bundle = _bundle(services, llm=llm, rag=None, available_tools=available_tools)
    graph = StateGraph(AerospaceAgentState)
    graph.add_node("classify_intent", lambda state: classify_intent_node(state, llm=bundle.llm, use_llm=False))
    graph.add_node("tool_select", lambda state: tool_select_node(state, services=bundle))
    graph.add_node("tool_execute", lambda state: tool_execute_node(state, services=bundle, available_tools=available_tools))
    graph.add_node("synthesize", lambda state: synthesize_node(state, llm=bundle.llm, services=bundle))
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "tool_select")
    graph.add_edge("tool_select", "tool_execute")
    graph.add_edge("tool_execute", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer, interrupt_before=list(interrupt_before or ()))


__all__ = ["ServiceBundle", "initial_input", "build_aerospace_graph", "build_simple_graph"]
