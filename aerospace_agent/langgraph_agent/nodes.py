"""Small, serializable LangGraph nodes for the aerospace agent.

Nodes deliberately contain very little policy.  Runtime services are injected
by :func:`build_aerospace_graph`; node return values are checkpoint-safe JSON
structures (Pydantic models are dumped before crossing the state boundary).
"""
from __future__ import annotations

import hashlib
import inspect
import re
import time
from collections.abc import Mapping
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from .cycle_detector import fingerprint
from .agent_core.dag import DAGExecutionOutcome
from .agent_core.models import ReviewResult, TaskPlan, ToolResult
from .agent_core.rag_gate import RagGateDecision, decide_private_rag
from .agent_core.routing import CapabilityRoute
from .prompts import AEROSPACE_ASSISTANT_IDENTITY, sanitize_assistant_answer
from .router import route_intent
from .schema import ActionType, Decision, EvidenceItem, ToolCallRequest, ToolCallResponse
from .safety import ApprovalRequired, SafetyValidationError, SafetyValidator
from .state import AerospaceAgentState


def _dump(value: Any) -> Any:
    """Convert model/dataclass-like values to JSON-safe values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, Mapping):
        return {str(k): _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dump(item) for item in value]
    if hasattr(value, "as_dict"):
        return _dump(value.as_dict())
    if hasattr(value, "__dict__"):
        return _dump(vars(value))
    return str(value)


def _message_text(state: Mapping[str, Any]) -> str:
    for message in reversed(state.get("messages", []) or []):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            content = getattr(message, "content", message)
            return content if isinstance(content, str) else str(content)
    return ""


def _memory_context_text(state: Mapping[str, Any]) -> str:
    """Return prompt-only project/session evidence assembled for this turn."""

    sections: list[str] = []
    for message in state.get("messages", []) or []:
        content = getattr(message, "content", message)
        if not isinstance(content, str):
            continue
        if content.startswith(("[PROJECT MEMORY]", "[SESSION SUMMARY]", "[SESSION MEMORY]")):
            sections.append(content)
    return "\n\n".join(sections)


def _conversation_history_text(state: Mapping[str, Any], *, max_chars: int = 16_000) -> str:
    """Render bounded prior dialogue for the synthesis model.

    Checkpoint state contains the conversation, but the model only receives
    explicit prompt text.  Session-memory retrieval is intentionally selective
    and cannot replace ordinary conversational history (for example, a
    follow-up asking what the assistant said one turn ago).
    """

    messages = list(state.get("messages", []) or [])
    latest_human_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if getattr(messages[index], "type", "") in {"human", "user"}
        ),
        None,
    )
    rendered: list[str] = []
    for index, message in enumerate(messages):
        if index == latest_human_index:
            continue
        if bool(
            (getattr(message, "additional_kwargs", None) or {}).get(
                "agent_context_ephemeral",
                False,
            )
        ):
            continue
        content = getattr(message, "content", message)
        if not isinstance(content, str) or not content.strip():
            continue
        role = getattr(message, "type", "message")
        label = {
            "human": "User",
            "user": "User",
            "ai": "Assistant",
            "assistant": "Assistant",
            "system": "System",
        }.get(str(role), "Message")
        rendered.append(f"{label}: {content.strip()}")
    text = "\n".join(rendered)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


_EXPLICIT_EVIDENCE_PATTERNS = (
    r"依据",
    r"来源",
    r"引用",
    r"引文",
    r"私域知识库",
    r"知识库.{0,8}(?:核实|审查|验证)",
    r"(?:核实|审查).{0,8}(?:资料|来源|知识库)",
    r"\bcitations?\b",
    r"\bcite\b",
    r"\bsources?\b",
    r"\bevidence\b",
    r"private\s+knowledge\s+base",
    r"\b(?:verify|audit)\b.{0,24}\b(?:source|knowledge\s+base|evidence)\b",
    r"(?:核实|审查|验证)(?:一下)?(?:这|该|上述|以下)?.{0,12}(?:说法|主张|结论|声明)",
    r"\b(?:verify|audit)\b.{0,24}\b(?:claim|statement|conclusion|assertion)\b",
)

_NEGATED_EVIDENCE_PATTERNS = (
    r"(?:不要|无需|不用|不必|不需要|别)\s*(?:引用|提供|给出|使用|检索|查询|核实|审查|验证)?\s*(?:任何)?\s*(?:依据|来源|引用|引文|证据|私域知识库|知识库|核实|审查|验证)",
    r"\b(?:do\s+not|don't|no\s+need\s+to|need\s+no)\s+(?:cite|provide|use|search|verify|audit)?\s*(?:citations?|sources?|evidence|private\s+knowledge\s+base|rag)\b",
    r"\bno\s+(?:citations?|sources?|evidence)\s*(?:is|are)?\s*(?:needed|required)?\b",
    r"\b(?:citations?|sources?|evidence)\s+(?:is|are)\s+not\s+(?:needed|required|necessary)\b",
    r"\bwithout\s+(?:citations?|sources?|evidence)\b",
)


def requests_evidence(message: str) -> bool:
    """Return whether the user explicitly requested sourced verification."""

    text = str(message or "").strip()
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in _NEGATED_EVIDENCE_PATTERNS):
        return False
    bare_request = text.lower().strip(" \t\r\n.!?。！？,，:：;；")
    if bare_request in {"核实", "审查", "验证", "verify", "audit", "verification"}:
        return True
    return bool(text) and any(
        re.search(pattern, text, re.IGNORECASE) for pattern in _EXPLICIT_EVIDENCE_PATTERNS
    )


def evidence_gate_node(
    state: AerospaceAgentState,
    *,
    confidence_threshold: float = 0.60,
) -> dict[str, Any]:
    """Decide whether this turn requires private-knowledge evidence."""

    started = time.perf_counter()
    message = _message_text(state)
    explicit = requests_evidence(message)
    low_confidence_fact = (
        state.get("intent") == "knowledge_query"
        and float(state.get("intent_confidence", 0.0) or 0.0) < float(confidence_threshold)
    )
    reason = "explicit_evidence" if explicit else (
        "low_confidence" if low_confidence_fact else ""
    )
    return _finish(
        state,
        "evidence_gate",
        started,
        {
            "retrieval_required": bool(reason),
            "retrieval_reason": reason,
            "metrics": {
                "retrieval_required": bool(reason),
                "retrieval_reason": reason,
            },
        },
    )


def capability_route_node(state: AerospaceAgentState, services: Any = None) -> dict[str, Any]:
    """Produce one validated seven-way Agent Core route."""

    started = time.perf_counter()
    router = getattr(services, "capability_router", None) if services is not None else None
    if router is None:
        return _finish(
            state,
            "capability_route",
            started,
            {
                "status": "error",
                "termination_reason": "capability_router_unavailable",
                "errors": [
                    *list(state.get("errors", []) or []),
                    _error("capability_router_unavailable", "Agent Core router is unavailable", node="capability_route"),
                ],
            },
        )
    try:
        route_arguments = {
            "requested_tool_name": str(state.get("requested_tool_name", "") or "") or None,
            "parsed_arguments": (
                dict(state.get("parsed_arguments", {}) or {})
                if state.get("parsed_arguments")
                else None
            ),
            "arguments_validated": bool(state.get("arguments_validated", False)),
            "classifier": getattr(services, "route_classifier", None),
        }
        prepared_arguments: dict[str, Any] | None = None
        prepared_tool_name: str | None = None
        prepared_validated = False
        prepare_request = getattr(router, "prepare_request", None)
        if (
            callable(prepare_request)
            and not route_arguments["requested_tool_name"]
            and route_arguments["parsed_arguments"] is None
        ):
            prepared_tool_name, prepared_arguments, prepared_validated = prepare_request(
                _message_text(state)
            )
            if prepared_tool_name:
                route_arguments["requested_tool_name"] = prepared_tool_name
                route_arguments["parsed_arguments"] = prepared_arguments
                route_arguments["arguments_validated"] = bool(prepared_validated)
        route_for_state = getattr(router, "route_for_state", None)
        if callable(route_for_state):
            raw = route_for_state(
                state=dict(state),
                message=_message_text(state),
                **route_arguments,
            )
        else:
            raw = router.route(_message_text(state), **route_arguments)
        route = CapabilityRoute.model_validate(_dump(raw))
    except Exception as exc:
        return _finish(
            state,
            "capability_route",
            started,
            {
                "status": "error",
                "termination_reason": "capability_route_error",
                "errors": [
                    *list(state.get("errors", []) or []),
                    _error("protocol_error", str(exc), node="capability_route"),
                ],
            },
        )
    metrics = dict(state.get("metrics", {}) or {})
    metrics.update(
        {
            "capability_route": route.route,
            "capability_route_confidence": route.confidence,
            "capability_route_reason": route.reason,
        }
    )
    delta: dict[str, Any] = {
        "capability_route": route.model_dump(mode="json"),
        "intent": route.intent,
        "intent_confidence": route.confidence,
        "metrics": metrics,
    }
    if prepared_tool_name is not None and prepared_arguments is not None:
        delta["requested_tool_name"] = prepared_tool_name
        delta["parsed_arguments"] = prepared_arguments
        delta["arguments_validated"] = bool(prepared_validated)
    return _finish(
        state,
        "capability_route",
        started,
        delta,
    )


def core_rag_gate_node(
    state: AerospaceAgentState,
    *,
    confidence_threshold: float = 0.60,
) -> dict[str, Any]:
    """Apply the design's three positive triggers with denial precedence."""

    started = time.perf_counter()
    route = CapabilityRoute.model_validate(state.get("capability_route") or {})
    decision = decide_private_rag(
        route=route.route,
        confidence=route.confidence,
        user_text=_message_text(state),
        planner_request=str(state.get("planner_retrieval_request", "") or "") or None,
        confidence_threshold=confidence_threshold,
    )
    metrics = dict(state.get("metrics", {}) or {})
    metrics.update(
        {
            "retrieval_required": decision.retrieve,
            "retrieval_reason": decision.reason,
        }
    )
    return _finish(
        state,
        "core_rag_gate",
        started,
        {
            "rag_gate_decision": decision.model_dump(mode="json"),
            "retrieval_required": decision.retrieve,
            "retrieval_reason": decision.reason,
            "metrics": metrics,
        },
    )
def _answer_is_grounded(answer: str, evidence: list[Mapping[str, Any]]) -> bool:
    """Conservatively reject model prose that cannot be mapped to one excerpt.

    This is a deterministic provenance gate, not a claim of semantic
    entailment: every sentence must have substantial token overlap with one
    cited excerpt and any numeric literals must occur in that excerpt.
    """
    excerpts = [str(item.get("excerpt", "")) for item in evidence if item.get("excerpt")]
    if not excerpts:
        return False
    stopwords = {"the", "and", "for", "with", "from", "that", "this", "are", "is", "of", "to", "in", "on", "a", "an"}
    chunks = [part.strip(" -*\t") for part in re.split(r"[\n.!?]+", str(answer or "")) if part.strip()]
    for chunk in chunks:
        if chunk.lower().rstrip(":") in {"evidence", "answer", "response"}:
            continue
        tokens = {token for token in re.findall(r"[a-z0-9_]+", chunk.lower()) if len(token) > 2 and token not in stopwords}
        if not tokens:
            continue
        numbers = set(re.findall(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", chunk.lower()))
        grounded = False
        for excerpt in excerpts:
            cited = {token for token in re.findall(r"[a-z0-9_]+", excerpt.lower()) if len(token) > 2 and token not in stopwords}
            cited_numbers = set(re.findall(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", excerpt.lower()))
            if len(tokens & cited) / len(tokens) >= 0.60 and numbers.issubset(cited_numbers):
                grounded = True
                break
        if not grounded:
            return False
    return True


def _error(category: str, message: str, *, code: str | None = None, node: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"category": category, "message": str(message)}
    if code:
        result["code"] = code
    if node:
        result["node"] = node
    return result


def _finish(state: Mapping[str, Any], name: str, started: float, delta: dict[str, Any]) -> dict[str, Any]:
    """Attach timing and cumulative metrics to a node delta."""

    elapsed = max(0.0, (time.perf_counter() - started) * 1000.0)
    timings = dict(state.get("node_timings_ms", {}) or {})
    timings[name] = elapsed
    metrics = dict(state.get("metrics", {}) or {})
    # Nodes may contribute counters (for example ``rag_hits`` or tool
    # duration) in their delta.  Merge them before adding runtime metrics so
    # the contribution is not discarded at the state boundary.
    delta_metrics = delta.get("metrics")
    if isinstance(delta_metrics, Mapping):
        metrics.update(delta_metrics)
    metrics["node_timings_ms"] = timings
    metrics["total_duration_ms"] = float(metrics.get("total_duration_ms", 0.0) or 0.0) + elapsed
    delta["node_timings_ms"] = timings
    delta["metrics"] = metrics
    return delta


def _call_service(service: Any, method_names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    """Invoke the first available service method without inventing arguments."""

    if service is None:
        return None
    target = service
    for name in method_names:
        if hasattr(service, name):
            target = getattr(service, name)
            break
    if not callable(target):
        return None
    # Service doubles in tests intentionally expose different small APIs. Try
    # the full state first, then the user message; do not synthesize tool args.
    try:
        return target(*args, **kwargs)
    except TypeError:
        if args:
            try:
                return target(args[0])
            except TypeError:
                return target()
        return target()


def classify_intent_node(
    state: AerospaceAgentState,
    llm: Any = None,
    use_llm: bool = False,
    services: Any = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    message = _message_text(state)
    intent, confidence = route_intent(message, llm=llm, use_llm=use_llm)
    return _finish(
        state,
        "classify_intent",
        started,
        {
            "intent": intent,
            "intent_confidence": confidence,
        },
    )


def context_node(state: AerospaceAgentState, services: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    context = getattr(services, "context", None) if services is not None else None
    delta: dict[str, Any] = {}
    if context is not None:
        try:
            assembled = _call_service(
                context,
                ("assemble",),
                messages=list(state.get("messages", []) or []),
                tool_results=list(state.get("tool_results", []) or []),
                thread_id=str(state.get("thread_id", "") or ""),
                current_request=_message_text(state),
            )
            if assembled is not None:
                payload = _dump(assembled)
                if isinstance(payload, Mapping):
                    delta["context_summary"] = str(payload.get("summary", ""))
                    delta["artifact_refs"] = list(payload.get("artifact_refs", []))
                # ``messages`` are a message channel; preserve native messages
                # returned by ContextService rather than dumping them to JSON.
                messages = getattr(assembled, "messages", None)
                if messages is not None:
                    # ``messages`` uses LangGraph's add_messages reducer.  A
                    # compacted list therefore has to replace the channel
                    # explicitly; returning the list alone only appends/merges
                    # it and leaves every omitted message in the checkpoint.
                    delta["messages"] = [
                        RemoveMessage(id=REMOVE_ALL_MESSAGES),
                        *list(messages),
                    ]
        except Exception as exc:
            delta.setdefault("warnings", list(state.get("warnings", []) or []))
            delta["warnings"] = [*delta["warnings"], _error("protocol_error", str(exc), node="context")]
    return _finish(state, "context", started, delta)


def context_compress_node(state: AerospaceAgentState, max_tokens: int = 4096, services: Any = None) -> dict[str, Any]:
    """Compatibility alias for the old context node."""

    if services is not None:
        return context_node(state, services)
    started = time.perf_counter()
    return _finish(state, "context_compress", started, {"context_strategy": "essential", "context_summary": ""})


def rag_retrieve_node(state: AerospaceAgentState, rag: Any = None, services: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    knowledge = getattr(services, "knowledge", None) if services is not None else rag
    query = _message_text(state)
    query_hash = hashlib.sha256(
        " ".join(query.lower().split()).encode("utf-8")
    ).hexdigest()
    delta: dict[str, Any] = {
        "rag_query": query,
        "evidence": [],
        "rag_documents": [],
        "retrieval_attempted": True,
        "retrieval_query_hash": query_hash,
    }
    if knowledge is None:
        return _finish(state, "retrieve", started, delta)
    try:
        gate_service = getattr(services, "rag_gate", None) if services is not None else None
        run_store = getattr(services, "execution_run_store", None) if services is not None else None
        raw_decision = state.get("rag_gate_decision") or {}
        if gate_service is not None and run_store is not None and raw_decision:
            decision = RagGateDecision.model_validate(raw_decision)
            run = run_store.get(str(state.get("root_run_id") or state.get("run_id") or ""))
            raw = gate_service.retrieve_once(
                run=run,
                decision=decision,
                query=delta["rag_query"],
                claimer_id=f"graph:{state.get('run_id', '')}",
                retriever=lambda value: _call_service(knowledge, ("search",), value, top_k=5),
            )
        else:
            raw = _call_service(knowledge, ("search",), delta["rag_query"], top_k=5)
        evidence: list[dict[str, Any]] = []
        protocol_errors: list[dict[str, Any]] = []
        for item in raw or []:
            try:
                evidence.append(
                    EvidenceItem.model_validate(_dump(item)).model_dump(mode="json")
                )
            except Exception as exc:
                protocol_errors.append(
                    _error("retrieval_protocol_error", str(exc), node="retrieve")
                )
        delta["evidence"] = evidence
        delta["rag_documents"] = evidence
        metrics = dict(state.get("metrics", {}) or {})
        metrics["rag_hits"] = len(evidence)
        delta["metrics"] = metrics
        if protocol_errors:
            delta["errors"] = [
                *list(state.get("errors", []) or []),
                *protocol_errors,
            ]
        if not evidence:
            delta["status"] = "partial"
            delta["warnings"] = [
                *list(state.get("warnings", []) or []),
                "RAG verification returned no usable evidence",
            ]
    except Exception as exc:
        delta["errors"] = [*list(state.get("errors", []) or []), _error("retrieval_error", str(exc), node="retrieve")]
        delta["status"] = "partial"
        delta["termination_reason"] = "retrieval_error"
        delta["metrics"] = {**dict(state.get("metrics", {}) or {}), "rag_hits": 0}
        delta["warnings"] = [
            *list(state.get("warnings", []) or []),
            "RAG verification was unavailable",
        ]
    return _finish(state, "retrieve", started, delta)


def tool_select_node(state: AerospaceAgentState, available_tools: Any = None, services: Any = None) -> dict[str, Any]:
    """Compatibility node; modern graphs select tools through ``Decision``."""

    started = time.perf_counter()
    return _finish(state, "tool_select", started, {"tool_requests": list(state.get("tool_requests", []) or [])})


def _coerce_decision(raw: Any) -> Decision:
    if isinstance(raw, Decision):
        return raw
    if isinstance(raw, Mapping):
        return Decision.model_validate(raw)
    raise TypeError(f"planner returned unsupported decision type: {type(raw).__name__}")


def planner_node(state: AerospaceAgentState, services: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    previous_steps = int(state.get("step_count", 0) or 0)
    delta: dict[str, Any] = {"step_count": previous_steps + 1}
    planner = getattr(services, "planner", None) if services is not None else None
    if planner is None:
        if state.get("evidence"):
            decision = Decision(
                action=ActionType.RESPOND,
                rationale="Retrieved evidence can be rendered without a model planner",
            )
        else:
            decision = Decision(action=ActionType.STOP, rationale="No model planner is configured")
            delta.update({
                "status": "partial",
                "termination_reason": "planner_unavailable",
                "warnings": [
                    *list(state.get("warnings", []) or []),
                    "Model planner is unavailable; the work request was not planned",
                ],
            })
    else:
        try:
            raw = _call_service(planner, ("plan", "decide"), dict(state))
            decision = _coerce_decision(raw)
        except Exception as exc:
            delta["errors"] = [*list(state.get("errors", []) or []), _error("protocol_error", str(exc), node="planner")]
            delta["status"] = "error"
            delta["termination_reason"] = "planner_error"
            delta["decision"] = None
            return _finish(state, "planner", started, delta)
    if decision.action == ActionType.RETRIEVE:
        if state.get("retrieval_attempted"):
            decision = Decision(
                action=ActionType.RESPOND,
                rationale="The current query has already been retrieved; use the available result",
            )
            delta["retrieval_required"] = False
            delta["warnings"] = [
                *list(state.get("warnings", []) or []),
                "Repeated retrieval request stopped; the current query was already reviewed",
            ]
        else:
            delta["retrieval_required"] = True
            delta["retrieval_reason"] = "planner_request"
            planner_metrics = dict(state.get("metrics", {}) or {})
            planner_metrics.update(
                {
                    "retrieval_required": True,
                    "retrieval_reason": "planner_request",
                }
            )
            delta["metrics"] = planner_metrics
    decision_json = _dump(decision)
    delta["decision"] = decision_json
    if decision.action == ActionType.CALL_TOOL and decision.tool_request is not None:
        delta["tool_requests"] = [_dump(decision.tool_request)]
    else:
        delta["tool_requests"] = []
    return _finish(state, "planner", started, delta)


def _gateway_call(gateway: Any, request: ToolCallRequest) -> Any:
    if gateway is None:
        raise RuntimeError("MCP gateway is not configured")
    if hasattr(gateway, "call_tool"):
        return gateway.call_tool(request)
    if isinstance(gateway, Mapping):
        handler = gateway.get(request.tool_name)
        if handler is None:
            raise KeyError(f"unknown tool: {request.tool_name}")
        return handler(**dict(request.arguments))
    if callable(gateway):
        return gateway(request)
    raise TypeError(f"unsupported MCP gateway: {type(gateway).__name__}")


def _safety_validator(services: Any) -> SafetyValidator | Any:
    """Return the injected safety policy, or a default local validator."""

    configured = getattr(services, "safety", None) if services is not None else None
    if configured is not None:
        return configured
    return SafetyValidator(approval_gate=getattr(services, "approval_gate", None) if services is not None else None)


def tool_execute_node(state: AerospaceAgentState, services: Any = None, available_tools: Any = None, default_timeout: float = 30.0) -> dict[str, Any]:
    started = time.perf_counter()
    gateway = getattr(services, "mcp_gateway", None) if services is not None else available_tools
    decision_raw = state.get("decision")
    results = list(state.get("tool_results", []) or [])
    errors = list(state.get("errors", []) or [])
    delta: dict[str, Any] = {}
    request: ToolCallRequest | None = None
    try:
        decision = _coerce_decision(decision_raw) if decision_raw else None
        if decision is None or decision.action != ActionType.CALL_TOOL:
            return _finish(state, "tool_execute", started, {"tool_results": results})
        if decision.tool_request is None:
            raise ValueError("CALL_TOOL decision has no tool_request")
        request = decision.tool_request
        validator = _safety_validator(services)
        validator.validate_tool_request(
            request.tool_name,
            request.arguments,
            is_read_only=bool(getattr(request, "is_read_only", True)),
        )
        tool_started = time.perf_counter()
        raw_result = _gateway_call(gateway, request)
        payload = _dump(raw_result)
        if isinstance(raw_result, ToolCallResponse):
            result = payload
        elif isinstance(payload, Mapping):
            result = dict(payload)
            result.setdefault("tool_name", request.tool_name)
            result.setdefault("status", "success")
            result.setdefault("duration_ms", max(0.0, (time.perf_counter() - tool_started) * 1000.0))
            if result.get("status") != "success" and not result.get("error"):
                result["error"] = str(result.get("status"))
        else:
            result = {
                "tool_name": request.tool_name,
                "status": "success",
                "result": payload,
                "duration_ms": max(0.0, (time.perf_counter() - tool_started) * 1000.0),
            }
        if result.get("status") == "success":
            output_payload = result.get("result") if "result" in result else result
            validator.validate_tool_output(request.tool_name, output_payload)
        if result.get("status") != "success":
            errors.append(_error("tool_error", result.get("error", result.get("status", "tool failed")), node="tool_execute"))
        results.append(result)
    except (ApprovalRequired, SafetyValidationError) as exc:
        duration = max(0.0, (time.perf_counter() - (tool_started if "tool_started" in locals() else started)) * 1000.0)
        tool_name = request.tool_name if request is not None else ""
        results.append({"tool_name": tool_name, "status": "blocked", "error": str(exc), "error_code": "human_approval_required" if isinstance(exc, ApprovalRequired) else "safety_validation_failed", "duration_ms": duration})
        errors.append(_error("safety_error", str(exc), node="tool_execute"))
    except Exception as exc:
        duration = max(0.0, (time.perf_counter() - (tool_started if "tool_started" in locals() else started)) * 1000.0)
        tool_name = request.tool_name if request is not None else ""
        results.append({"tool_name": tool_name, "status": "error", "error": str(exc), "duration_ms": duration})
        errors.append(_error("tool_error", str(exc), node="tool_execute"))
    total_tool_ms = sum(float(item.get("duration_ms", 0.0) or 0.0) for item in results)
    metrics = dict(state.get("metrics", {}) or {})
    metrics["tool_duration_ms"] = total_tool_ms
    delta.update({"tool_results": results, "errors": errors, "metrics": metrics})
    if results:
        delta["observation"] = _dump(results[-1])
    return _finish(state, "tool_execute", started, delta)


def _legacy_tool_result(tool_name: str, result: ToolResult) -> dict[str, Any]:
    error = result.error.message if result.error is not None else None
    return {
        "tool_name": tool_name,
        "status": "success" if result.status == "success" else result.status,
        "result": dict(result.result),
        "error": error,
        "duration_ms": 0.0,
        "audit_id": result.audit_id,
        "operation_id": result.operation_id,
        "recovery_class": result.recovery_class,
    }


def core_direct_execute_node(state: AerospaceAgentState, services: Any = None) -> dict[str, Any]:
    """Invoke only a pre-wired trust-boundary facade for direct execution."""

    started = time.perf_counter()
    route = CapabilityRoute.model_validate(state.get("capability_route") or {})
    executor = getattr(services, "direct_executor", None) if services is not None else None
    if executor is None or not hasattr(executor, "execute_route"):
        return _finish(
            state,
            "core_direct_execute",
            started,
            {
                "status": "partial",
                "termination_reason": "direct_executor_unavailable",
                "warnings": [
                    *list(state.get("warnings", []) or []),
                    "Selected tool has no approved Agent Core execution adapter",
                ],
            },
        )
    try:
        raw = executor.execute_route(
            route=route,
            arguments=dict(state.get("parsed_arguments", {}) or {}),
            state=dict(state),
        )
        result = ToolResult.model_validate(_dump(raw))
    except Exception as exc:
        return _finish(
            state,
            "core_direct_execute",
            started,
            {
                "status": "error",
                "termination_reason": "direct_execution_error",
                "errors": [
                    *list(state.get("errors", []) or []),
                    _error("tool_error", str(exc), node="core_direct_execute"),
                ],
            },
        )
    tool_name = str(route.selected_executor_name or "")
    legacy = _legacy_tool_result(tool_name, result)
    delta: dict[str, Any] = {"tool_results": [legacy], "observation": legacy}
    if result.status != "success":
        delta.update(
            {
                "status": "partial" if result.status in {"blocked", "unavailable"} else "error",
                "termination_reason": f"direct_execution_{result.status}",
            }
        )
    return _finish(state, "core_direct_execute", started, delta)


def core_plan_node(state: AerospaceAgentState, services: Any = None) -> dict[str, Any]:
    """Create and validate the sole immutable TaskPlan for a complex run."""

    started = time.perf_counter()
    planner = getattr(services, "task_plan_service", None) if services is not None else None
    if planner is None:
        return _finish(
            state,
            "core_plan",
            started,
            {
                "status": "partial",
                "termination_reason": "task_plan_service_unavailable",
                "warnings": [
                    *list(state.get("warnings", []) or []),
                    "Complex task planning is unavailable",
                ],
            },
        )
    route = CapabilityRoute.model_validate(state.get("capability_route") or {})
    try:
        if hasattr(planner, "create_task_plan"):
            raw = planner.create_task_plan(route=route, state=dict(state))
        elif hasattr(planner, "plan"):
            raw = planner.plan(route=route, state=dict(state))
        else:
            raise TypeError("task plan service has no create_task_plan/plan method")
        payload = dict(raw) if isinstance(raw, Mapping) else {"task_plan": raw}
        raw_plan = payload.get("task_plan")
        if raw_plan is None:
            return _finish(
                state,
                "core_plan",
                started,
                {
                    "status": "partial",
                    "termination_reason": "task_plan_service_unavailable",
                    "warnings": [
                        *list(state.get("warnings", []) or []),
                        "The model did not produce a valid executable TaskPlan",
                    ],
                },
            )
        plan = TaskPlan.model_validate(_dump(raw_plan))
        expected = (
            str(state.get("project_id", "")),
            str(state.get("thread_id", "")),
            str(state.get("root_run_id") or state.get("run_id") or ""),
        )
        if (plan.project_id, plan.thread_id, plan.root_run_id) != expected:
            raise ValueError("TaskPlan namespace does not match the active run")
        retrieval_request = payload.get("retrieval_request")
        if retrieval_request not in {None, "", "retrieve"}:
            raise ValueError("planner retrieval request must be 'retrieve' or null")
        verifier = (
            getattr(services, "plan_execution_verifier", None)
            if services is not None
            else None
        )
        if verifier is None or not hasattr(verifier, "register_plan"):
            raise RuntimeError("TaskPlan verifier is unavailable")
        verifier.register_plan(plan)
    except Exception as exc:
        return _finish(
            state,
            "core_plan",
            started,
            {
                "status": "error",
                "termination_reason": "task_plan_error",
                "errors": [
                    *list(state.get("errors", []) or []),
                    _error("protocol_error", str(exc), node="core_plan"),
                ],
            },
        )
    return _finish(
        state,
        "core_plan",
        started,
        {
            "task_plan": plan.model_dump(mode="json"),
            "planner_retrieval_request": str(retrieval_request or ""),
        },
    )


def core_execute_review_node(state: AerospaceAgentState, services: Any = None) -> dict[str, Any]:
    """Execute a checkpointed DAG and require a bound ReviewResult."""

    started = time.perf_counter()
    executor = getattr(services, "dag_executor", None) if services is not None else None
    reviewer = getattr(services, "review_service", None) if services is not None else None
    assessor = getattr(services, "review_assessor", None) if services is not None else None
    if executor is None or reviewer is None:
        return _finish(
            state,
            "core_execute_review",
            started,
            {
                "status": "partial",
                "termination_reason": "complex_execution_unavailable",
                "warnings": [
                    *list(state.get("warnings", []) or []),
                    "Checkpointed DAG execution or completion review is unavailable",
                ],
            },
        )
    try:
        plan = TaskPlan.model_validate(state.get("task_plan") or {})
        execute = getattr(executor, "execute")
        parameters = inspect.signature(execute).parameters
        if "project_id" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            raw_outcome = execute(
                plan,
                project_id=str(state.get("project_id", "")),
                thread_id=str(state.get("thread_id", "")),
                root_run_id=str(state.get("root_run_id") or state.get("run_id") or ""),
            )
        else:
            # Compatibility for deterministic test doubles and older injected
            # executors; the production CheckpointedDAGExecutor takes the
            # explicit namespace above.
            raw_outcome = execute(plan)
        outcome = DAGExecutionOutcome.model_validate(_dump(raw_outcome))
        if outcome.plan_id != plan.plan_id or outcome.plan_sha256 != plan.plan_sha256:
            raise ValueError("DAG outcome does not match TaskPlan identity")
        if hasattr(reviewer, "review_outcome"):
            raw_review = reviewer.review_outcome(
                plan=plan,
                outcome=outcome,
                evidence=list(state.get("evidence", []) or []),
            )
        else:
            if outcome.state is None:
                raise ValueError(
                    "DAG outcome has no execution state"
                    + (f": {outcome.error}" if outcome.error else "")
                )
            if assessor is None or not hasattr(assessor, "assess"):
                raise RuntimeError("ReviewService requires an explicit review assessor")
            assessment = assessor.assess(
                plan=plan,
                outcome=outcome,
                evidence=list(state.get("evidence", []) or []),
            )
            raw_review = reviewer.review(
                plan=plan,
                state=outcome.state,
                step_results=outcome.step_results,
                assessment=assessment,
            )
        review = ReviewResult.model_validate(_dump(raw_review))
        if (
            review.project_id,
            review.thread_id,
            review.root_run_id,
            review.plan_id,
            review.plan_sha256,
        ) != (
            plan.project_id,
            plan.thread_id,
            plan.root_run_id,
            plan.plan_id,
            plan.plan_sha256,
        ):
            raise ValueError("ReviewResult does not match TaskPlan identity")
    except Exception as exc:
        return _finish(
            state,
            "core_execute_review",
            started,
            {
                "status": "error",
                "termination_reason": "complex_execution_error",
                "errors": [
                    *list(state.get("errors", []) or []),
                    _error("protocol_error", str(exc), node="core_execute_review"),
                ],
            },
        )
    tool_results = [
        _legacy_tool_result(
            next(
                (str(step.tool_name) for step in plan.steps if step.step_id == step_id),
                step_id,
            ),
            result,
        )
        for step_id, result in outcome.step_results.items()
    ]
    status = {
        "passed": "success",
        "needs_confirmation": "interrupted",
        "partial": "partial",
        "failed": "error",
    }[review.status]
    return _finish(
        state,
        "core_execute_review",
        started,
        {
            "plan_execution": outcome.model_dump(mode="json"),
            "review_result": review.model_dump(mode="json"),
            "tool_results": tool_results,
            "status": status,
            "termination_reason": "" if status == "success" else review.recommended_action,
            "is_complete": status == "success",
        },
    )


def observe_node(state: AerospaceAgentState) -> dict[str, Any]:
    started = time.perf_counter()
    results = list(state.get("tool_results", []) or [])
    return _finish(state, "observe", started, {"observation": _dump(results[-1]) if results else None})


def evaluate_node(
    state: AerospaceAgentState,
    cycle_detector: Any = None,
    llm: Any = None,
    *,
    max_repeats: int = 3,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Apply budget and repeated-state policy after one tool observation."""

    started = time.perf_counter()
    if cycle_detector is not None:
        max_repeats = int(getattr(cycle_detector, "max_repeats", max_repeats))
        if max_steps is None:
            max_steps = getattr(cycle_detector, "max_steps", None)
    decision = state.get("decision") or {}
    tool_request = decision.get("tool_request") if isinstance(decision, Mapping) else None
    tool_name = tool_request.get("tool_name") if isinstance(tool_request, Mapping) else None
    arguments = tool_request.get("arguments", {}) if isinstance(tool_request, Mapping) else {}
    # Include the logical target and the previous observation in the
    # production fingerprint.  Tool name + parameters alone is insufficient
    # for orbit tasks where the same operation is applied to different
    # spacecraft/epochs, and omitting observations causes distinct outcomes
    # to collapse into a false cycle.
    target = arguments.get("target", state.get("target")) if isinstance(arguments, Mapping) else state.get("target")
    observation = state.get("observation")
    if observation is None:
        prior_results = list(state.get("tool_results", []) or [])
        observation = prior_results[-1] if prior_results else None
    # Durations/timestamps are telemetry, not observations.  Strip them from
    # the fingerprint so the same physical tool result is recognized across
    # retries while substantive payload/units/frame changes remain distinct.
    if isinstance(observation, Mapping):
        observation = {
            key: value
            for key, value in observation.items()
            if str(key) not in {"duration_ms", "started_at", "finished_at", "run_id"}
        }
    fp = fingerprint(
        action="call_tool",
        tool_name=tool_name,
        target=target,
        params=arguments,
        observation=observation,
        intent=state.get("intent", ""),
    )
    history = list(state.get("state_fingerprints", []) or [])
    interventions = int(state.get("intervention_count", 0) or 0)
    step_count = int(state.get("step_count", 0) or 0)
    # A caller-provided state budget is authoritative; the graph-level value
    # is only a fallback for legacy/raw state dictionaries.
    state_budget = state.get("max_steps")
    configured_max_steps = int(state_budget if state_budget is not None else (max_steps or 15))
    delta: dict[str, Any] = {"state_fingerprints": [*history, fp], "cycle_hash_history": [*history, fp]}
    # Tool failures are terminal and remain structured in ``errors`` and
    # ``tool_results``; synthesis will preserve the error status.
    latest_result = (state.get("tool_results", []) or [])[-1:] if state.get("tool_results") else []
    if latest_result and isinstance(latest_result[0], Mapping) and latest_result[0].get("status") != "success":
        delta.update({"status": "error", "termination_reason": "tool_error", "is_complete": False})
    elif step_count >= configured_max_steps:
        delta.update({"status": "partial", "termination_reason": "max_steps", "is_complete": False})
    # The first duplicate receives one intervention; a subsequent duplicate
    # after that intervention is terminal.  ``max_repeats`` controls how many
    # occurrences are allowed before the terminal transition (the default of
    # three gives one intervention plus one final retry).
    elif fp in history:
        occurrences = history.count(fp) + 1
        if interventions < 1 and occurrences < max(2, int(max_repeats)):
            delta["intervention_count"] = 1
            delta["cycle_count"] = int(state.get("cycle_count", 0) or 0) + 1
            warnings = list(state.get("warnings", []) or [])
            warnings.append("Repeated action detected; planner intervention requested")
            delta["warnings"] = warnings
            delta["messages"] = [AIMessage(content="[SYSTEM INTERVENTION] Change the action or decomposition.")]
        else:
            delta.update(
                {
                    "status": "cycle_detected",
                    "termination_reason": "repeated_state_after_intervention",
                    "is_complete": False,
                    "intervention_count": max(1, interventions),
                }
            )
    else:
        delta["intervention_count"] = interventions
    if "status" not in delta and state.get("status") == "error":
        delta["status"] = "error"
    metrics = dict(state.get("metrics", {}) or {})
    metrics["intervention_count"] = int(delta.get("intervention_count", interventions) or 0)
    metrics["steps"] = step_count
    delta["metrics"] = metrics
    return _finish(state, "evaluate", started, delta)


def validate_output_node(state: AerospaceAgentState, services: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        _safety_validator(services).validate_state_output(state)
        return _finish(state, "validate_output", started, {"output_valid": True})
    except SafetyValidationError as exc:
        errors = [*list(state.get("errors", []) or []), _error("safety_error", str(exc), node="validate_output")]
        return _finish(state, "validate_output", started, {
            "output_valid": False,
            "status": "error",
            "termination_reason": "safety_validation_failed",
            "errors": errors,
        })


def synthesize_node(state: AerospaceAgentState, llm: Any = None, services: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    evidence = list(state.get("evidence", []) or [])
    results = list(state.get("tool_results", []) or [])
    question = _message_text(state)
    status = state.get("status")
    if status not in {"cycle_detected", "partial", "error", "interrupted"}:
        status = "success"
    verification_unavailable = bool(state.get("retrieval_required")) and not evidence
    if verification_unavailable and status == "success":
        status = "partial"
    lines: list[str] = []
    if evidence:
        lines.append("Evidence:")
        lines.extend(f"- {item.get('excerpt', '')}" for item in evidence if isinstance(item, Mapping))
    for item in results:
        if isinstance(item, Mapping) and item.get("status") == "success":
            lines.append(f"{item.get('tool_name', 'tool')}: {item.get('result', '')}")
    if verification_unavailable:
        lines.append(
            "Private-knowledge verification was requested, but no usable evidence was available. "
            "Please clarify the claim or restore the knowledge source before treating it as verified."
        )
    elif not lines:
        lines.append(f"No dialogue-model response is available for: {question}")
    answer = "\n".join(lines)
    grounding_warning = False
    # When a local OpenAI-compatible model is explicitly injected, let it
    # synthesize from the retrieved excerpts. Deterministic/offline tests pass
    # ``llm=None`` and retain the exact evidence rendering above. Any model
    # failure falls back to the evidence-only answer rather than inventing
    # unsupported content.
    if llm is not None and evidence:
        evidence_text = "\n".join(
            f"- {item.get('excerpt', '')}" for item in evidence if isinstance(item, Mapping)
        )
        prompt = (
            "Answer the user's aerospace question using only the evidence below. "
            "Preserve equations exactly, state assumptions explicitly, and do not add facts.\n\n"
            f"Question: {question}\nEvidence:\n{evidence_text}"
        )
        try:
            response = llm.chat(
                prompt,
                system_prompt=AEROSPACE_ASSISTANT_IDENTITY,
                max_tokens=512,
                temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            try:
                response = llm.chat(prompt, max_tokens=512, temperature=0.0)
            except Exception:
                response = ""
        except Exception:
            response = ""
        if isinstance(response, str) and response.strip():
            candidate = sanitize_assistant_answer(response.strip())
            if _answer_is_grounded(candidate, [item for item in evidence if isinstance(item, Mapping)]):
                answer = candidate
            else:
                # Keep the deterministic evidence rendering and expose the
                # rejection in the output instead of returning ungrounded
                # model prose.
                grounding_warning = True
    elif llm is not None and not results and not verification_unavailable:
        memory_context = _memory_context_text(state)
        conversation_history = _conversation_history_text(state)
        history_block = (
            "Recent conversation history (use it to resolve follow-up references):\n"
            f"{conversation_history}\n\n"
            if conversation_history
            else ""
        )
        context_block = (
            "The following project/session memory is evidence for this turn. "
            "Use it when relevant, but do not treat assumptions as verified facts:\n"
            f"{memory_context}\n\n"
            if memory_context
            else ""
        )
        prompt = (
            "Respond directly to the user. This turn did not request private-knowledge "
            "verification, so do not claim that a private source was consulted. "
            "State uncertainty explicitly when appropriate.\n\n"
            f"{history_block}"
            f"{context_block}"
            f"User: {question}"
        )
        try:
            response = llm.chat(
                prompt,
                system_prompt=AEROSPACE_ASSISTANT_IDENTITY,
                max_tokens=512,
                temperature=0.2,
                chat_template_kwargs={"enable_thinking": False},
            )
        except TypeError:
            try:
                response = llm.chat(prompt, max_tokens=512, temperature=0.2)
            except Exception:
                response = ""
        except Exception:
            response = ""
        if isinstance(response, str) and response.strip():
            answer = sanitize_assistant_answer(response.strip())
    delta: dict[str, Any] = {
        "final_answer": answer,
        "status": status,
        "is_complete": status == "success",
        "citations": evidence,
    }
    if grounding_warning:
        delta["warnings"] = [*list(state.get("warnings", []) or []), "LLM answer rejected by citation-grounding gate"]
    metrics = dict(state.get("metrics", {}) or {})
    metrics.update({
        "rag_hits": len(evidence),
        "intervention_count": int(state.get("intervention_count", 0) or 0),
        "cycle_count": int(state.get("cycle_count", 0) or 0),
        "cycle_triggers": int(state.get("cycle_count", 0) or 0),
        "steps": int(state.get("step_count", 0) or 0),
        "status": status,
        "run_id": str(state.get("run_id", "")),
        "thread_id": str(state.get("thread_id", "")),
    })
    if services is not None:
        metrics["model_name"] = str(getattr(services, "model_name", "") or "")
        metrics["endpoint"] = str(getattr(services, "endpoint", "") or "")
    delta["metrics"] = metrics
    return _finish(state, "synthesize", started, delta)


__all__ = [
    "capability_route_node", "classify_intent_node", "context_compress_node", "context_node",
    "core_direct_execute_node", "core_execute_review_node", "core_plan_node", "core_rag_gate_node",
    "evidence_gate_node",
    "requests_evidence", "rag_retrieve_node",
    "tool_select_node", "planner_node", "tool_execute_node", "observe_node", "evaluate_node",
    "validate_output_node", "synthesize_node",
]
