from __future__ import annotations

from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

from aerospace_agent.langgraph_agent.graph import build_aerospace_graph
from aerospace_agent.langgraph_agent.nodes import synthesize_node
import aerospace_agent.langgraph_agent.nodes as graph_nodes
from aerospace_agent.langgraph_agent.state import create_initial_state
from aerospace_agent.langgraph_agent.services.planner import LLMPlanner
from aerospace_agent.langgraph_agent.router import classify_intent_keyword
from .conftest import initial_input


def _invoke(services, message, *, thread_id="test", max_steps=15):
    graph = build_aerospace_graph(services=services, checkpointer=InMemorySaver(), max_steps=max_steps)
    return graph.invoke(
        initial_input(message, thread_id=thread_id, max_steps=max_steps),
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 20},
    )


def test_knowledge_query_retrieves_and_terminates(services):
    state = _invoke(services, "Please cite evidence for two-body dynamics.", thread_id="k1")
    assert state["status"] == "success"
    assert state["evidence"]
    assert state["step_count"] < 10
    assert state["metrics"]["rag_hits"] >= 1
    assert state["node_timings_ms"]
    assert state["run_id"] and state["thread_id"] == "k1"


def test_repeated_planner_is_intervened_then_terminated(services_with_repeating_planner):
    state = _invoke(services_with_repeating_planner, "Check engine availability", thread_id="repeat")
    assert state["status"] == "cycle_detected"
    assert state["intervention_count"] == 1
    assert state["termination_reason"] == "repeated_state_after_intervention"


def test_max_steps_returns_partial(services_with_repeating_planner):
    state = _invoke(
        services_with_repeating_planner,
        "Check engine availability",
        thread_id="limit",
        max_steps=2,
    )
    assert state["status"] == "partial"
    assert state["step_count"] == 2
    assert state["termination_reason"] == "max_steps"


def test_failing_tool_is_structured(services_with_failing_tool):
    state = _invoke(services_with_failing_tool, "Check engine availability", thread_id="fail")
    assert state["errors"][0]["category"] == "tool_error"
    assert state["tool_results"][0]["status"] == "error"
    assert state["metrics"]["tool_duration_ms"] >= 0


def test_llm_synthesis_falls_back_when_claims_are_not_cited():
    class InventingLLM:
        def chat(self, *_args, **_kwargs):
            return "A 999 km Mars orbit is guaranteed safe."

    state = create_initial_state(thread_id="grounding")
    state["messages"] = [HumanMessage(content="State the governing acceleration.")]
    state["evidence"] = [{
        "excerpt": "The governing acceleration is -mu r / |r|^3.",
        "source_id": "seed",
        "page_path": "knowledge/orbital-dynamics/two-body-orbital-dynamics.md",
    }]
    result = synthesize_node(state, llm=InventingLLM())
    assert "-mu r / |r|^3" in result["final_answer"]
    assert "LLM answer rejected" in result["warnings"][-1]


def test_llm_synthesis_receives_recent_conversation_history_for_followups():
    class HistoryProbeLLM:
        def __init__(self):
            self.prompts = []

        def chat(self, prompt, **_kwargs):
            self.prompts.append(prompt)
            return "I can continue the conversation."

    llm = HistoryProbeLLM()
    state = create_initial_state(thread_id="history-probe")
    state["messages"] = [
        HumanMessage(content="We are designing an optical navigation test."),
        graph_nodes.AIMessage(content="The next step is to define the observation contract."),
        HumanMessage(content="What did you say the next step was?"),
    ]

    result = synthesize_node(state, llm=llm)

    assert result["final_answer"] == "I can continue the conversation."
    assert llm.prompts
    assert "We are designing an optical navigation test." in llm.prompts[0]
    assert "The next step is to define the observation contract." in llm.prompts[0]
    assert "What did you say the next step was?" in llm.prompts[0]


def test_stream_text_uses_verified_graph_path(agent_factory, make_services):
    class StreamBypassProbe:
        def chat(self, *_args, **_kwargs):
            return "The governing acceleration is -mu r / |r|^3."

        def stream_chat(self, *_args, **_kwargs):
            raise AssertionError("raw model streaming must not bypass the graph")

    services = replace(make_services(), llm=StreamBypassProbe())
    agent = agent_factory(services=services)
    try:
        answer = "".join(agent.stream_text("State the governing acceleration.", thread_id="safe-stream"))
    finally:
        agent.close()
    assert "-mu r / |r|^3" in answer


def test_agent_close_closes_injected_gateway_and_checkpointer_once(agent_factory, make_services):
    class Gateway:
        closed = False

        def __init__(self):
            self.close_calls = 0

        def list_tools(self):
            return []

        def call_tool(self, request):
            raise AssertionError(f"unexpected tool call: {request}")

        def close(self):
            self.close_calls += 1
            self.closed = True

    class CloseCountingSaver(InMemorySaver):
        def __init__(self):
            super().__init__()
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    gateway = Gateway()
    checkpointer = CloseCountingSaver()
    services = replace(make_services(), mcp_gateway=gateway)
    agent = agent_factory(services=services, checkpointer=checkpointer)

    agent.close()
    agent.close()

    assert gateway.close_calls == 1
    assert checkpointer.close_calls == 1


def _gate_state(message: str, *, intent: str, confidence: float):
    state = create_initial_state(thread_id="gate")
    state["messages"] = [HumanMessage(content=message)]
    state["intent"] = intent
    state["intent_confidence"] = confidence
    return state


def test_general_low_confidence_does_not_require_rag():
    delta = graph_nodes.evidence_gate_node(
        _gate_state("你好，我们先聊聊任务安排", intent="general", confidence=0.0),
        confidence_threshold=0.60,
    )

    assert delta["retrieval_required"] is False
    assert delta["retrieval_reason"] == ""


def test_confident_knowledge_query_does_not_require_rag():
    delta = graph_nodes.evidence_gate_node(
        _gate_state("What is two-body dynamics?", intent="knowledge_query", confidence=0.90),
        confidence_threshold=0.60,
    )

    assert delta["retrieval_required"] is False


def test_low_confidence_knowledge_query_requires_rag():
    delta = graph_nodes.evidence_gate_node(
        _gate_state("这个航天概念具体是什么意思？", intent="knowledge_query", confidence=0.40),
        confidence_threshold=0.60,
    )

    assert delta["retrieval_required"] is True
    assert delta["retrieval_reason"] == "low_confidence"


@pytest.mark.parametrize(
    "message",
    [
        "请给出依据和来源后再回答",
        "Verify this against the private knowledge base and cite evidence",
        "请验证这一说法",
        "请核实该结论",
        "verify this claim",
        "audit this statement",
        "核实",
        "审查",
        "验证",
        "verify",
        "audit",
    ],
)
def test_explicit_evidence_request_requires_rag(message):
    delta = graph_nodes.evidence_gate_node(
        _gate_state(message, intent="general", confidence=0.95),
        confidence_threshold=0.60,
    )

    assert delta["retrieval_required"] is True
    assert delta["retrieval_reason"] == "explicit_evidence"


@pytest.mark.parametrize(
    "message",
    [
        "不要引用私域知识库，直接回答",
        "无需来源，直接回答",
        "不要核实该结论",
        "Do not cite sources; answer directly",
        "No evidence needed",
        "不需要来源，直接回答",
        "Evidence is not necessary",
        "Citations are not needed",
    ],
)
def test_negative_evidence_instruction_bypasses_rag(message):
    delta = graph_nodes.evidence_gate_node(
        _gate_state(message, intent="general", confidence=0.95),
        confidence_threshold=0.60,
    )

    assert delta["retrieval_required"] is False
    assert delta["retrieval_reason"] == ""


class CountingKnowledge:
    def __init__(self, results=()):
        self.calls = []
        self.results = list(results)

    def search(self, query, *, top_k=5):
        self.calls.append((query, top_k))
        return list(self.results)


class DirectConversationLLM:
    def __init__(self, response="Hello. I can help organize the work."):
        self.response = response
        self.calls = []

    def chat(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if "意图分类器" in prompt:
            return '{"intent": "general", "confidence": 0.95}'
        return self.response


class RespondingPlanner:
    def __init__(self):
        self.states = []

    def plan(self, state):
        self.states.append(dict(state))
        return graph_nodes.Decision(
            action=graph_nodes.ActionType.RESPOND,
            rationale="No tool or evidence is required",
        )


def _seed_evidence():
    return {
        "source_id": "wiki:seed:two_body_dynamics",
        "page_path": "knowledge/orbital-dynamics/two-body-orbital-dynamics.md",
        "chunk_id": "seed:two_body_dynamics:chunk-0",
        "score": 0.98,
        "excerpt": "The governing acceleration is -mu r / |r|^3.",
    }


def test_general_conversation_bypasses_rag_and_uses_llm(make_services):
    knowledge = CountingKnowledge([_seed_evidence()])
    llm = DirectConversationLLM()
    planner = RespondingPlanner()
    services = replace(make_services(), knowledge=knowledge, llm=llm, planner=planner)

    state = _invoke(services, "你好，我们先聊聊任务安排", thread_id="chat-no-rag")

    assert knowledge.calls == []
    assert "retrieve" not in state["node_timings_ms"]
    assert state["evidence"] == []
    assert state["final_answer"] == llm.response
    assert planner.states == []
    assert llm.calls[0][1]["temperature"] == 0.0


def test_research_work_phrase_is_not_misrouted_as_runge_kutta():
    intent, confidence = classify_intent_keyword(
        "Hello. Let us organize today's research work."
    )

    assert intent == "general"
    assert confidence == 0.0


@pytest.mark.parametrize("message", ["Use RK integration", "Use RK4 integration", "Use RK45"])
def test_explicit_runge_kutta_terms_route_to_propagation(message):
    intent, _confidence = classify_intent_keyword(message)

    assert intent == "orbit_propagation"


@pytest.mark.parametrize(
    "message",
    [
        "Please give advice on writing.",
        "We are studying geology.",
        "Enlist the project risks.",
        "请帮助我润色论文",
        "这个功能不好用",
        "我的能力有限",
        "List the project risks",
        "The dataset is available",
        "My availability is Friday",
        "不要传播未经核实的消息",
        "请机动安排一下会议",
        "Integrate the reviewer feedback into the paper.",
        "Please define the project scope.",
        "The army will maneuver tomorrow.",
        "为什么这个项目延期",
    ],
)
def test_domain_substrings_do_not_misroute_general_language(message):
    intent, confidence = classify_intent_keyword(message)

    assert intent == "general"
    assert confidence == 0.0


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Design a GEO orbit", "orbit_design"),
        ("Plan a delta-v maneuver", "maneuver_planning"),
        ("List available tools", "tool_discovery"),
        ("Numerically integrate the orbit", "orbit_propagation"),
        ("Plan a spacecraft maneuver", "maneuver_planning"),
        ("Define two-body orbital dynamics", "knowledge_query"),
    ],
)
def test_bounded_domain_terms_still_route(message, expected):
    intent, _confidence = classify_intent_keyword(message)

    assert intent == expected


def test_confident_knowledge_query_can_bypass_rag(make_services):
    knowledge = CountingKnowledge([_seed_evidence()])
    llm = DirectConversationLLM("Two-body dynamics is a central-force model.")
    planner = RespondingPlanner()
    services = replace(make_services(), knowledge=knowledge, llm=llm, planner=planner)

    state = _invoke(services, "What is two-body dynamics?", thread_id="fact-no-rag")

    assert knowledge.calls == []
    assert state["retrieval_required"] is False
    assert state["citations"] == []
    assert state["final_answer"] == llm.response
    assert planner.states == []


def test_direct_offline_response_does_not_imply_rag_was_searched(make_services):
    state = _invoke(make_services(), "你好", thread_id="offline-chat-no-rag")

    assert state["retrieval_attempted"] is False
    assert "evidence was found" not in state["final_answer"].lower()
    assert "检索" not in state["final_answer"]


def test_work_intent_reaches_planner_without_preplanning_rag(make_services):
    knowledge = CountingKnowledge([_seed_evidence()])
    planner = RespondingPlanner()
    services = replace(make_services(), knowledge=knowledge, planner=planner)

    state = _invoke(services, "Propagate this orbit for 24 hours", thread_id="work-no-rag")

    assert planner.states
    assert knowledge.calls == []
    assert "retrieve" not in state["node_timings_ms"]


def test_work_intent_without_model_planner_is_partial(make_services):
    services = replace(make_services(), planner=None, llm=None)

    state = _invoke(services, "Propagate this orbit for 24 hours", thread_id="work-no-planner")

    assert state["status"] == "partial"
    assert state["termination_reason"] == "planner_unavailable"
    assert any("planner" in str(item).lower() for item in state["warnings"])


def test_llm_planner_returns_validated_retrieval_decision():
    class PlannerLLM:
        def __init__(self):
            self.calls = []

        def chat(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return '{"action":"retrieve","rationale":"The force model is uncertain"}'

    llm = PlannerLLM()
    planner = LLMPlanner(llm, tool_names=["propagate_orbit"])
    state = create_initial_state(thread_id="model-planner")
    state["messages"] = [HumanMessage(content="Propagate this orbit")]
    state["intent"] = "orbit_propagation"

    decision = planner.plan(state)

    assert decision.action == graph_nodes.ActionType.RETRIEVE
    assert llm.calls
    assert llm.calls[0][1]["temperature"] == 0.0


def test_llm_planner_rejects_tool_outside_dispatch_allowlist():
    class PlannerLLM:
        def chat(self, prompt, **kwargs):
            return (
                '{"action":"call_tool","rationale":"run it",'
                '"tool_request":{"tool_name":"unknown_tool","arguments":{}}}'
            )

    planner = LLMPlanner(PlannerLLM(), tool_names=["propagate_orbit"])

    with pytest.raises(ValueError, match="not available"):
        planner.plan(create_initial_state(thread_id="model-planner-tool"))


def test_llm_planner_observes_tool_results_before_next_decision():
    class PlannerLLM:
        def __init__(self):
            self.prompts = []

        def chat(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return '{"action":"respond","rationale":"tool result is sufficient"}'

    llm = PlannerLLM()
    planner = LLMPlanner(llm, tool_names=["propagate_orbit"])
    before = create_initial_state(thread_id="planner-observation")
    before["messages"] = [HumanMessage(content="Propagate this orbit")]
    after = dict(before)
    after["tool_results"] = [{
        "tool_name": "propagate_orbit",
        "status": "success",
        "result": {"final_altitude_km": 512.5, "trace": "x" * 7000},
    }]
    after["observation"] = {"summary": "Propagation completed"}
    after["decision"] = {"action": "call_tool"}
    after["step_count"] = 2

    planner.plan(before)
    planner.plan(after)

    assert llm.prompts[0] != llm.prompts[1]
    assert "final_altitude_km" in llm.prompts[1]
    assert "Propagation completed" in llm.prompts[1]
    assert "call_tool" in llm.prompts[1]


def test_low_confidence_fact_query_calls_rag_once(make_services):
    knowledge = CountingKnowledge([_seed_evidence()])
    services = replace(make_services(), knowledge=knowledge, llm=None)

    state = _invoke(services, "What is orbit design?", thread_id="low-confidence-rag")

    assert len(knowledge.calls) == 1
    assert state["retrieval_reason"] == "low_confidence"
    assert state["citations"]


def test_explicit_evidence_request_calls_rag_once(make_services):
    knowledge = CountingKnowledge([_seed_evidence()])
    services = replace(make_services(), knowledge=knowledge, llm=None)

    state = _invoke(
        services,
        "请根据私域知识库给出二体动力学依据和来源",
        thread_id="explicit-rag",
    )

    assert len(knowledge.calls) == 1
    assert state["retrieval_reason"] == "explicit_evidence"
    assert state["citations"]


class RetrieveThenRespondPlanner:
    def __init__(self):
        self.states = []

    def plan(self, state):
        self.states.append(dict(state))
        if not state.get("evidence"):
            return graph_nodes.Decision(
                action=graph_nodes.ActionType.RETRIEVE,
                rationale="Private evidence is required before responding",
            )
        return graph_nodes.Decision(
            action=graph_nodes.ActionType.RESPOND,
            rationale="Retrieved evidence is sufficient",
        )


class AlwaysRetrievePlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, state):
        self.calls += 1
        return graph_nodes.Decision(
            action=graph_nodes.ActionType.RETRIEVE,
            rationale="Request the same evidence again",
        )


def test_planner_request_calls_rag_once_then_receives_evidence(make_services):
    knowledge = CountingKnowledge([_seed_evidence()])
    planner = RetrieveThenRespondPlanner()
    services = replace(make_services(), knowledge=knowledge, planner=planner)

    state = _invoke(services, "Propagate this orbit and check an uncertain model detail", thread_id="planner-rag")

    assert len(knowledge.calls) == 1
    assert len(planner.states) == 2
    assert planner.states[0]["evidence"] == []
    assert planner.states[1]["evidence"]
    assert state["retrieval_reason"] == "planner_request"
    assert state["retrieval_attempted"] is True
    assert state["retrieval_query_hash"]


def test_repeated_planner_retrieval_is_stopped_without_second_search(make_services):
    knowledge = CountingKnowledge([_seed_evidence()])
    planner = AlwaysRetrievePlanner()
    services = replace(make_services(), knowledge=knowledge, planner=planner)

    state = _invoke(services, "Propagate this orbit and check an uncertain model detail", thread_id="planner-rag-repeat")

    assert len(knowledge.calls) == 1
    assert planner.calls == 2
    assert state["status"] == "success"
    assert any("retrieval" in str(item).lower() for item in state["warnings"])


class FailingKnowledge:
    def __init__(self):
        self.calls = 0

    def search(self, query, *, top_k=5):
        self.calls += 1
        raise RuntimeError("private index unavailable")


def test_explicit_retrieval_exception_returns_partial_without_citations(make_services):
    knowledge = FailingKnowledge()
    services = replace(make_services(), knowledge=knowledge, llm=None)

    state = _invoke(
        services,
        "请根据私域知识库给出依据",
        thread_id="rag-exception",
    )

    assert knowledge.calls == 1
    assert state["status"] == "partial"
    assert state["citations"] == []
    assert state["errors"][0]["category"] == "retrieval_error"
    assert "verification" in state["final_answer"].lower()


def test_explicit_retrieval_no_hits_returns_partial_without_citations(make_services):
    knowledge = CountingKnowledge([])
    services = replace(make_services(), knowledge=knowledge, llm=None)

    state = _invoke(
        services,
        "Please cite evidence from the private knowledge base",
        thread_id="rag-empty",
    )

    assert len(knowledge.calls) == 1
    assert state["status"] == "partial"
    assert state["citations"] == []
    assert any("no usable evidence" in str(item).lower() for item in state["warnings"])
    assert "verification" in state["final_answer"].lower()


def test_malformed_retrieval_evidence_is_rejected(make_services):
    knowledge = CountingKnowledge([{"excerpt": "missing provenance"}])
    services = replace(make_services(), knowledge=knowledge, llm=None)

    state = _invoke(
        services,
        "Please provide cited evidence",
        thread_id="rag-malformed",
    )

    assert state["status"] == "partial"
    assert state["evidence"] == []
    assert state["citations"] == []
    assert any(error["category"] == "retrieval_protocol_error" for error in state["errors"])


def test_agent_uses_explicitly_injected_evolution_service(agent_factory, make_services):
    injected_evolution = object()
    agent = agent_factory(services=make_services(), evolution_service=injected_evolution)
    try:
        assert agent.evolution_service is injected_evolution
    finally:
        agent.close()
