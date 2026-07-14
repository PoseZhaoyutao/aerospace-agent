"""Acceptance checks for the local Qwen endpoint.

These tests are intentionally integration-only.  They never spawn a model
process; an unavailable endpoint is reported as a blocked/skip outcome.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent, SimpleLLMClient
from aerospace_agent.langgraph_agent.config import load_settings
from aerospace_agent.langgraph_agent.graph import ServiceBundle
from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService
from aerospace_agent.langgraph_agent.services.planner import LLMPlanner
from scripts.run_langgraph_acceptance import verify_answer_against_citations


@dataclass(frozen=True)
class QwenEndpoint:
    endpoint: str
    model: str
    available: bool
    models: tuple[str, ...] = ()
    error: str | None = None


def _probe_qwen() -> QwenEndpoint:
    endpoint = os.environ.get("AEROSPACE_LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    model = os.environ.get("AEROSPACE_LOCAL_LLM_MODEL", "qwythos")
    try:
        request = urllib.request.Request(f"{endpoint}/models", method="GET")
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = tuple(str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict))
        return QwenEndpoint(endpoint, model, model in models, models=models)
    except Exception as exc:  # no process management or auto-start here
        return QwenEndpoint(endpoint, model, False, error=f"{type(exc).__name__}: {exc}")


@pytest.fixture(scope="session")
def qwen_endpoint() -> QwenEndpoint:
    info = _probe_qwen()
    if not info.available:
        pytest.skip(f"Qwen endpoint blocked/unavailable: {info.error or info.models}")
    return info


@pytest.mark.qwen3
@pytest.mark.integration
def test_direct_qwen_claim_support_is_grounded(qwen_endpoint: QwenEndpoint):
    client = SimpleLLMClient(endpoint=qwen_endpoint.endpoint, model=qwen_endpoint.model, timeout=30)
    answer = client.chat(
        "Output exactly this sentence and nothing else: The governing acceleration is -mu r / |r|^3.",
        system_prompt="Copy the requested sentence exactly. Do not identify yourself or add commentary.",
        max_tokens=64,
        temperature=0.0,
        chat_template_kwargs={"enable_thinking": False},
    )
    citations = [{
        "page_path": "knowledge/orbital-dynamics/two-body-orbital-dynamics.md",
        "excerpt": "The governing acceleration is -mu r / |r|^3.",
    }]
    support = verify_answer_against_citations(answer, citations)
    assert support["unsupported_claims"] == [], {"answer": answer, "support": support}


@pytest.mark.qwen3
@pytest.mark.integration
def test_qwen_agent_two_turn_thread_is_checkpointed(qwen_endpoint: QwenEndpoint, tmp_path: Path):
    knowledge = KnowledgeService(workspace=tmp_path)
    knowledge.initialize_seed_wiki()
    settings = load_settings(workspace=tmp_path)
    # Keep the graph deterministic while using the local Qwen model metadata.
    agent = LangGraphAerospaceAgent(
        settings=settings,
        llm_endpoint=qwen_endpoint.endpoint,
        model_name=qwen_endpoint.model,
        checkpoint_backend="memory",
        services=None,
        rag=knowledge,
    )
    try:
        first = agent.run(
            "Using the private knowledge base, cite evidence for the governing acceleration in two-body dynamics.",
            thread_id="qwen-acceptance",
        )
        assert first.status == "success"
        assert any("two-body-orbital-dynamics.md" in item.page_path for item in first.citations)
        first_support = verify_answer_against_citations(first.answer, [item.model_dump() for item in first.citations])
        assert first_support["unsupported_claims"] == []

        second = agent.run("Restate the exact claim and its assumptions.", thread_id="qwen-acceptance")
        assert second.status == "success"
        assert second.checkpoint_id and second.checkpoint_id != first.checkpoint_id
        snapshot = agent.get_conversation_state("qwen-acceptance")
        assert len(snapshot.values["messages"]) >= 4
    finally:
        agent.close()


class CountingKnowledgeService:
    def __init__(self, service: KnowledgeService):
        self.service = service
        self.calls: list[str] = []

    def search(self, query: str, *, top_k: int = 5):
        self.calls.append(query)
        return self.service.search(query, top_k=top_k)


class RecordingPlanner:
    def __init__(self, planner):
        self.planner = planner
        self.states: list[dict] = []

    def plan(self, state):
        self.states.append(dict(state))
        return self.planner.plan(state)


@pytest.mark.qwen3
@pytest.mark.integration
def test_qwen_general_conversation_bypasses_private_rag(
    qwen_endpoint: QwenEndpoint,
    tmp_path: Path,
):
    knowledge = KnowledgeService(workspace=tmp_path)
    knowledge.initialize_seed_wiki()
    counting = CountingKnowledgeService(knowledge)
    settings = load_settings(workspace=tmp_path)
    agent = LangGraphAerospaceAgent(
        settings=settings,
        llm_endpoint=qwen_endpoint.endpoint,
        model_name=qwen_endpoint.model,
        checkpoint_backend="memory",
        rag=counting,
    )
    try:
        result = agent.run("你好，我们先正常讨论今天的研究安排。", thread_id="qwen-chat-no-rag")
        assert result.status == "success"
        assert result.answer.strip()
        assert result.citations == []
        assert counting.calls == []
        assert result.metrics["retrieval_required"] is False
    finally:
        agent.close()


@pytest.mark.qwen3
@pytest.mark.integration
def test_qwen_explicit_evidence_request_uses_private_rag_once(
    qwen_endpoint: QwenEndpoint,
    tmp_path: Path,
):
    knowledge = KnowledgeService(workspace=tmp_path)
    knowledge.initialize_seed_wiki()
    counting = CountingKnowledgeService(knowledge)
    settings = load_settings(workspace=tmp_path)
    agent = LangGraphAerospaceAgent(
        settings=settings,
        llm_endpoint=qwen_endpoint.endpoint,
        model_name=qwen_endpoint.model,
        checkpoint_backend="memory",
        rag=counting,
    )
    try:
        result = agent.run(
            "请根据私域知识库给出二体动力学加速度的依据和来源。",
            thread_id="qwen-explicit-rag",
        )
        assert result.status == "success"
        assert len(counting.calls) == 1
        assert result.citations
        assert all(item.page_path for item in result.citations)
        assert result.metrics["retrieval_reason"] == "explicit_evidence"
    finally:
        agent.close()


@pytest.mark.qwen3
@pytest.mark.integration
def test_qwen_work_intent_is_dispatched_to_model_planner(
    qwen_endpoint: QwenEndpoint,
    tmp_path: Path,
):
    knowledge = KnowledgeService(workspace=tmp_path)
    knowledge.initialize_seed_wiki()
    counting = CountingKnowledgeService(knowledge)
    llm = SimpleLLMClient(
        endpoint=qwen_endpoint.endpoint,
        model=qwen_endpoint.model,
        timeout=30,
    )
    planner = RecordingPlanner(LLMPlanner(llm, tool_names=[]))
    services = ServiceBundle(
        knowledge=counting,
        planner=planner,
        llm=llm,
        model_name=qwen_endpoint.model,
        endpoint=qwen_endpoint.endpoint,
    )
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=services,
        checkpoint_backend="memory",
    )
    try:
        result = agent.run(
            "Draft a high-level orbit propagation work plan.",
            thread_id="qwen-model-planner",
        )

        assert result.status == "success"
        assert result.answer.strip()
        assert planner.states
        assert planner.states[0]["intent"] == "orbit_propagation"
        assert planner.states[0]["retrieval_attempted"] is False
        assert planner.states[0]["evidence"] == []
        if counting.calls:
            assert result.metrics["retrieval_reason"] == "planner_request"
    finally:
        agent.close()
