from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

from aerospace_agent.langgraph_agent.cycle_detector import (
    CycleDetector,
    evaluate_cycle,
    fingerprint,
)
from aerospace_agent.langgraph_agent.services.context import ContextService
from aerospace_agent.langgraph_agent.graph import ServiceBundle
from aerospace_agent.langgraph_agent.nodes import context_node


class _MemorySections:
    def prompt_sections(self) -> list[str]:
        return ["[PROJECT MEMORY]\nverified constraint", "[SESSION MEMORY]\nsame-thread fact"]


class _MemoryAssembler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def assemble(self, *, thread_id: str, query: str) -> _MemorySections:
        self.calls.append((thread_id, query))
        return _MemorySections()


def test_context_assembly_keeps_constraints_and_current_and_offloads_tool_result(tmp_path: Path):
    service = ContextService(
        tmp_path,
        max_tokens=100,
        recent_turns=2,
        artifact_chars=50,
    )
    result = service.assemble(
        messages=[
            SystemMessage(content="SYSTEM CONSTRAINT: use SI units"),
            HumanMessage(content="old " * 100),
            HumanMessage(content="CURRENT"),
        ],
        tool_results=[{"payload": "x" * 500}],
    )

    rendered = "\n".join(str(m.content) for m in result.messages)
    assert "SYSTEM CONSTRAINT" in rendered
    assert "CURRENT" in rendered
    assert result.estimated_tokens <= 100
    assert len(result.messages) <= 4
    assert len(result.artifact_refs) == 1

    ref = result.artifact_refs[0]
    assert Path(ref.path).is_relative_to(tmp_path)
    assert len(ref.sha256) == 64
    assert ref.media_type == "application/json"
    assert ref.summary
    payload = json.loads(Path(ref.path).read_text(encoding="utf-8"))
    assert payload == {"payload": "x" * 500}
    assert "x" * 500 not in rendered


def test_context_assembly_injects_only_runtime_bound_memory_for_current_thread(tmp_path: Path):
    memory = _MemoryAssembler()
    service = ContextService(tmp_path, max_tokens=200, memory_context=memory)

    result = service.assemble(
        messages=[SystemMessage(content="system"), HumanMessage(content="current request")],
        thread_id="thread-a",
        current_request="current request",
    )

    rendered = "\n".join(str(message.content) for message in result.messages)
    assert memory.calls == [("thread-a", "current request")]
    assert "[PROJECT MEMORY]" in rendered
    assert "same-thread fact" in rendered


def test_context_assembly_does_not_read_memory_without_thread_binding(tmp_path: Path):
    memory = _MemoryAssembler()
    service = ContextService(tmp_path, memory_context=memory)

    service.assemble(messages=[HumanMessage(content="request")])

    assert memory.calls == []


def test_context_node_replaces_additive_message_channel_after_compaction(tmp_path: Path):
    service = ContextService(tmp_path, max_tokens=100, recent_turns=1)
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="old request"),
        AIMessage(content="old response"),
        HumanMessage(content="current request"),
    ]

    delta = context_node(
        {"messages": messages, "thread_id": "thread-a", "tool_results": []},
        ServiceBundle(context=service),
    )
    merged = add_messages(messages, delta["messages"])
    rendered = [str(message.content) for message in merged]

    assert "old request" not in rendered
    assert rendered == [
        "system",
        "[CONTEXT SUMMARY] omitted 1 earlier message(s): human",
        "old response",
        "current request",
    ]


def test_context_service_does_not_reinject_prior_ephemeral_memory(tmp_path: Path):
    memory = _MemoryAssembler()
    service = ContextService(tmp_path, memory_context=memory)

    first = service.assemble(
        messages=[HumanMessage(content="first")],
        thread_id="thread-a",
        current_request="first",
    )
    second = service.assemble(
        messages=[*first.messages, AIMessage(content="answer"), HumanMessage(content="second")],
        thread_id="thread-a",
        current_request="second",
    )
    rendered = "\n".join(str(message.content) for message in second.messages)

    assert rendered.count("[PROJECT MEMORY]") == 1


def test_evaluate_cycle_state_isolation():
    a: dict = {}
    b: dict = {}
    for _ in range(3):
        delta = evaluate_cycle(a, action="retrieve", payload={"q": "same"}, max_repeats=3)
        a.update(delta)
    assert a.get("intervention_count") == 1
    assert b.get("intervention_count", 0) == 0


def test_fingerprint_distinguishes_tool_name_and_target():
    first = fingerprint(
        action="call_tool",
        tool_name="orbit.propagate",
        target="sat-a",
        params={"duration_s": 60},
    )
    second = fingerprint(
        action="call_tool",
        tool_name="orbit.state",
        target="sat-a",
        params={"duration_s": 60},
    )
    third = fingerprint(
        action="call_tool",
        tool_name="orbit.propagate",
        target="sat-b",
        params={"duration_s": 60},
    )
    assert first != second
    assert first != third


def test_compatibility_check_returns_reason_and_delta():
    detector = CycleDetector(max_repeats=2)
    state = {"messages": [HumanMessage(content="same")], "step_count": 0}
    first = detector.check(state)
    assert len(first) == 3
    assert first[0] is False
    assert isinstance(first[1], str)
    assert isinstance(first[2], dict)
