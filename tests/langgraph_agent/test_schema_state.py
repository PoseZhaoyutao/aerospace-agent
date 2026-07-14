from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from aerospace_agent.langgraph_agent.schema import (
    ActionType,
    AgentInput,
    AgentOutput,
    Decision,
    EvidenceItem,
    EvolutionFileChange,
    EvolutionProposal,
    EvolutionRecord,
    RunStatus,
    ToolCallRequest,
    ToolCallResponse,
    export_json_schemas,
)
from aerospace_agent.langgraph_agent.state import create_initial_state


def test_agent_input_has_versioned_runtime_contract():
    value = AgentInput(user_message="Explain two-body dynamics", thread_id="t1")
    assert value.schema_version == "1.0.0"
    assert value.recursion_limit > value.max_steps
    assert value.run_id


def test_agent_output_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        AgentOutput(status="looks-good", answer="x")


def test_intermediate_protocol_constraints_and_independent_defaults():
    evidence = EvidenceItem(
        source_id="seed:two-body",
        page_path="knowledge/a.md",
        chunk_id="a:0",
        score=0.8,
        excerpt="central gravity",
    )
    decision = Decision(action=ActionType.RESPOND, rationale="evidence is enough")
    output_a = AgentOutput(status=RunStatus.SUCCESS, answer="x", citations=[evidence])
    output_b = AgentOutput(status=RunStatus.SUCCESS, answer="y")
    output_a.warnings.append("only-a")
    assert output_b.warnings == []
    assert decision.action == "respond"
    with pytest.raises(ValueError, match="score"):
        EvidenceItem(source_id="s", page_path="knowledge/a.md", chunk_id="c", score=2)
    with pytest.raises(ValueError, match="relative"):
        EvolutionFileChange(operation="update", path=Path("../outside"), content="x")


def test_tool_and_evolution_models_cover_success_and_failure():
    request = ToolCallRequest(tool_name="check_engine_availability")
    response = ToolCallResponse(tool_name=request.tool_name, status="success", result={})
    proposal = EvolutionProposal(thread_id="t", run_id="r", rationale="reuse", changes=[])
    record = EvolutionRecord(evolution_id="e", thread_id="t", run_id="r", status="proposed")
    assert response.status == "success"
    assert proposal.changes == []
    assert record.status == "proposed"


def test_state_round_trips_through_langgraph_jsonplus():
    state = create_initial_state("t1", "r1", max_steps=5)
    state["messages"] = [HumanMessage(content="hello")]
    assert "knowledge_service" not in state
    assert "db_connection" not in state
    serializer = JsonPlusSerializer()
    type_name, payload = serializer.dumps_typed(state)
    restored = serializer.loads_typed((type_name, payload))
    assert restored["thread_id"] == "t1"
    assert restored["messages"][0].content == "hello"


def test_initial_state_has_checkpoint_safe_retrieval_defaults():
    state = create_initial_state("t1", "r1")

    assert state["retrieval_required"] is False
    assert state["retrieval_reason"] == ""
    assert state["retrieval_attempted"] is False
    assert state["retrieval_query_hash"] == ""


def test_json_schema_export_contains_every_public_protocol():
    schemas = export_json_schemas()
    required = {
        "AgentInput",
        "AgentOutput",
        "Decision",
        "EvidenceItem",
        "ToolCallRequest",
        "ToolCallResponse",
        "EvolutionProposal",
        "EvolutionRecord",
        "OrbitState",
        "KeplerianOrbitState",
        "OrbitDesignRequest",
        "OrbitDesignResponse",
        "RagQueryRequest",
        "RagQueryResponse",
    }
    assert required <= schemas.keys()
    assert all(schema.get("type") == "object" for schema in schemas.values())


def test_schema_versions_and_evolution_status_are_closed_sets():
    with pytest.raises(ValueError, match="schema_version"):
        AgentInput(user_message="x", schema_version="2.0.0")
    with pytest.raises(ValueError, match="schema_version"):
        AgentOutput(status=RunStatus.SUCCESS, schema_version="1.0.1")
    output_schema = AgentOutput.model_json_schema()
    assert output_schema["$defs"]["RunStatus"]["enum"] == [
        "success", "partial", "error", "interrupted", "cycle_detected", "limit_reached",
    ]
    assert EvolutionRecord.model_json_schema()["properties"]["status"]["default"] == "proposed"
    with pytest.raises(ValueError):
        EvolutionRecord(evolution_id="e", thread_id="t", run_id="r", status="unknown")


@pytest.mark.parametrize("bad_path", [
    "/absolute/path.md", "\\absolute\\path.md", "C:/drive-relative.md",
    "C:drive-relative.md", "\\\\server\\share\\x.md", "a/../x.md",
])
def test_protocol_paths_reject_cross_platform_escape_forms(bad_path):
    with pytest.raises(ValueError, match="relative"):
        EvidenceItem(source_id="s", page_path=bad_path, chunk_id="c", score=0.5)
    with pytest.raises(ValueError, match="relative"):
        EvolutionFileChange(operation="update", path=bad_path, content="x")


def test_delete_change_cannot_carry_content():
    with pytest.raises(ValueError, match="content"):
        EvolutionFileChange(operation="delete", path="knowledge/a.md", content="not allowed")
