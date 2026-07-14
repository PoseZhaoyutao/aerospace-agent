import pytest

from aerospace_agent.langgraph_agent.schema import EvolutionProposal, EvolutionFileChange
from aerospace_agent.langgraph_agent.services.evolution import EvolutionService
from aerospace_agent.langgraph_agent.services.evolution_policy import EvolutionPolicy, parse_llm_proposal
from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService


def test_apply_create_commit_and_manual_rollback(tmp_path):
    service = EvolutionService(workspace=tmp_path)
    proposal = EvolutionProposal(thread_id="t", run_id="r", rationale="test", changes=[
        EvolutionFileChange(operation="create", path="knowledge/new.md", content="hello")
    ])
    record = service.apply(proposal)
    assert record.status == "committed"
    assert (tmp_path / "knowledge/new.md").read_text() == "hello"
    rolled = service.rollback(record.evolution_id)
    assert rolled.status == "rolled_back"
    assert not (tmp_path / "knowledge/new.md").exists()


@pytest.mark.parametrize("path", ["evolved_skills/direct.md", "workflows/evolved/direct.yaml"])
def test_legacy_apply_cannot_bypass_agent_core_trust_chain(tmp_path, path):
    service = EvolutionService(workspace=tmp_path)
    proposal = EvolutionProposal(
        thread_id="t",
        run_id="legacy-bypass",
        rationale="attempt unsigned activation",
        changes=[EvolutionFileChange(operation="create", path=path, content="unsigned")],
    )

    with pytest.raises(PermissionError, match="Agent Core"):
        service.apply(proposal)

    assert not (tmp_path / path).exists()


def test_update_delete_and_modes_are_recorded(tmp_path):
    target = tmp_path / "memory/item.md"
    target.parent.mkdir()
    target.write_text("before")
    service = EvolutionService(workspace=tmp_path, data_dir=tmp_path / "evolution")
    update = EvolutionProposal(thread_id="t", run_id="u", rationale="test", changes=[
        EvolutionFileChange(operation="update", path="memory/item.md", content="after")
    ])
    record = service.apply(update)
    assert record.status == "committed"
    assert record.manifest[0].before_sha256
    assert record.manifest[0].after_sha256
    assert record.manifest[0].mode == record.manifest[0].prior_mode
    delete = EvolutionProposal(thread_id="t", run_id="d", rationale="test", changes=[
        EvolutionFileChange(operation="delete", path="memory/item.md")
    ])
    assert service.apply(delete).status == "committed"
    assert not target.exists()


def test_failure_injection_compensates_all_operations(tmp_path):
    target = tmp_path / "knowledge/item.md"
    target.parent.mkdir()
    target.write_text("before")
    service = EvolutionService(workspace=tmp_path, data_dir=tmp_path / "evolution")
    proposal = EvolutionProposal(thread_id="t", run_id="f", rationale="test", changes=[
        EvolutionFileChange(operation="update", path="knowledge/item.md", content="after"),
        EvolutionFileChange(operation="create", path="memory/new.md", content="new"),
    ])
    record = service.apply(proposal, fail_at=2)
    assert record.status == "rolled_back"
    assert target.read_text() == "before"
    assert not (tmp_path / "memory/new.md").exists()
    assert "commit_failed" in record.state_history


def test_validation_failure_and_path_policy(tmp_path):
    service = EvolutionService(workspace=tmp_path, data_dir=tmp_path / "evolution",
                               validators=[lambda *_: False])
    proposal = EvolutionProposal(thread_id="t", run_id="v", rationale="test", changes=[
        EvolutionFileChange(operation="create", path="knowledge/new.md", content="x")
    ])
    record = service.apply(proposal)
    assert record.status == "rolled_back"
    assert "validation_failed" in record.state_history
    bad = EvolutionProposal(thread_id="t", run_id="p", rationale="test", changes=[
        EvolutionFileChange(operation="create", path="aerospace_agent/bad.py", content="x")
    ])
    try:
        service.apply(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("external source path must be rejected")


def test_due_parser_and_pending_noop(tmp_path):
    policy = EvolutionPolicy(enabled=True, idle_minutes=10, min_turns=3)
    assert not policy.is_due(idle_minutes=1, turn_count=3).due
    assert policy.is_due(idle_minutes=10, turn_count=3).due
    assert parse_llm_proposal('{"thread_id":"t","run_id":"r","rationale":"x","changes":[],"extra":1}') is None
    service = EvolutionService(workspace=tmp_path, data_dir=tmp_path / "evolution")
    proposal = {"thread_id": "t", "run_id": "pending", "rationale": "x", "changes": [],
                "unfinished_items": ["external action"]}
    assert service.evolve_due(proposal, idle_minutes=10, turn_count=3) is None
    assert "external action" in (tmp_path / "memory/pending.md").read_text()
    assert service.evolve_due(proposal, idle_minutes=10, turn_count=3) is None


def test_successful_wiki_change_is_indexed_by_knowledge_service(tmp_path):
    knowledge = KnowledgeService(workspace=tmp_path, data_dir=tmp_path / "rag")
    knowledge.initialize_seed_wiki()
    service = EvolutionService(workspace=tmp_path, data_dir=tmp_path / "evolution",
                               knowledge_service=knowledge)
    proposal = EvolutionProposal(thread_id="t", run_id="wiki", rationale="add wiki fact", changes=[
        EvolutionFileChange(operation="create", path="knowledge/new.md",
                             content="# New\\n\\nQuasar telemetry keyword-lambda-42.")
    ])
    record = service.apply(proposal)
    assert record.status == "committed"
    assert knowledge.search("keyword-lambda-42")
    assert record.validation_details["rebuild"]["rag"] == "ok"
    assert record.validation_details["rebuild"]["graph"] == "ok"


def test_rebuild_failure_restores_wiki_and_derived_index(tmp_path):
    knowledge = KnowledgeService(workspace=tmp_path, data_dir=tmp_path / "rag")
    knowledge.initialize_seed_wiki()
    service = EvolutionService(workspace=tmp_path, data_dir=tmp_path / "evolution",
                               knowledge_service=knowledge,
                               failure_injection="rag_rebuild")
    proposal = EvolutionProposal(thread_id="t", run_id="failed-wiki", rationale="add wiki fact", changes=[
        EvolutionFileChange(operation="create", path="knowledge/new.md",
                             content="# New\\n\\nQuasar telemetry keyword-mu-99.")
    ])
    record = service.apply(proposal)
    assert record.status == "rolled_back"
    assert not (tmp_path / "knowledge/new.md").exists()
    assert not any(item.page_path == "knowledge/new.md" for item in knowledge.search("keyword-mu-99"))
    assert record.validation_details["rebuild"]["rag"] == "failed"
    assert record.validation_details["rebuild"]["errors"]


def test_empty_evolve_without_proposal_is_noop(tmp_path):
    from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent

    agent = LangGraphAerospaceAgent(checkpoint_backend="memory", checkpoint_db_path=":memory:")
    try:
        result = agent.evolve("missing-thread")
    finally:
        agent.close()
    assert result["status"] == "no_op"


def test_agent_rejects_spoofed_boolean_evolution_approval(tmp_path):
    from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent
    from aerospace_agent.langgraph_agent.graph import ServiceBundle
    from aerospace_agent.langgraph_agent.safety import ApprovalRequired, SafetyValidator

    service = EvolutionService(workspace=tmp_path, data_dir=tmp_path / "evolution")
    agent = LangGraphAerospaceAgent(
        checkpoint_backend="memory",
        checkpoint_db_path=":memory:",
        services=ServiceBundle(evolution=service, safety=SafetyValidator()),
    )
    proposal = EvolutionProposal(
        thread_id="thread-1",
        run_id="run-1",
        rationale="spoof a boolean approval",
        changes=[
            EvolutionFileChange(
                operation="create",
                path="knowledge/spoofed.md",
                content="unapproved",
            )
        ],
    )
    try:
        with pytest.raises(ApprovalRequired, match="boolean approval flags"):
            agent.evolve(proposal, human_approved=True)
    finally:
        agent.close()

    assert not (tmp_path / "knowledge" / "spoofed.md").exists()
