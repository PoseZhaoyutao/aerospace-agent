from __future__ import annotations

from dataclasses import replace
from pathlib import Path


def test_same_thread_survives_agent_recreation(agent_factory, services, tmp_path: Path):
    db = tmp_path / "checkpoints.sqlite"
    first = agent_factory(services=services, checkpoint_backend="sqlite", checkpoint_db_path=db)
    result = first.run("What is two-body dynamics?", thread_id="persist")
    first.close()

    second = agent_factory(services=services, checkpoint_backend="sqlite", checkpoint_db_path=db)
    snapshot = second.get_conversation_state("persist")
    assert len(snapshot.values["messages"]) >= 2
    follow_up = second.run("Summarize that.", thread_id="persist")
    assert follow_up.checkpoint_id
    second.close()


def test_interrupt_and_resume(agent_factory, services):
    agent = agent_factory(services=services, checkpoint_backend="memory", interrupt_before=["synthesize"])
    interrupted = agent.run("What is two-body dynamics?", thread_id="resume")
    assert interrupted.status == "interrupted"
    assert agent.get_conversation_state("resume").next == ("synthesize",)
    resumed = agent.resume_execution("resume")
    assert resumed.status == "success"
    agent.close()


def test_history_replay_and_fork(agent_factory, services):
    agent = agent_factory(services=services, checkpoint_backend="memory")
    agent.run("What is two-body dynamics?", thread_id="source")
    agent.run("Explain it briefly.", thread_id="source")
    history = agent.get_checkpoint_history("source")
    assert history and history[0]["created_at"] >= history[-1]["created_at"]
    selected = history[-1]["checkpoint_id"]
    agent.fork_from_checkpoint(selected, new_thread_id="forked", source_thread_id="source")
    assert "source" in agent.list_conversations()
    assert "forked" in agent.list_conversations()
    assert agent.get_state("source").values["thread_id"] == "source"
    assert agent.get_state_history("source")
    agent.close()


def test_graph_recursion_limit_is_structured(agent_factory, services_with_repeating_planner):
    agent = agent_factory(services=services_with_repeating_planner, checkpoint_backend="memory", max_recursion_depth=2)
    result = agent.run("Check engine availability", thread_id="limit")
    assert result.status == "limit_reached"
    assert result.checkpoint_id
    assert result.metrics["model_name"] == "deterministic-test"
    assert result.metrics["error_category"] == "graph_recursion_limit"
    agent.close()


def test_checkpoint_write_failure_is_not_success(agent_factory, services, failing_checkpointer):
    agent = agent_factory(services=services, checkpointer=failing_checkpointer)
    result = agent.run("What is two-body dynamics?", thread_id="failed")
    assert result.status == "error"
    assert result.checkpoint_id is None
    assert result.metrics["error_category"] == "checkpoint_write_error"
    assert all("success" not in str(error).lower() for error in result.errors)
    agent.close()


def test_new_turn_resets_retrieval_state(agent_factory, services):
    agent = agent_factory(services=services, checkpoint_backend="memory")
    try:
        agent.run("What is two-body dynamics?", thread_id="retrieval-reset")
        agent.graph.update_state(
            agent._config_for("retrieval-reset"),
            {
                "retrieval_required": True,
                "retrieval_reason": "explicit_evidence",
                "retrieval_attempted": True,
                "retrieval_query_hash": "old-query",
                "evidence": [{"source_id": "old"}],
                "citations": [{"source_id": "old"}],
            },
        )

        next_input = agent._input_for_run("hello", "retrieval-reset", None)

        assert next_input["retrieval_required"] is False
        assert next_input["retrieval_reason"] == ""
        assert next_input["retrieval_attempted"] is False
        assert next_input["retrieval_query_hash"] == ""
        assert next_input["evidence"] == []
        assert next_input["citations"] == []
        assert next_input["status"] == ""
        assert next_input["termination_reason"] == ""
        assert next_input["errors"] == []
        assert next_input["warnings"] == []
        assert next_input["metrics"] == {}
    finally:
        agent.close()


class _EmptyKnowledge:
    def search(self, query, *, top_k=5):
        return []


def test_partial_retrieval_failure_does_not_pollute_next_turn(agent_factory, services):
    agent = agent_factory(
        services=replace(services, knowledge=_EmptyKnowledge()),
        checkpoint_backend="memory",
    )
    try:
        first = agent.run("Please cite evidence for this claim.", thread_id="turn-reset-partial")
        assert first.status == "partial"
        assert first.warnings

        second = agent.run("hello", thread_id="turn-reset-partial")

        assert second.status == "success"
        assert second.warnings == []
        assert second.errors == []
        assert second.citations == []
        assert second.metrics["rag_hits"] == 0
    finally:
        agent.close()


def test_successful_rag_metrics_do_not_pollute_next_turn(agent_factory, services):
    agent = agent_factory(services=services, checkpoint_backend="memory")
    try:
        first = agent.run(
            "Please cite evidence for two-body dynamics.",
            thread_id="turn-reset-metrics",
        )
        assert first.metrics["rag_hits"] > 0

        second = agent.run("hello", thread_id="turn-reset-metrics")

        assert second.status == "success"
        assert second.citations == []
        assert second.metrics["rag_hits"] == 0
    finally:
        agent.close()
