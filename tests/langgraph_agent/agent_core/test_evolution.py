from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from aerospace_agent.langgraph_agent.agent_core.confirmation import (
    ConfirmationError,
    ConfirmationService,
)
from aerospace_agent.langgraph_agent.agent_core.evolution import EvolutionService
from aerospace_agent.langgraph_agent.agent_core.models import CheckpointRef, SessionMemory


NOW = datetime(2026, 7, 13, 9, 0, tzinfo=UTC)


def _memory() -> SessionMemory:
    return SessionMemory(
        memory_id="memory-1",
        project_id="project-1",
        thread_id="thread-1",
        kind="decision",
        content="Use the internal SQLite queue.",
        source_checkpoints=[
            CheckpointRef(
                project_id="project-1",
                thread_id="thread-1",
                checkpoint_id="checkpoint-1",
            )
        ],
        source_content_hash=hashlib.sha256(b"source turn").hexdigest(),
        truth_status="user_stated",
        confidence=1.0,
        created_at=NOW.isoformat(),
        updated_at=NOW.isoformat(),
    )


def _services(tmp_path):
    evolution = EvolutionService(
        tmp_path,
        database_path=tmp_path / "data/langgraph/evolution.sqlite",
        project_id="project-1",
        clock=lambda: NOW,
    )
    confirmation = ConfirmationService(
        tmp_path / "data/langgraph/confirmations.sqlite",
        clock=lambda: NOW,
    )
    return evolution, confirmation


def _issue_confirmation(evolution, confirmation, candidate):
    return confirmation.issue(
        project_id=candidate.project_id,
        thread_id=candidate.thread_id,
        root_run_id=candidate.root_run_id,
        operation_id=f"activate:{candidate.candidate_id}",
        action_hash=evolution.activation_action_hash(candidate.candidate_id),
    )


def test_session_memory_promotion_is_staged_until_single_use_human_confirmation(tmp_path):
    evolution, confirmation = _services(tmp_path)
    candidate = evolution.stage_session_promotion(_memory(), root_run_id="run-1")

    assert candidate.status == "staged"
    assert not (tmp_path / candidate.target_path).exists()
    assert evolution.list_active() == []
    with pytest.raises(ConfirmationError, match="confirmation grant not found"):
        evolution.activate(
            candidate.candidate_id,
            confirmation_service=confirmation,
            confirmation_id="missing",
        )

    grant = _issue_confirmation(evolution, confirmation, candidate)
    activated = evolution.activate(
        candidate.candidate_id,
        confirmation_service=confirmation,
        confirmation_id=grant.confirmation_id,
    )

    target = tmp_path / activated.target_path
    assert activated.status == "active"
    assert target.is_file()
    assert "Use the internal SQLite queue." in target.read_text(encoding="utf-8")
    assert evolution.list_active() == [activated]
    assert [event["event_type"] for event in evolution.events(candidate.candidate_id)] == [
        "staged",
        "approved",
    ]
    with pytest.raises(ConfirmationError, match="already used"):
        evolution.activate(
            candidate.candidate_id,
            confirmation_service=confirmation,
            confirmation_id=grant.confirmation_id,
        )


def test_project_promotion_rejects_cross_project_session_memory(tmp_path):
    evolution, _ = _services(tmp_path)
    foreign = _memory().model_copy(update={"project_id": "project-2", "source_checkpoints": [
        CheckpointRef(project_id="project-2", thread_id="thread-1", checkpoint_id="checkpoint-1")
    ]})

    with pytest.raises(ValueError, match="project namespace"):
        evolution.stage_session_promotion(foreign, root_run_id="run-1")


def test_workflow_candidate_is_not_an_active_workflow_before_approval(tmp_path):
    evolution, confirmation = _services(tmp_path)
    candidate = evolution.stage_workflow_candidate(
        thread_id="thread-1",
        root_run_id="run-2",
        workflow_id="inspect-orbit",
        version="1.0.0",
        workflow_body={"steps": [{"id": "read", "tool": "file.read"}]},
        manifest={"input_schema": {"type": "object"}, "sensitive_fields": []},
        source_checkpoint=CheckpointRef(
            project_id="project-1",
            thread_id="thread-1",
            checkpoint_id="checkpoint-2",
        ),
    )

    assert candidate.kind == "workflow"
    assert evolution.list_active() == []
    assert not (tmp_path / candidate.target_path).exists()

    grant = _issue_confirmation(evolution, confirmation, candidate)
    active = evolution.activate(
        candidate.candidate_id,
        confirmation_service=confirmation,
        confirmation_id=grant.confirmation_id,
    )
    assert active.status == "active"
    rendered = (tmp_path / active.target_path).read_text(encoding="utf-8")
    assert "inspect-orbit" in rendered
    assert "file.read" in rendered

