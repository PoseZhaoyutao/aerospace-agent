from __future__ import annotations

from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.agent_core.models import CheckpointRef
from aerospace_agent.langgraph_agent.agent_core.project_memory import ProjectIdentityService
from aerospace_agent.langgraph_agent.agent_core.session_memory import (
    SessionMemoryService,
    SessionSummary,
)


def _checkpoint(project_id: str, thread_id: str, checkpoint_id: str = "cp-1") -> CheckpointRef:
    return CheckpointRef(
        project_id=project_id,
        thread_id=thread_id,
        checkpoint_id=checkpoint_id,
    )


def _service(
    tmp_path: Path,
    *,
    project_id: str,
    thread_id: str,
    valid_checkpoints: set[str] | None = None,
) -> SessionMemoryService:
    return SessionMemoryService(
        tmp_path / "data" / "langgraph" / "session_memory.sqlite",
        project_id=project_id,
        thread_id=thread_id,
        checkpoint_validator=lambda checkpoint, _content_hash: (
            checkpoint.checkpoint_id in (valid_checkpoints or {"cp-1", "cp-2"})
        ),
    )


def test_same_thread_memory_survives_restart_and_other_thread_cannot_read(tmp_path) -> None:
    project_id = ProjectIdentityService(tmp_path).initialize().project_id
    assert project_id is not None
    service = _service(tmp_path, project_id=project_id, thread_id="thread-a")
    saved = service.remember(
        kind="fact",
        content="The spacecraft dry mass is 120 kg",
        source_checkpoints=[_checkpoint(project_id, "thread-a")],
        source_content_hash="a" * 64,
        truth_status="user_stated",
        confidence=1.0,
    )

    restarted = _service(tmp_path, project_id=project_id, thread_id="thread-a")
    isolated = _service(tmp_path, project_id=project_id, thread_id="thread-b")

    assert [item.memory_id for item in restarted.search("spacecraft")] == [saved.memory_id]
    assert isolated.search("spacecraft") == []
    assert isolated.list() == []


def test_natural_language_memory_query_ignores_instruction_stopwords(tmp_path) -> None:
    project_id = ProjectIdentityService(tmp_path).initialize().project_id
    assert project_id is not None
    service = _service(tmp_path, project_id=project_id, thread_id="thread-natural")
    saved = service.remember(
        kind="fact",
        content="The spacecraft dry mass is 120 kg",
        source_checkpoints=[_checkpoint(project_id, "thread-natural")],
        source_content_hash="a" * 64,
        truth_status="user_stated",
        confidence=1.0,
    )

    matches = service.search("What is the spacecraft dry mass?")

    assert [item.memory_id for item in matches] == [saved.memory_id]


def test_chinese_session_memory_search_uses_substring_fallback(tmp_path) -> None:
    project_id = ProjectIdentityService(tmp_path).initialize().project_id
    assert project_id is not None
    service = _service(tmp_path, project_id=project_id, thread_id="thread-cn")
    saved = service.remember(
        kind="fact",
        content="航天器干质量为120千克",
        source_checkpoints=[_checkpoint(project_id, "thread-cn")],
        source_content_hash="a" * 64,
        truth_status="user_stated",
        confidence=1.0,
    )

    matches = service.search("干质量")

    assert [item.memory_id for item in matches] == [saved.memory_id]


def test_project_and_thread_are_constructor_bound_not_method_parameters(tmp_path) -> None:
    project_id = ProjectIdentityService(tmp_path).initialize().project_id
    assert project_id is not None
    service = _service(tmp_path, project_id=project_id, thread_id="thread-a")

    with pytest.raises(TypeError):
        service.search("fact", thread_id="thread-b")  # type: ignore[call-arg]


def test_checkpoint_namespace_and_validator_are_enforced(tmp_path) -> None:
    project_id = ProjectIdentityService(tmp_path).initialize().project_id
    assert project_id is not None
    service = _service(
        tmp_path,
        project_id=project_id,
        thread_id="thread-a",
        valid_checkpoints={"cp-valid"},
    )

    with pytest.raises(ValueError, match="namespace"):
        service.remember(
            kind="fact",
            content="wrong namespace",
            source_checkpoints=[_checkpoint(project_id, "thread-b", "cp-valid")],
            source_content_hash="a" * 64,
            truth_status="user_stated",
            confidence=1.0,
        )
    with pytest.raises(ValueError, match="checkpoint validation"):
        service.remember(
            kind="fact",
            content="uncommitted turn",
            source_checkpoints=[_checkpoint(project_id, "thread-a", "cp-missing")],
            source_content_hash="a" * 64,
            truth_status="user_stated",
            confidence=1.0,
        )


def test_user_correction_supersedes_old_memory_without_deleting_history(tmp_path) -> None:
    project_id = ProjectIdentityService(tmp_path).initialize().project_id
    assert project_id is not None
    service = _service(tmp_path, project_id=project_id, thread_id="thread-a")
    old = service.remember(
        kind="constraint",
        content="Maximum thrust is 10 N",
        source_checkpoints=[_checkpoint(project_id, "thread-a")],
        source_content_hash="a" * 64,
        truth_status="user_stated",
        confidence=1.0,
    )

    corrected = service.update(
        old.memory_id,
        content="Maximum thrust is 12 N",
        source_checkpoints=[_checkpoint(project_id, "thread-a", "cp-2")],
        source_content_hash="b" * 64,
        truth_status="user_stated",
        confidence=1.0,
    )

    assert corrected.supersedes == old.memory_id
    assert [item.content for item in service.list()] == ["Maximum thrust is 12 N"]
    audit = service.list(include_history=True)
    assert {item.truth_status for item in audit} == {"superseded", "user_stated"}


def test_forget_and_clear_require_explicit_bulk_confirmation(tmp_path) -> None:
    project_id = ProjectIdentityService(tmp_path).initialize().project_id
    assert project_id is not None
    service = _service(tmp_path, project_id=project_id, thread_id="thread-a")
    first = service.remember(
        kind="preference",
        content="Use SI units",
        source_checkpoints=[_checkpoint(project_id, "thread-a")],
        source_content_hash="a" * 64,
        truth_status="user_stated",
        confidence=1.0,
    )
    service.forget(first.memory_id)
    assert service.list() == []
    assert service.list(include_history=True)[0].truth_status == "retracted"

    service.remember(
        kind="decision",
        content="Use RK4 for the smoke test",
        source_checkpoints=[_checkpoint(project_id, "thread-a", "cp-2")],
        source_content_hash="b" * 64,
        truth_status="user_stated",
        confidence=1.0,
    )
    with pytest.raises(PermissionError, match="confirmation"):
        service.clear(confirmation_consumed=False)
    assert service.clear(confirmation_consumed=True) == 1
    assert service.list() == []


def test_search_excludes_superseded_and_retracted_unless_history_requested(tmp_path) -> None:
    project_id = ProjectIdentityService(tmp_path).initialize().project_id
    assert project_id is not None
    service = _service(tmp_path, project_id=project_id, thread_id="thread-a")
    old = service.remember(
        kind="fact",
        content="alpha old value",
        source_checkpoints=[_checkpoint(project_id, "thread-a")],
        source_content_hash="a" * 64,
        truth_status="assumption",
        confidence=0.5,
    )
    service.update(
        old.memory_id,
        content="alpha corrected value",
        source_checkpoints=[_checkpoint(project_id, "thread-a", "cp-2")],
        source_content_hash="b" * 64,
        truth_status="verified",
        confidence=0.9,
    )

    assert [item.content for item in service.search("alpha")] == ["alpha corrected value"]
    assert len(service.search("alpha", include_history=True)) == 2


def test_summary_revision_and_due_policy_are_namespace_bound(tmp_path) -> None:
    project_id = ProjectIdentityService(tmp_path).initialize().project_id
    assert project_id is not None
    service = _service(tmp_path, project_id=project_id, thread_id="thread-a")
    summary = service.save_summary(
        current_goal="Validate orbit propagation",
        confirmed_constraints=["Use SI units"],
        decisions=["Use two independent propagators"],
        completed_items=[],
        open_items=["Run cross validation"],
        artifacts=[],
        assumptions=["Atmosphere omitted in smoke test"],
        source_checkpoints=[_checkpoint(project_id, "thread-a")],
    )

    assert isinstance(summary, SessionSummary)
    assert summary.revision == 1
    assert service.latest_summary() == summary
    assert service.summary_due(turn_count=6, context_ratio=0.1) is True
    assert service.summary_due(turn_count=1, context_ratio=0.7) is True
    assert service.summary_due(turn_count=1, context_ratio=0.1, user_corrected=True) is True
    assert service.summary_due(turn_count=1, context_ratio=0.1) is False

