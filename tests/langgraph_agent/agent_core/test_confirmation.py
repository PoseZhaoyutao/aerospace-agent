from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aerospace_agent.langgraph_agent.agent_core.confirmation import (
    ConfirmationError,
    ConfirmationService,
    compute_action_hash,
)


def _hash(content: str = "payload") -> str:
    return compute_action_hash(
        tool_name="file.delete",
        arguments={"path": content},
        target_paths=[content],
        run_id="root-run",
        risk_level="high_risk",
    )


def test_action_hash_is_canonical_and_changes_with_protected_input() -> None:
    first = compute_action_hash(
        tool_name="file.write",
        arguments={"b": 2, "a": 1},
        target_paths=["b", "a"],
        run_id="run-1",
        risk_level="project_write",
    )
    reordered = compute_action_hash(
        tool_name="file.write",
        arguments={"a": 1, "b": 2},
        target_paths=["a", "b"],
        run_id="run-1",
        risk_level="project_write",
    )
    changed = compute_action_hash(
        tool_name="file.write",
        arguments={"a": 1, "b": 3},
        target_paths=["a", "b"],
        run_id="run-1",
        risk_level="project_write",
    )

    assert first == reordered
    assert first != changed
    assert len(first) == 64


def test_confirmation_is_single_use_and_bound_to_namespace_and_action(tmp_path) -> None:
    now = datetime(2026, 7, 13, 8, 0, tzinfo=UTC)
    service = ConfirmationService(tmp_path / "confirmation.sqlite", clock=lambda: now)
    grant = service.issue(
        project_id="project",
        thread_id="thread-a",
        root_run_id="root-run",
        operation_id="operation",
        action_hash=_hash(),
    )

    consumed = service.consume(
        confirmation_id=grant.confirmation_id,
        project_id="project",
        thread_id="thread-a",
        root_run_id="root-run",
        operation_id="operation",
        action_hash=_hash(),
        continuation_checkpoint={"checkpoint_id": "confirmation:operation"},
    )

    assert consumed.used_at is not None
    with pytest.raises(ConfirmationError) as replay:
        service.consume(
            confirmation_id=grant.confirmation_id,
            project_id="project",
            thread_id="thread-a",
            root_run_id="root-run",
            operation_id="operation",
            action_hash=_hash(),
            continuation_checkpoint={"checkpoint_id": "confirmation:operation"},
        )
    assert replay.value.code == "confirmation_replayed"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"thread_id": "thread-b"}, "confirmation_required"),
        ({"operation_id": "other"}, "confirmation_required"),
        ({"action_hash": _hash("changed")}, "confirmation_required"),
    ],
)
def test_confirmation_rejects_context_or_argument_changes(tmp_path, overrides, code) -> None:
    now = datetime(2026, 7, 13, 8, 0, tzinfo=UTC)
    service = ConfirmationService(tmp_path / "confirmation.sqlite", clock=lambda: now)
    grant = service.issue(
        project_id="project",
        thread_id="thread-a",
        root_run_id="root-run",
        operation_id="operation",
        action_hash=_hash(),
    )
    request = {
        "confirmation_id": grant.confirmation_id,
        "project_id": "project",
        "thread_id": "thread-a",
        "root_run_id": "root-run",
        "operation_id": "operation",
        "action_hash": _hash(),
        "continuation_checkpoint": {"checkpoint_id": "confirmation:operation"},
    }
    request.update(overrides)

    with pytest.raises(ConfirmationError) as error:
        service.consume(**request)

    assert error.value.code == code


def test_confirmation_expires_and_ttl_is_bounded(tmp_path) -> None:
    current = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    service = ConfirmationService(tmp_path / "confirmation.sqlite", clock=lambda: current[0])
    grant = service.issue(
        project_id="project",
        thread_id=None,
        root_run_id="root-run",
        operation_id="operation",
        action_hash=_hash(),
        ttl_seconds=600,
    )
    current[0] += timedelta(seconds=600)

    with pytest.raises(ConfirmationError) as expired:
        service.consume(
            confirmation_id=grant.confirmation_id,
            project_id="project",
            thread_id=None,
            root_run_id="root-run",
            operation_id="operation",
            action_hash=_hash(),
            continuation_checkpoint={"checkpoint_id": "confirmation:operation"},
        )
    assert expired.value.code == "confirmation_expired"

    with pytest.raises(ValueError, match="between 1 and 600"):
        service.issue(
            project_id="project",
            thread_id=None,
            root_run_id="root-run",
            operation_id="operation-2",
            action_hash=_hash(),
            ttl_seconds=601,
        )


def test_confirmation_database_uses_versioned_schema(tmp_path) -> None:
    service = ConfirmationService(tmp_path / "confirmation.sqlite")

    assert service.schema_version() == 2


def test_confirmation_consumption_atomically_persists_continuation_checkpoint(tmp_path) -> None:
    service = ConfirmationService(tmp_path / "confirmation.sqlite")
    grant = service.issue(
        project_id="project",
        thread_id="thread",
        root_run_id="run",
        operation_id="operation",
        action_hash=_hash(),
    )
    checkpoint = {
        "checkpoint_id": "confirmation:operation",
        "project_id": "project",
        "thread_id": "thread",
        "root_run_id": "run",
        "operation_id": "operation",
    }

    service.consume(
        confirmation_id=grant.confirmation_id,
        project_id="project",
        thread_id="thread",
        root_run_id="run",
        operation_id="operation",
        action_hash=_hash(),
        continuation_checkpoint=checkpoint,
    )

    assert service.continuation_checkpoint(grant.confirmation_id) == checkpoint

