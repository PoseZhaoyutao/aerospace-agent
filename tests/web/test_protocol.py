import pytest

from aerospace_agent.langgraph_agent.schema import RunStatus
from aerospace_agent.web.protocol import (
    HealthResponse,
    RunStartRequest,
    RunTerminalEvent,
    ThreadCreateRequest,
    terminal_event_type,
)


def test_run_start_requires_non_empty_versioned_fields():
    request = RunStartRequest(
        request_id="req-1",
        thread_id="thread-1",
        message="hello",
    )
    assert request.schema_version == "1.0.0"
    assert request.type == "run.start"

    with pytest.raises(ValueError):
        RunStartRequest(request_id="", thread_id="thread-1", message="hello")
    with pytest.raises(ValueError):
        RunStartRequest(request_id="req-1", thread_id="thread-1", message="")


def test_protocol_models_reject_unknown_fields():
    with pytest.raises(ValueError):
        HealthResponse(status="ready", extra_field="nope")
    with pytest.raises(ValueError):
        ThreadCreateRequest(title="new", extra_field="nope")


@pytest.mark.parametrize(
    ("status", "event_type"),
    [
        (RunStatus.SUCCESS, "run.completed"),
        (RunStatus.PARTIAL, "run.completed"),
        (RunStatus.INTERRUPTED, "run.interrupted"),
        (RunStatus.ERROR, "run.failed"),
        (RunStatus.LIMIT_REACHED, "run.failed"),
        (RunStatus.CYCLE_DETECTED, "run.failed"),
    ],
)
def test_terminal_status_mapping_preserves_original_status(status, event_type):
    assert terminal_event_type(status) == event_type


def test_approval_is_a_reason_code_on_interrupted_event():
    event = RunTerminalEvent(
        type="run.interrupted",
        request_id="req-1",
        thread_id="thread-1",
        status=RunStatus.INTERRUPTED,
        reason_code="human_approval_required",
    )
    assert event.type == "run.interrupted"
    assert event.status == RunStatus.INTERRUPTED
    assert event.reason_code == "human_approval_required"
