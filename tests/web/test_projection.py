from aerospace_agent.langgraph_agent.schema import AgentOutput, RunStatus
from aerospace_agent.web.projection import project_agent_output


def test_projection_preserves_terminal_status_and_approval_reason():
    output = AgentOutput(
        status=RunStatus.INTERRUPTED,
        answer="waiting",
        errors=[{"code": "human_approval_required", "message": "confirm"}],
    )

    projected = project_agent_output(output)

    assert projected["status"] == "interrupted"
    assert projected["event_type"] == "run.interrupted"
    assert projected["reason_code"] == "human_approval_required"


def test_projection_bounds_errors_warnings_metrics_and_redacts_secrets():
    output = AgentOutput(
        status=RunStatus.ERROR,
        errors=[{"message": "x" * 2000, "api_key": "do-not-leak"}] * 40,
        warnings=["w" * 1000] * 40,
        metrics={"token": "secret", "nested": {"value": "ok"}},
    )

    projected = project_agent_output(output)

    assert len(projected["errors"]) == 16
    assert len(projected["warnings"]) == 32
    assert len(projected["errors"][0]["message"]) == 1024
    assert projected["errors"][0]["api_key"] == "[REDACTED]"
    assert projected["metrics"]["token"] == "[REDACTED]"


def test_projection_never_exposes_model_objects():
    output = AgentOutput(status=RunStatus.PARTIAL, answer="answer")
    projected = project_agent_output(output)
    assert projected["answer"] == "answer"
    assert all(isinstance(value, (str, int, float, bool, type(None), list, dict)) for value in projected.values())
