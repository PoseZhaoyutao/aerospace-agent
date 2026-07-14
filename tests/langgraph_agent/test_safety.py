from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.offline

from aerospace_agent.langgraph_agent.safety import (
    ApprovalRequired,
    SafetyValidationError,
    SafetyValidator,
    orbital_specific_energy,
    two_body_acceleration,
    validate_orbital_payload,
    validate_tool_output,
)
from aerospace_agent.langgraph_agent.graph import ServiceBundle
from aerospace_agent.langgraph_agent.nodes import tool_execute_node, validate_output_node
from aerospace_agent.langgraph_agent.schema import ActionType, Decision, ToolCallRequest
from aerospace_agent.langgraph_agent.state import create_initial_state


def test_two_body_acceleration_and_energy_are_finite_and_physical():
    acceleration = two_body_acceleration([7_000_000.0, 0.0, 0.0], 3.986004418e14)
    energy = orbital_specific_energy([7_000_000.0, 0.0, 0.0], [0.0, 7_500.0, 0.0], 3.986004418e14)
    assert acceleration[0] < 0.0
    assert acceleration[1:] == (0.0, 0.0)
    assert math.isfinite(energy)


def test_two_body_checks_reject_nonfinite_and_zero_radius():
    with pytest.raises(SafetyValidationError):
        two_body_acceleration([0.0, 0.0, 0.0], 3.986e14)
    with pytest.raises(SafetyValidationError):
        orbital_specific_energy([math.nan, 0.0, 0.0], [0.0, 1.0, 0.0], 3.986e14)


def test_orbital_payload_requires_units_frame_timescale_and_finite_values():
    valid = {
        "position": [7_000_000.0, 0.0, 0.0],
        "velocity": [0.0, 7_500.0, 0.0],
        "units": {"position": "m", "velocity": "m/s"},
        "frame": "gcrf",
        "time_scale": "utc",
    }
    assert validate_orbital_payload(valid)["ok"] is True
    with pytest.raises(SafetyValidationError, match="units"):
        validate_orbital_payload({**valid, "units": {}})
    with pytest.raises(SafetyValidationError, match="finite"):
        validate_orbital_payload({**valid, "position": [math.inf, 0.0, 0.0]})


def test_orbital_payload_checks_two_body_acceleration_and_energy_when_declared():
    payload = {
        "position": [7_000_000.0, 0.0, 0.0],
        "velocity": [0.0, 7_500.0, 0.0],
        "acceleration": [-8.1347, 0.0, 0.0],
        "specific_energy": -28_817_920.0,
        "mu": 3.986004418e14,
        "units": {"position": "m", "velocity": "m/s", "acceleration": "m/s^2", "specific_energy": "J/kg"},
        "frame": "gcrf",
        "time_scale": "utc",
    }
    assert validate_orbital_payload(payload)["ok"] is True
    with pytest.raises(SafetyValidationError, match="acceleration"):
        validate_orbital_payload({**payload, "acceleration": [1.0, 0.0, 0.0]})


def test_tool_output_validation_and_high_risk_approval_gate():
    with pytest.raises(SafetyValidationError):
        validate_tool_output("orbit_propagation", {"position": [1.0, 2.0, 3.0]})
    with pytest.raises(ApprovalRequired):
        validate_tool_output("maneuver_execute", {"status": "success"})
    assert validate_tool_output("orbit_query", {"status": "success"})["ok"] is True


def test_nested_mcp_orbit_state_requires_frame_and_time_scale():
    request = {
        "initial_state_dict": {
            "position_m": [7_000_000.0, 0.0, 0.0],
            "velocity_mps": [0.0, 7_500.0, 0.0],
        }
    }
    with pytest.raises(SafetyValidationError, match="frame"):
        SafetyValidator().validate_tool_request("space.propagate_orbit", request)
    request["initial_state_dict"].update({"frame": "gcrf", "time_scale": "utc"})
    assert SafetyValidator().validate_tool_request("space.propagate_orbit", request)["ok"] is True


def test_graph_tool_execution_blocks_high_risk_without_confirmation():
    state = create_initial_state(thread_id="safety")
    state["decision"] = Decision(
        action=ActionType.CALL_TOOL,
        rationale="execute maneuver",
        tool_request=ToolCallRequest(tool_name="maneuver_execute", arguments={}, is_read_only=False),
    ).model_dump(mode="json")
    services = ServiceBundle(
        mcp_gateway=lambda _request: {"status": "success"},
        safety=SafetyValidator(),
    )
    result = tool_execute_node(state, services=services)
    assert result["tool_results"][-1]["status"] == "blocked"
    assert result["errors"][-1]["category"] == "safety_error"


def test_graph_output_validation_rejects_nonfinite_tool_result():
    state = create_initial_state(thread_id="safety")
    state["tool_results"] = [{
        "tool_name": "orbit_query", "status": "success",
        "result": {"position": [float("inf"), 0.0, 0.0], "units": {"position": "m"}, "frame": "gcrf", "time_scale": "utc"},
    }]
    result = validate_output_node(state)
    assert result["output_valid"] is False
    assert result["status"] == "error"
    assert result["errors"][-1]["category"] == "safety_error"


def test_evolution_write_gate_requires_explicit_confirmation():
    with pytest.raises(ApprovalRequired):
        SafetyValidator().validate_evolution_write({"run_id": "r", "changes": [{"path": "memory/x.md"}]})
    assert SafetyValidator(approval_gate=lambda *_args: True).validate_evolution_write({"run_id": "r"})["ok"] is True
