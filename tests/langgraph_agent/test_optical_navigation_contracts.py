"""Executable boundary tests for the optical-navigation design.

Numerical estimation remains intentionally unavailable.  These tests prove
that its inputs are strict and that the current domain fails closed, while
the already available SpaceBasicTools support chain remains usable.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from aerospace_agent.domains.navigation_orbit_determination import (
    OpticalNavigationRequest,
    OpticalObservation,
    request_capability_gap,
)
from aerospace_agent.langgraph_agent.agent_core.models import CapabilityGap
from aerospace_agent.mcp.tools import TOOL_REGISTRY


def _observation(index: int, *, epoch: str | None = None) -> dict:
    return {
        "observation_id": f"obs-{index}",
        "epoch": epoch or f"2026-01-01T00:0{index}:00",
        "time_system": "UTC",
        "frame_id": "camera_body",
        "camera_id": "cam-01",
        "line_of_sight": [0.0, 0.0, 1.0],
        "angular_covariance_rad2": [[1e-8, 0.0], [0.0, 1e-8]],
        "catalog_ids": [f"HIP-{index}"],
        "exposure_duration_s": 0.02,
        "provenance": {"source": "synthetic", "fixture": "on-v1"},
    }


def test_valid_optical_observation_contract_is_strict_and_auditable() -> None:
    observation = OpticalObservation.model_validate(_observation(1))
    assert observation.time_system == "UTC"
    assert observation.frame_id == "camera_body"
    assert observation.provenance["source"] == "synthetic"

    extra = _observation(1)
    extra["unapproved_field"] = True
    with pytest.raises(ValidationError):
        OpticalObservation.model_validate(extra)


@pytest.mark.parametrize(
    "field, value",
    [
        ("line_of_sight", [0.0, 0.0, 0.0]),
        ("line_of_sight", [0.0, 0.0, 2.0]),
        ("line_of_sight", [float("nan"), 0.0, 1.0]),
        ("angular_covariance_rad2", [[0.0, 0.0], [0.0, 1e-8]]),
        ("angular_covariance_rad2", [[1e-8, 1e-3], [0.0, 1e-8]]),
        ("angular_covariance_rad2", [[1e-8, 2e-8], [2e-8, 1e-8]]),
    ],
)
def test_invalid_optical_geometry_is_rejected(field: str, value: list) -> None:
    payload = _observation(1)
    payload[field] = value
    with pytest.raises(ValidationError):
        OpticalObservation.model_validate(payload)


def test_request_contract_enforces_order_uniqueness_and_orbit_minimum() -> None:
    valid = OpticalNavigationRequest.model_validate(
        {
            "request_id": "on-request-1",
            "mode": "orbit_determination",
            "target_frame_id": "GCRF",
            "output_epoch": "2026-01-01T00:02:00",
            "observations": [_observation(1), _observation(2), _observation(3)],
            "required_outputs": ["state", "covariance", "provenance"],
        }
    )
    assert len(valid.observations) == 3

    duplicate = valid.model_dump(mode="python")
    duplicate["observations"][1]["observation_id"] = "obs-1"
    with pytest.raises(ValidationError):
        OpticalNavigationRequest.model_validate(duplicate)

    out_of_order = valid.model_dump(mode="python")
    out_of_order["observations"] = list(reversed(out_of_order["observations"]))
    with pytest.raises(ValidationError):
        OpticalNavigationRequest.model_validate(out_of_order)

    too_short = deepcopy(valid.model_dump(mode="python"))
    too_short["observations"] = too_short["observations"][:2]
    with pytest.raises(ValidationError):
        OpticalNavigationRequest.model_validate(too_short)


def test_navigation_domain_fails_closed_with_capability_gap() -> None:
    required_contract = OpticalNavigationRequest.model_json_schema()
    gap = request_capability_gap("estimate-orbit", required_contract)

    assert isinstance(gap, CapabilityGap)
    assert gap.capability_id == "navigation_orbit_determination"
    assert gap.resolution is None
    assert not hasattr(__import__("aerospace_agent.domains.navigation_orbit_determination", fromlist=["x"]), "execute")


def test_space_basic_support_chain_is_not_mislabeled_as_optical_navigation() -> None:
    state = {
        "epoch": {"value": "2026-01-01T00:00:00", "scale": "UTC", "format": "ISO"},
        "frame": {"name": "GCRF", "center": "Earth", "realization": "IERS2010"},
        "representation": "cartesian",
        "position_m": [6778137.0, 0.0, 0.0],
        "velocity_mps": [0.0, 7668.558175407055, 0.0],
    }
    converted_time = TOOL_REGISTRY["space.convert_time"](
        value="2026-01-01T00:00:00",
        from_scale="UTC",
        from_format="ISO",
        # Keep this offline and deterministic; a real UTC→TT conversion is
        # an estimator acceptance case and may require external leap-second
        # data, not a unit test dependency.
        to_scale="UTC",
        to_format="ISO",
    )
    transformed = TOOL_REGISTRY["space.transform_frame"](
        state_dict=state,
        target_frame="GCRF",
        target_center="Earth",
    )
    propagated = TOOL_REGISTRY["space.propagate_orbit"](
        initial_state_dict=state,
        force_model_dict={"gravity": "point_mass"},
        duration_s=120.0,
        output_step_s=60.0,
        engine="builtin",
    )
    assert converted_time["status"] == "success"
    assert transformed["status"] == "success"
    assert propagated["status"] == "success"
    assert propagated["metadata"]["frame"] == "GCRF"
    assert "optical_navigation" not in propagated["metadata"].get("tool", "")
