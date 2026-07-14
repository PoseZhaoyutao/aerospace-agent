"""Discovery-only navigation and orbit-determination domain boundary."""

from aerospace_agent.domains.base import build_interface_descriptor

from .contracts import OpticalNavigationRequest, OpticalObservation

DESCRIPTOR = build_interface_descriptor("navigation_orbit_determination")


def request_capability_gap(requested_by_step_id, required_contract):
    return DESCRIPTOR.capability_gap(requested_by_step_id, required_contract)


__all__ = [
    "DESCRIPTOR",
    "OpticalNavigationRequest",
    "OpticalObservation",
    "request_capability_gap",
]

