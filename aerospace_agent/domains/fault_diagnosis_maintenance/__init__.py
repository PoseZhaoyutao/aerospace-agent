"""Discovery-only fault-diagnosis and maintenance domain boundary."""

from aerospace_agent.domains.base import build_interface_descriptor

DESCRIPTOR = build_interface_descriptor("fault_diagnosis_maintenance")


def request_capability_gap(requested_by_step_id, required_contract):
    return DESCRIPTOR.capability_gap(requested_by_step_id, required_contract)


__all__ = ["DESCRIPTOR", "request_capability_gap"]

