from __future__ import annotations

import importlib

import pytest

from aerospace_agent.langgraph_agent.agent_core.models import (
    CapabilityGap,
    CheckpointRef,
    DomainArtifact,
    PlanStep,
)


DOMAIN_IDS = {
    "simulation",
    "navigation_orbit_determination",
    "control_planning",
    "orbit_design",
    "mechanical_thermal_electrical",
    "fault_diagnosis_maintenance",
}


def _domains_module():
    try:
        return importlib.import_module("aerospace_agent.domains")
    except ModuleNotFoundError as exc:
        pytest.fail(f"domain interface packages are missing: {exc}")


def test_six_domain_packages_are_discovery_only_interface_manifests() -> None:
    domains = _domains_module()
    descriptors = domains.DOMAIN_DESCRIPTORS

    assert {item.manifest.capability_id for item in descriptors} == DOMAIN_IDS
    for descriptor in descriptors:
        assert descriptor.manifest.category == "domain"
        assert descriptor.manifest.status == "interface_only"
        assert descriptor.manifest.tool_names == []
        assert descriptor.manifest.source == (
            f"aerospace_agent.domains.{descriptor.manifest.capability_id}"
        )
        assert not hasattr(descriptor, "execute")
        assert not hasattr(descriptor, "checkpoint_state")


def test_domain_placeholders_return_only_capability_gaps_and_no_execution_objects() -> None:
    domains = _domains_module()

    for descriptor in domains.DOMAIN_DESCRIPTORS:
        gap = descriptor.capability_gap(
            requested_by_step_id="step-1",
            required_contract={"type": "object", "required": ["state"]},
        )
        assert isinstance(gap, CapabilityGap)
        assert gap.capability_id == descriptor.manifest.capability_id
        assert gap.resolution is None
        assert gap.candidates == []
        assert not isinstance(gap, (PlanStep, CheckpointRef, DomainArtifact))


def test_interface_manifest_cannot_be_mutated_into_an_available_capability() -> None:
    descriptor = _domains_module().DOMAIN_DESCRIPTORS[0]

    exposed = descriptor.manifest
    exposed.status = "available"

    assert descriptor.manifest.status == "interface_only"


@pytest.mark.parametrize("domain_id", sorted(DOMAIN_IDS))
def test_each_domain_package_exposes_the_same_non_executable_gap_contract(domain_id: str) -> None:
    module = importlib.import_module(f"aerospace_agent.domains.{domain_id}")

    assert module.DESCRIPTOR.manifest.status == "interface_only"
    result = module.request_capability_gap("step-2", {"type": "object"})
    assert isinstance(result, CapabilityGap)
    assert not hasattr(module, "execute")

