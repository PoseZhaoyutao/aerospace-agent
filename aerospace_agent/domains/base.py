"""Discovery-only contracts for aerospace domain boundaries."""

from __future__ import annotations

from typing import Any

from pydantic import Field, JsonValue, field_validator

from aerospace_agent.langgraph_agent.agent_core.models import (
    CapabilityGap,
    CapabilityManifest,
    FrozenContractModel,
)


class DomainDescriptor(FrozenContractModel):
    """A non-executable domain interface visible only to capability discovery."""

    manifest_data: dict[str, Any] = Field(alias="manifest", repr=False)
    input_overview: dict[str, JsonValue]
    output_overview: dict[str, JsonValue]
    gap_description: str

    @field_validator("manifest_data", mode="before")
    @classmethod
    def copy_manifest_data(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, CapabilityManifest):
            return value.model_dump(mode="python")
        if isinstance(value, dict):
            return dict(value)
        raise ValueError("domain descriptor manifest must be a CapabilityManifest")

    @property
    def manifest(self) -> CapabilityManifest:
        """Return a defensive copy so discovery cannot promote the interface."""

        return CapabilityManifest.model_validate(dict(self.manifest_data))

    def capability_gap(
        self,
        requested_by_step_id: str,
        required_contract: dict[str, JsonValue],
    ) -> CapabilityGap:
        return CapabilityGap(
            capability_id=self.manifest.capability_id,
            requested_by_step_id=requested_by_step_id,
            description=self.gap_description,
            required_contract=required_contract,
        )


def build_interface_descriptor(domain_id: str) -> DomainDescriptor:
    return DomainDescriptor(
        manifest=CapabilityManifest(
            capability_id=domain_id,
            version="1.0.0",
            category="domain",
            status="interface_only",
            intents=[domain_id],
            tool_names=[],
            risk_level="read_only",
            required_dependencies=[],
            validators=[],
            source=f"aerospace_agent.domains.{domain_id}",
        ),
        input_overview={
            "status": "interface_only",
            "description": "No executable input contract has been approved.",
        },
        output_overview={
            "status": "interface_only",
            "description": "No computational output is available.",
        },
        gap_description=(
            f"The {domain_id} boundary exists, but no verified domain implementation is available."
        ),
    )


__all__ = ["DomainDescriptor", "build_interface_descriptor"]

