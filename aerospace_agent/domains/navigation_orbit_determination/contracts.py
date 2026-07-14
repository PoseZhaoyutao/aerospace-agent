"""Non-executable optical-navigation data contracts.

These contracts define the boundary for a future navigation/orbit-determination
implementation.  They intentionally contain validation only; no estimator,
star matcher, filter, or propagator is exposed from this domain package.
"""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, model_validator

from aerospace_agent.langgraph_agent.agent_core.models import ContractModel


class OpticalObservation(ContractModel):
    """One camera line-of-sight observation with auditable metadata."""

    observation_id: str = Field(min_length=1)
    epoch: str = Field(min_length=1)
    time_system: Literal["UTC", "TAI", "TT", "TDB"]
    frame_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    line_of_sight: list[float] = Field(min_length=3, max_length=3)
    angular_covariance_rad2: list[list[float]] = Field(min_length=2, max_length=2)
    catalog_ids: list[str] = Field(default_factory=list)
    exposure_duration_s: float = Field(gt=0)
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        if not all(math.isfinite(value) for value in self.line_of_sight):
            raise ValueError("line_of_sight values must be finite")
        norm = math.sqrt(sum(value * value for value in self.line_of_sight))
        if norm <= 0:
            raise ValueError("line_of_sight must be non-zero")
        if abs(norm - 1.0) > 1e-6:
            raise ValueError("line_of_sight must be a unit vector")
        if any(len(row) != 2 for row in self.angular_covariance_rad2):
            raise ValueError("angular_covariance_rad2 must be a 2x2 matrix")
        if not all(math.isfinite(value) for row in self.angular_covariance_rad2 for value in row):
            raise ValueError("angular_covariance_rad2 values must be finite")
        diagonal = (self.angular_covariance_rad2[0][0], self.angular_covariance_rad2[1][1])
        if any(value <= 0 for value in diagonal):
            raise ValueError("angular covariance diagonal must be positive")
        if abs(self.angular_covariance_rad2[0][1] - self.angular_covariance_rad2[1][0]) > 1e-12:
            raise ValueError("angular covariance must be symmetric")
        determinant = (
            self.angular_covariance_rad2[0][0] * self.angular_covariance_rad2[1][1]
            - self.angular_covariance_rad2[0][1] * self.angular_covariance_rad2[1][0]
        )
        if determinant < -1e-24:
            raise ValueError("angular covariance must be positive semidefinite")
        return self


class OpticalNavigationRequest(ContractModel):
    """Future estimator input; it is not executable in the current release."""

    request_id: str = Field(min_length=1)
    mode: Literal["attitude", "orbit_determination"]
    target_frame_id: str = Field(min_length=1)
    output_epoch: str = Field(min_length=1)
    observations: list[OpticalObservation] = Field(min_length=1)
    required_outputs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_observations(self) -> Self:
        identifiers = [item.observation_id for item in self.observations]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("observation_id values must be unique")
        epochs = [item.epoch for item in self.observations]
        if epochs != sorted(epochs):
            raise ValueError("observations must be ordered by epoch")
        if self.mode == "orbit_determination" and len(self.observations) < 3:
            raise ValueError("orbit_determination requires at least three observations")
        return self


__all__ = ["OpticalNavigationRequest", "OpticalObservation"]
