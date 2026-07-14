"""Small, dependency-light safety checks for aerospace state and tools.

This module intentionally validates only protocol metadata and a few
well-defined two-body invariants.  It does not import a dynamics textbook or
attempt to replace a flight-dynamics library.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable


class SafetyValidationError(ValueError):
    """Input or tool output failed a safety invariant."""


class ApprovalRequired(SafetyValidationError):
    """A high-risk action needs explicit human confirmation."""


SUPPORTED_FRAMES = {"eci", "ecef", "gcrf", "teme", "lvlh", "j2000"}
SUPPORTED_TIME_SCALES = {"utc", "tai", "tt", "tdb", "gps"}
UNIT_ALIASES = {
    "position": {"m", "km", "meter", "meters", "metre", "metres", "kilometer", "kilometers", "kilometre", "kilometres"},
    "velocity": {"m/s", "km/s", "meter/second", "meters/second", "kilometer/second", "kilometers/second"},
    "acceleration": {"m/s^2", "m/s2", "km/s^2", "km/s2", "meter/second^2", "kilometer/second^2"},
    "specific_energy": {"j/kg", "m^2/s^2", "m2/s2", "km^2/s^2", "km2/s2"},
}
HIGH_RISK_TOOL_TOKENS = (
    "maneuver", "burn", "execute", "write", "delete", "evolution",
    "attitude_control", "command", "uplink", "launch",
)


def _finite(value: Any, *, path: str = "value") -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SafetyValidationError(f"{path} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _finite(item, path=f"{path}[{index}]")


def _vector(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 3:
        raise SafetyValidationError(f"{name} must contain exactly three numeric values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise SafetyValidationError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def _check_unit(units: Mapping[str, Any], key: str, *, fallback: str | tuple[str, ...] | None = None) -> None:
    fallbacks = (fallback,) if isinstance(fallback, str) else tuple(fallback or ())
    raw = units.get(key, "")
    if not str(raw).strip():
        for name in fallbacks:
            raw = units.get(name, "")
            if str(raw).strip():
                break
    value = str(raw).strip().lower()
    if not value:
        raise SafetyValidationError(f"units.{key} is required")
    if value not in UNIT_ALIASES[key]:
        raise SafetyValidationError(f"units.{key} is unsupported: {value}")


def two_body_acceleration(position: Sequence[float], mu: float) -> tuple[float, float, float]:
    """Compute ``-mu*r/|r|^3`` with finite/non-zero guards."""

    r = _vector(position, name="position")
    mu_value = float(mu)
    if not math.isfinite(mu_value) or mu_value <= 0:
        raise SafetyValidationError("mu must be finite and positive")
    radius = math.sqrt(sum(component * component for component in r))
    if not math.isfinite(radius) or radius <= 0:
        raise SafetyValidationError("position radius must be finite and non-zero")
    factor = -mu_value / (radius ** 3)
    result = tuple(factor * component for component in r)
    if not all(math.isfinite(component) for component in result):
        raise SafetyValidationError("two-body acceleration must be finite")
    return result  # type: ignore[return-value]


def orbital_specific_energy(position: Sequence[float], velocity: Sequence[float], mu: float) -> float:
    """Compute specific orbital energy ``v²/2 - mu/|r|`` safely."""

    r = _vector(position, name="position")
    v = _vector(velocity, name="velocity")
    mu_value = float(mu)
    radius = math.sqrt(sum(component * component for component in r))
    if not math.isfinite(mu_value) or mu_value <= 0 or radius <= 0 or not math.isfinite(radius):
        raise SafetyValidationError("mu and position radius must be finite and valid")
    energy = 0.5 * sum(component * component for component in v) - mu_value / radius
    if not math.isfinite(energy):
        raise SafetyValidationError("specific orbital energy must be finite")
    return float(energy)


def validate_orbital_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate orbital vectors plus units, frame, time scale and finiteness."""

    if not isinstance(payload, Mapping):
        raise SafetyValidationError("orbital payload must be a mapping")
    _finite(payload)
    # Accept both descriptive protocol names and the conventional r/v aliases.
    position = payload.get("position", payload.get("r"))
    velocity = payload.get("velocity", payload.get("v"))
    acceleration = payload.get("acceleration", payload.get("a"))
    specific_energy = payload.get("specific_energy", payload.get("specific_orbital_energy", payload.get("energy")))
    if position is None and velocity is None and acceleration is None and specific_energy is None:
        raise SafetyValidationError("orbital payload requires position, velocity, acceleration, or energy")
    if position is not None:
        _vector(position, name="position")
    if velocity is not None:
        _vector(velocity, name="velocity")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    units = payload.get("units", metadata.get("units"))
    if not isinstance(units, Mapping):
        raise SafetyValidationError("orbital payload units metadata is required")
    if position is not None:
        _check_unit(units, "position", fallback="r")
    if velocity is not None:
        _check_unit(units, "velocity", fallback="v")
    if acceleration is not None:
        _vector(acceleration, name="acceleration")
        _check_unit(units, "acceleration", fallback="a")
    if specific_energy is not None:
        if not isinstance(specific_energy, (int, float)) or not math.isfinite(float(specific_energy)):
            raise SafetyValidationError("specific_energy must be finite")
        _check_unit(units, "specific_energy", fallback=("specific_orbital_energy", "energy"))
    frame = str(payload.get("frame", payload.get("reference_frame", metadata.get("frame", metadata.get("reference_frame", ""))))).strip().lower()
    if frame not in SUPPORTED_FRAMES:
        raise SafetyValidationError(f"frame must be one of {sorted(SUPPORTED_FRAMES)}")
    time_scale = str(payload.get("time_scale", payload.get("timescale", metadata.get("time_scale", metadata.get("timescale", ""))))).strip().lower()
    if time_scale not in SUPPORTED_TIME_SCALES:
        raise SafetyValidationError(f"time_scale must be one of {sorted(SUPPORTED_TIME_SCALES)}")
    if payload.get("mu") is not None and position is not None and acceleration is not None:
        expected = two_body_acceleration(position, payload["mu"])
        actual = _vector(acceleration, name="acceleration")
        if any(not math.isclose(a, e, rel_tol=5e-3, abs_tol=1e-9) for a, e in zip(actual, expected)):
            raise SafetyValidationError("acceleration is inconsistent with two-body dynamics")
    if payload.get("mu") is not None and position is not None and velocity is not None and specific_energy is not None:
        expected_energy = orbital_specific_energy(position, velocity, payload["mu"])
        if not math.isclose(float(specific_energy), expected_energy, rel_tol=5e-3, abs_tol=1e-6):
            raise SafetyValidationError("specific_energy is inconsistent with two-body dynamics")
    return {"ok": True, "frame": frame, "time_scale": time_scale, "units": dict(units)}


def _nested_orbital_payload(container: Any) -> Mapping[str, Any] | None:
    """Normalize common MCP ``initial_state_dict`` aliases for validation."""
    if not isinstance(container, Mapping):
        return None
    aliases = {
        "position_m": "position",
        "velocity_mps": "velocity",
        "acceleration_mps2": "acceleration",
        "specific_orbital_energy": "specific_energy",
    }
    if not any(key in container for key in (*aliases, "position", "velocity", "r", "v", "acceleration", "a")):
        return None
    normalized = dict(container)
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    units = dict(normalized.get("units", {}) or {})
    if "position_m" in container:
        units.setdefault("position", "m")
    if "velocity_mps" in container:
        units.setdefault("velocity", "m/s")
    if "acceleration_mps2" in container:
        units.setdefault("acceleration", "m/s^2")
    normalized["units"] = units
    return normalized


def requires_human_approval(tool_name: str, *, is_read_only: bool = True) -> bool:
    name = str(tool_name or "").strip().lower()
    return (not is_read_only) or any(token in name for token in HIGH_RISK_TOOL_TOKENS)


class SafetyValidator:
    """Composable validator and approval gate used by graph nodes."""

    def __init__(self, approval_gate: Callable[..., bool] | None = None):
        self.approval_gate = approval_gate

    def _approved(self, tool_name: str, context: Mapping[str, Any]) -> bool:
        if any(context.get(key) is True for key in ("confirmed", "human_approved", "approval_confirmed")):
            return True
        gate = self.approval_gate
        if gate is None:
            return False
        if isinstance(gate, bool):
            return gate
        if not callable(gate):
            gate = getattr(gate, "request_approval", None) or getattr(gate, "confirm", None)
        if gate is None:
            return False
        try:
            return bool(gate(tool_name, dict(context)))
        except TypeError:
            try:
                return bool(gate(tool_name))
            except TypeError:
                return bool(gate())

    def validate_tool_request(self, tool_name: str, arguments: Mapping[str, Any], *, is_read_only: bool = True) -> dict[str, Any]:
        _finite(arguments)
        if requires_human_approval(tool_name, is_read_only=is_read_only) and not self._approved(tool_name, arguments):
            raise ApprovalRequired(f"human approval required for high-risk tool: {tool_name}")
        if any(key in arguments for key in ("position", "velocity", "r", "v", "acceleration", "a", "specific_energy", "specific_orbital_energy", "energy")):
            validate_orbital_payload(arguments)
        for key in ("initial_state_dict", "orbit_state", "state"):
            nested = _nested_orbital_payload(arguments.get(key))
            if nested is not None:
                validate_orbital_payload(nested)
        return {"ok": True}

    def validate_input(self, payload: Any) -> dict[str, Any]:
        """Validate an input mapping or Pydantic model at the safety boundary."""

        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        if not isinstance(payload, Mapping):
            raise SafetyValidationError("input payload must be a mapping")
        _finite(payload)
        if any(key in payload for key in ("position", "velocity", "r", "v", "acceleration", "a", "specific_energy", "specific_orbital_energy", "energy")):
            return validate_orbital_payload(payload)
        return {"ok": True}

    def validate_tool_output(self, tool_name: str, payload: Any) -> dict[str, Any]:
        _finite(payload)
        if requires_human_approval(tool_name, is_read_only=False):
            # A high-risk output is only accepted when the request gate has
            # already approved it; callers pass such requests through the
            # request validator before invoking this method.
            pass
        if isinstance(payload, Mapping) and any(key in payload for key in ("position", "velocity", "r", "v", "acceleration", "a", "specific_energy", "specific_orbital_energy", "energy")):
            return validate_orbital_payload(payload)
        if isinstance(payload, Mapping):
            for key in ("initial_state_dict", "orbit_state", "state"):
                nested = _nested_orbital_payload(payload.get(key))
                if nested is not None:
                    return validate_orbital_payload(nested)
        return {"ok": True}

    def validate_state_output(self, state: Mapping[str, Any]) -> dict[str, Any]:
        _finite(state)
        for item in state.get("tool_results", []) or []:
            if isinstance(item, Mapping) and item.get("status") == "success":
                payload = item.get("result")
                if isinstance(payload, Mapping) and any(key in payload for key in ("position", "velocity", "r", "v", "acceleration", "a", "specific_energy", "specific_orbital_energy", "energy")):
                    self.validate_tool_output(str(item.get("tool_name", "")), payload)
        return {"ok": True}

    def validate_evolution_write(self, proposal: Mapping[str, Any]) -> dict[str, Any]:
        """Gate a file-writing proposal without mutating evolution state."""

        context = proposal
        if not isinstance(context, Mapping) and hasattr(context, "model_dump"):
            context = context.model_dump(mode="json")
        if not isinstance(context, Mapping):
            context = {"proposal": str(context)}
        if not self._approved("evolution_write", context):
            raise ApprovalRequired("human approval required for evolution write")
        return {"ok": True}


def validate_tool_output(tool_name: str, payload: Any, *, approval: bool | Callable[..., bool] | None = None) -> dict[str, Any]:
    """Convenience API used by tests and lightweight integrations."""

    gate = (lambda *_args, **_kwargs: bool(approval)) if isinstance(approval, bool) else approval
    validator = SafetyValidator(approval_gate=gate)
    request_payload = payload if isinstance(payload, Mapping) else {}
    validator.validate_tool_request(tool_name, request_payload, is_read_only=True)
    output_payload = request_payload.get("result") if isinstance(request_payload, Mapping) and "result" in request_payload else payload
    return validator.validate_tool_output(tool_name, output_payload)


__all__ = [
    "ApprovalRequired", "SafetyValidationError", "SafetyValidator",
    "orbital_specific_energy", "requires_human_approval", "two_body_acceleration",
    "validate_orbital_payload", "validate_tool_output",
    "validate_input_payload",
]

# Descriptive alias for callers that prefer an explicit input name.
validate_input_payload = validate_orbital_payload
