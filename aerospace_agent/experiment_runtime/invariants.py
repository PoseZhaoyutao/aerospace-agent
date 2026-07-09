"""Aerospace invariant checks used by minimal closed-loop experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping


def _check(code: str, status: str, message: str, evidence: Any = None) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "code": code,
        "status": status,
        "message": message,
    }
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _path_exists(value: Any) -> bool:
    if not value:
        return False
    try:
        return Path(str(value)).is_file()
    except OSError:
        return False


def _missing_fields(value: Any, required: list[str]) -> list[str]:
    if not isinstance(value, Mapping):
        return list(required)
    return [field for field in required if value.get(field) in (None, "", [])]


def _has_any_field(value: Any, candidates: list[str]) -> bool:
    return isinstance(value, Mapping) and any(value.get(field) not in (None, "", []) for field in candidates)


def _attitude_missing(attitude: Any) -> list[str]:
    missing = _missing_fields(attitude, ["representation", "frame"])
    representation = str(attitude.get("representation", "")).lower() if isinstance(attitude, Mapping) else ""
    if representation == "quaternion":
        missing.extend(_missing_fields(attitude, ["quaternion"]))
    elif representation in {"euler", "euler_angles"}:
        if not _has_any_field(attitude, ["euler_deg", "euler_rad"]):
            missing.append("euler_deg_or_euler_rad")
    elif representation == "dcm":
        missing.extend(_missing_fields(attitude, ["dcm"]))
    elif not _has_any_field(attitude, ["quaternion", "euler_deg", "euler_rad", "dcm"]):
        missing.append("attitude_values")
    return sorted(set(missing))


def check_aerospace_invariants(
    config: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return pass/risk/fail checks for minimal aerospace run metadata.

    Missing domain metadata is treated as risk, not as success. This is a
    deliberate guardrail for agent output: a completed run can still be
    scientifically weak.
    """

    artifacts = artifacts or {}
    initial_state = dict(config.get("initial_state") or {})
    epoch = dict(initial_state.get("epoch") or config.get("epoch") or {})
    frame = dict(initial_state.get("frame") or config.get("frame") or {})

    checks = []

    if epoch.get("scale") and epoch.get("format") and epoch.get("value"):
        checks.append(_check("time_system", "passed", "Epoch value, scale, and format are explicit.", epoch))
    else:
        checks.append(_check("time_system", "risk", "Epoch is missing value, scale, or format.", epoch))

    if frame.get("name") and frame.get("center") and frame.get("realization"):
        checks.append(_check("frame", "passed", "Frame name, center, and realization are explicit.", frame))
    else:
        checks.append(_check("frame", "risk", "Frame is missing name, center, or realization.", frame))

    units = config.get("units")
    if isinstance(units, Mapping) and units.get("position") == "m" and units.get("velocity") == "m/s":
        checks.append(_check("units", "passed", "Position and velocity units are explicit SI units.", units))
    else:
        checks.append(_check("units", "risk", "SI units for position and velocity are not explicit.", units))

    attitude_missing = _attitude_missing(config.get("attitude"))
    if not attitude_missing:
        checks.append(_check("attitude", "passed", "Attitude representation and values are explicit."))
    else:
        checks.append(
            _check(
                "attitude",
                "risk",
                "Attitude metadata is missing required representation, frame, or values.",
                {"missing": attitude_missing},
            )
        )

    camera_missing = _missing_fields(
        config.get("camera_model"),
        ["focal_length_m", "pixel_size_m", "resolution_px", "optical_axis"],
    )
    if not camera_missing:
        checks.append(_check("camera_model", "passed", "Camera intrinsics and optical axis are explicit."))
    else:
        checks.append(
            _check(
                "camera_model",
                "risk",
                "Camera model is missing required intrinsics or optical axis.",
                {"missing": camera_missing},
            )
        )

    image_origin = config.get("image_origin")
    image_origin_ok = (
        isinstance(image_origin, Mapping)
        and (
            image_origin.get("convention")
            or (image_origin.get("row_origin") and image_origin.get("col_origin"))
        )
    )
    if image_origin_ok:
        checks.append(_check("image_origin", "passed", "Image origin convention is explicit."))
    else:
        checks.append(
            _check(
                "image_origin",
                "risk",
                "Image origin must declare convention or row_origin plus col_origin.",
            )
        )

    photometry = config.get("photometry") or {}
    snr_model = config.get("snr_model") or {}
    photometry_missing = _missing_fields(
        photometry,
        ["exposure_s", "gain_e_per_dn"],
    )
    if not _has_any_field(photometry, ["source_magnitude", "zero_point_mag", "source_flux_e_per_s"]):
        photometry_missing.append("source_magnitude_or_flux")
    snr_missing = _missing_fields(snr_model, ["read_noise_e", "background_e_per_pixel"])
    if not photometry_missing and not snr_missing:
        checks.append(_check("snr_photometry", "passed", "Photometry and SNR model fields are explicit."))
    else:
        checks.append(
            _check(
                "snr_photometry",
                "risk",
                "Photometry or SNR model is missing required fields.",
                {"missing": sorted(set(photometry_missing + snr_missing))},
            )
        )

    psf_model = config.get("psf_model")
    psf_missing = _missing_fields(psf_model, ["type"])
    if not _has_any_field(psf_model, ["sigma_px", "fwhm_px"]):
        psf_missing.append("sigma_px_or_fwhm_px")
    if not psf_missing:
        checks.append(_check("psf_model", "passed", "PSF type and width are explicit."))
    else:
        checks.append(
            _check(
                "psf_model",
                "risk",
                "PSF model is missing type or width.",
                {"missing": sorted(set(psf_missing))},
            )
        )

    truth_mapping_missing = _missing_fields(
        config.get("truth_image_mapping"),
        ["truth_path", "image_path", "coordinate_columns"],
    )
    if not truth_mapping_missing:
        checks.append(_check("truth_image_consistency", "passed", "Truth-to-image mapping fields are explicit."))
    else:
        checks.append(
            _check(
                "truth_image_consistency",
                "risk",
                "Truth/image consistency cannot be checked without truth path, image path, and coordinate columns.",
                {"missing": truth_mapping_missing},
            )
        )

    required_artifacts = {
        "config_path": artifacts.get("config_path"),
        "ledger_path": artifacts.get("ledger_path"),
        "state_history_json": artifacts.get("state_history_json"),
        "state_history_csv": artifacts.get("state_history_csv"),
        "orbit_plot": artifacts.get("orbit_plot"),
        "reproduce_path": artifacts.get("reproduce_path"),
    }
    missing = sorted(name for name, value in required_artifacts.items() if not _path_exists(value))
    if missing:
        checks.append(
            _check(
                "run_completeness",
                "risk",
                "Run is missing required closed-loop files.",
                {"missing": missing},
            )
        )
    else:
        checks.append(_check("run_completeness", "passed", "Required closed-loop files exist.", required_artifacts))

    summary = {"passed": 0, "risk": 0, "failed": 0}
    for item in checks:
        if item["status"] in summary:
            summary[item["status"]] += 1

    return {
        "status": "passed" if summary["risk"] == 0 and summary["failed"] == 0 else "risk",
        "summary": summary,
        "checks": checks,
    }
