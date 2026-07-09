"""`space.*` MCP tool contracts and thin wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping

from .environment_tools import check_engine_availability
from .propagation_tools import propagate_orbit
from .time_tools import convert_time


def _schema_object(properties: Dict[str, Any] | None = None, required: List[str] | None = None) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": properties or {},
    }
    if required:
        schema["required"] = required
    return schema


def _spec(
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    output_schema: Dict[str, Any],
    units: str,
    frame: str,
    time_system: str,
    risk_level: str,
    required_files: List[str] | None = None,
    generated_artifacts: List[str] | None = None,
    validation_rules: List[str] | None = None,
    error_codes: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "units": units,
        "frame": frame,
        "time_system": time_system,
        "risk_level": risk_level,
        "required_files": required_files or [],
        "generated_artifacts": generated_artifacts or [],
        "validation_rules": validation_rules or [],
        "error_codes": error_codes or [],
    }


SPACE_TOOL_SPECS: List[Dict[str, Any]] = [
    _spec(
        "space.check_environment",
        "Check availability and versions of local astrodynamics engines.",
        _schema_object({"engines": {"type": "array", "items": {"type": "string"}}}),
        _schema_object({"status": {"type": "string"}, "engines": {"type": "object"}}),
        "N/A",
        "N/A",
        "N/A",
        "P0",
        validation_rules=["Return structured status for every requested engine."],
        error_codes=["ENGINE_CHECK_FAILED"],
    ),
    _spec(
        "space.convert_time",
        "Convert time value across explicit scales and formats.",
        _schema_object(
            {
                "value": {"type": ["string", "number"]},
                "from_scale": {"type": "string"},
                "from_format": {"type": "string"},
                "to_scale": {"type": "string"},
                "to_format": {"type": "string"},
            },
            ["value"],
        ),
        _schema_object({"status": {"type": "string"}, "value": {"type": ["string", "number"]}}),
        "s or calendar units as declared by input",
        "N/A",
        "UTC/TAI/TT/TDB/ET",
        "P0",
        validation_rules=["Input and output time scale must be explicit."],
        error_codes=["TIME_CONVERSION_FAILED", "MISSING_TIME_SYSTEM"],
    ),
    _spec(
        "space.propagate_orbit",
        "Propagate a canonical orbit state with explicit units, frame, and force model.",
        _schema_object(
            {
                "initial_state_dict": {"type": "object"},
                "force_model_dict": {"type": "object"},
                "duration_s": {"type": "number"},
                "output_step_s": {"type": "number"},
                "engine": {"type": "string"},
            },
            ["initial_state_dict", "force_model_dict", "duration_s"],
        ),
        _schema_object({"state_history": {"type": "array"}, "metadata": {"type": "object"}}),
        "SI: m, m/s, s",
        "Declared in initial_state_dict.frame",
        "Declared in initial_state_dict.epoch",
        "P0",
        validation_rules=[
            "initial_state_dict must contain epoch and frame.",
            "duration_s must be positive.",
            "metadata must report engine, units, frame, and propagator_type.",
        ],
        error_codes=["PROPAGATION_FAILED", "MISSING_FRAME", "MISSING_UNITS"],
    ),
    _spec(
        "space.plot_orbit",
        "Generate a PNG orbit plot from propagated state history.",
        _schema_object(
            {
                "state_history": {"type": "array"},
                "output_path": {"type": "string"},
            },
            ["state_history", "output_path"],
        ),
        _schema_object({"status": {"type": "string"}, "artifact_path": {"type": "string"}}),
        "km on plotted axes; source states remain SI",
        "Inherited from state_history",
        "Inherited from state_history",
        "P0",
        generated_artifacts=["orbit.png"],
        validation_rules=["state_history must contain position_m samples."],
        error_codes=["PLOT_FAILED", "EMPTY_STATE_HISTORY"],
    ),
    _spec(
        "space.build_report",
        "Write a Markdown report with completed, verified, unverified, assumptions, failures, and risks.",
        _schema_object(
            {
                "run_dir": {"type": "string"},
                "result": {"type": "object"},
            },
            ["run_dir", "result"],
        ),
        _schema_object({"status": {"type": "string"}, "report_path": {"type": "string"}}),
        "N/A",
        "N/A",
        "N/A",
        "P0",
        generated_artifacts=["report.md"],
        validation_rules=["Report must contain anti-hype sections."],
        error_codes=["REPORT_FAILED"],
    ),
    _spec(
        "space.reproduce_run",
        "Write a shell script that reruns the same config through the experiment CLI.",
        _schema_object(
            {
                "run_dir": {"type": "string"},
                "task": {"type": "string"},
                "config_name": {"type": "string"},
            },
            ["run_dir"],
        ),
        _schema_object({"status": {"type": "string"}, "reproduce_path": {"type": "string"}}),
        "N/A",
        "N/A",
        "N/A",
        "P0",
        required_files=["config.yaml"],
        generated_artifacts=["reproduce.sh"],
        validation_rules=["Script must call the local experiment CLI with a config file."],
        error_codes=["REPRODUCE_SCRIPT_FAILED"],
    ),
    _spec(
        "space.load_spice_kernels",
        "Stub: load and validate SPICE kernels for geometry calculations.",
        _schema_object({"kernel_paths": {"type": "array", "items": {"type": "string"}}}, ["kernel_paths"]),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "N/A",
        "SPICE frames declared by kernels",
        "Kernel coverage must declare supported epochs",
        "P1",
        required_files=["SPICE kernels"],
        validation_rules=["Kernel coverage and frame IDs must be checked before geometry claims."],
        error_codes=["NOT_IMPLEMENTED", "KERNEL_MISSING", "KERNEL_COVERAGE_GAP"],
    ),
    _spec(
        "space.transform_frame",
        "Stub: transform state vectors between declared astrodynamics frames.",
        _schema_object({"state": {"type": "object"}, "target_frame": {"type": "object"}}, ["state", "target_frame"]),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "SI: m, m/s, s",
        "Input and target frames must be explicit",
        "Epoch must be explicit",
        "P1",
        validation_rules=["Input state, source frame, target frame, and epoch must be explicit."],
        error_codes=["NOT_IMPLEMENTED", "MISSING_FRAME", "MISSING_EPOCH"],
    ),
    _spec(
        "space.compute_attitude",
        "Stub: compute or convert spacecraft attitude with explicit representation.",
        _schema_object({"attitude_spec": {"type": "object"}}, ["attitude_spec"]),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "rad or deg as declared by input",
        "Body/inertial/sensor frames must be explicit",
        "Epoch must be explicit",
        "P1",
        validation_rules=["Quaternion, Euler, or DCM representation must be explicit."],
        error_codes=["NOT_IMPLEMENTED", "MISSING_ATTITUDE_REPRESENTATION"],
    ),
    _spec(
        "space.project_to_sensor",
        "Stub: project inertial truth states to sensor image coordinates.",
        _schema_object({"truth": {"type": "object"}, "camera_model": {"type": "object"}}, ["truth", "camera_model"]),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "m, rad, pixel",
        "Truth, spacecraft, sensor, and image frames must be explicit",
        "Exposure epoch must be explicit",
        "P1",
        validation_rules=["Camera intrinsics, attitude, and image origin must be explicit."],
        error_codes=["NOT_IMPLEMENTED", "MISSING_CAMERA_MODEL", "MISSING_IMAGE_ORIGIN"],
    ),
    _spec(
        "space.query_star_catalog",
        "Stub: query a star catalog for image simulation.",
        _schema_object({"catalog": {"type": "string"}, "region": {"type": "object"}}, ["catalog", "region"]),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "mag, deg/rad as declared by query",
        "Catalog and query region frame must be explicit",
        "Catalog epoch/proper motion policy must be explicit",
        "P1",
        required_files=["catalog file or configured catalog service"],
        validation_rules=["Catalog version and magnitude band must be explicit."],
        error_codes=["NOT_IMPLEMENTED", "CATALOG_UNAVAILABLE"],
    ),
    _spec(
        "space.render_starfield",
        "Stub: render a synthetic starfield image from catalog and camera metadata.",
        _schema_object({"stars": {"type": "array"}, "camera_model": {"type": "object"}}, ["stars", "camera_model"]),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "pixel, DN/electron as declared by detector model",
        "Sensor and image frames must be explicit",
        "Exposure epoch must be explicit",
        "P1",
        generated_artifacts=["starfield image"],
        validation_rules=["PSF, exposure, gain, and image origin must be explicit."],
        error_codes=["NOT_IMPLEMENTED", "MISSING_PSF", "MISSING_PHOTOMETRY"],
    ),
    _spec(
        "space.compute_snr",
        "Stub: compute SNR from source, background, detector, and exposure metadata.",
        _schema_object({"photometry": {"type": "object"}, "detector": {"type": "object"}}, ["photometry", "detector"]),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "electron, DN, s, mag",
        "N/A",
        "Exposure timing must be explicit",
        "P1",
        validation_rules=["Source flux, background, gain, read noise, and exposure must be explicit."],
        error_codes=["NOT_IMPLEMENTED", "MISSING_SNR_MODEL"],
    ),
    _spec(
        "space.validate_truth",
        "Stub: validate truth tables against generated images and artifact paths.",
        _schema_object({"truth_path": {"type": "string"}, "image_path": {"type": "string"}}, ["truth_path", "image_path"]),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "pixel and declared truth units",
        "Truth and image frames must be explicit",
        "Exposure epoch must be explicit",
        "P1",
        required_files=["truth table", "image artifact"],
        validation_rules=["Truth table rows must map to declared image artifacts and coordinate origin."],
        error_codes=["NOT_IMPLEMENTED", "TRUTH_IMAGE_MISMATCH"],
    ),
    _spec(
        "space.generate_synthetic_image",
        "Stub: generate optical image from truth, attitude, camera, PSF, and photometry models.",
        _schema_object({"scene_config": {"type": "object"}}),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "SI plus detector units as declared by input",
        "Requires explicit camera and truth frames",
        "Requires explicit exposure epoch",
        "P1",
        validation_rules=["Do not claim image realism until camera, PSF, and SNR models are implemented."],
        error_codes=["NOT_IMPLEMENTED"],
    ),
    _spec(
        "space.cross_validate_orbit",
        "Stub: run the same propagation through an independent engine and compare residuals.",
        _schema_object({"task_spec": {"type": "object"}, "engines": {"type": "array", "items": {"type": "string"}}}),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "SI: m, m/s, s",
        "Declared by task_spec",
        "Declared by task_spec",
        "P1",
        validation_rules=["At least two independent engines or references are required."],
        error_codes=["NOT_IMPLEMENTED", "ENGINE_UNAVAILABLE"],
    ),
    _spec(
        "space.optimize_trajectory",
        "Stub: optimize maneuvers or trajectory objectives with bounded constraints.",
        _schema_object({"problem": {"type": "object"}}),
        _schema_object({"status": {"type": "string"}, "reason": {"type": "string"}}),
        "Declared by problem",
        "Declared by problem",
        "Declared by problem",
        "P2",
        validation_rules=["No optimality claim without objective, constraints, and solver report."],
        error_codes=["NOT_IMPLEMENTED"],
    ),
]


def get_space_tool_specs() -> List[Dict[str, Any]]:
    return [dict(spec) for spec in SPACE_TOOL_SPECS]


def get_space_tool_definitions() -> List[Dict[str, Any]]:
    definitions: List[Dict[str, Any]] = []
    for spec in SPACE_TOOL_SPECS:
        definitions.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "inputSchema": spec["input_schema"],
                "outputSchema": spec["output_schema"],
                "units": spec["units"],
                "frame": spec["frame"],
                "time_system": spec["time_system"],
                "risk_level": spec["risk_level"],
                "required_files": spec["required_files"],
                "generated_artifacts": spec["generated_artifacts"],
                "validation_rules": spec["validation_rules"],
                "error_codes": spec["error_codes"],
            }
        )
    return definitions


def _space_check_environment(**kwargs: Any) -> Dict[str, Any]:
    return check_engine_availability(**kwargs)


def _space_convert_time(**kwargs: Any) -> Dict[str, Any]:
    return convert_time(**kwargs)


def _space_propagate_orbit(**kwargs: Any) -> Dict[str, Any]:
    result = propagate_orbit(**kwargs)
    if isinstance(result, dict) and result.get("status") != "error":
        result.setdefault("metadata", {})
        result["metadata"]["tool"] = "space.propagate_orbit"
    return result


def _space_plot_orbit(state_history: List[Dict[str, Any]], output_path: str, **_: Any) -> Dict[str, Any]:
    try:
        from aerospace_agent.experiment_runtime import plot_orbit_png

        path = plot_orbit_png(state_history, output_path)
        return {"status": "completed", "artifact_path": str(path)}
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "error_code": "PLOT_FAILED"}


def _space_build_report(run_dir: str, result: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    try:
        from aerospace_agent.experiment_runtime import write_experiment_report

        report_path = write_experiment_report(run_dir, result)
        return {"status": "completed", "report_path": str(report_path)}
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "error_code": "REPORT_FAILED"}


def _space_reproduce_run(run_dir: str, task: str | None = None, config_name: str = "config.yaml", **_: Any) -> Dict[str, Any]:
    try:
        from aerospace_agent.experiment_runtime import build_reproduce_script

        config_path = Path(run_dir) / config_name
        if not config_path.is_file():
            return {
                "status": "error",
                "reason": f"Missing required config file: {config_path}",
                "error_code": "REPRODUCE_SCRIPT_FAILED",
            }
        path = build_reproduce_script(run_dir, task=task, config_name=config_name)
        return {"status": "completed", "reproduce_path": str(path)}
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "error_code": "REPRODUCE_SCRIPT_FAILED"}


def _stub(tool_name: str) -> Callable[..., Dict[str, Any]]:
    def caller(**kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "tool": tool_name,
            "reason": "Schema is registered, but implementation is not part of the minimal closed loop.",
            "received_keys": sorted(kwargs),
            "error_code": "NOT_IMPLEMENTED",
        }

    caller.__name__ = tool_name.replace(".", "_")
    return caller


SPACE_TOOL_CALLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "space.check_environment": _space_check_environment,
    "space.convert_time": _space_convert_time,
    "space.propagate_orbit": _space_propagate_orbit,
    "space.plot_orbit": _space_plot_orbit,
    "space.build_report": _space_build_report,
    "space.reproduce_run": _space_reproduce_run,
    "space.load_spice_kernels": _stub("space.load_spice_kernels"),
    "space.transform_frame": _stub("space.transform_frame"),
    "space.compute_attitude": _stub("space.compute_attitude"),
    "space.project_to_sensor": _stub("space.project_to_sensor"),
    "space.query_star_catalog": _stub("space.query_star_catalog"),
    "space.render_starfield": _stub("space.render_starfield"),
    "space.compute_snr": _stub("space.compute_snr"),
    "space.validate_truth": _stub("space.validate_truth"),
    "space.generate_synthetic_image": _stub("space.generate_synthetic_image"),
    "space.cross_validate_orbit": _stub("space.cross_validate_orbit"),
    "space.optimize_trajectory": _stub("space.optimize_trajectory"),
}


def register_space_tools(registry: MutableMapping[str, Callable[..., Dict[str, Any]]]) -> None:
    registry.update(SPACE_TOOL_CALLS)


__all__ = [
    "SPACE_TOOL_SPECS",
    "get_space_tool_specs",
    "get_space_tool_definitions",
    "register_space_tools",
]
