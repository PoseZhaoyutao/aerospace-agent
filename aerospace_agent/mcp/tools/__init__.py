"""tools -- MCP tools unified registration and export.

First principles (K2 whitelist encapsulation):
  1. LLM can only call tools registered in TOOL_REGISTRY -- cannot directly call underlying libraries
  2. Each tool returns a JSON-serializable dict
  3. All failures return structured {status:"error", reason:...} -- never silent failure
  4. get_tool_definitions() generates JSON Schema for MCP protocol
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from .core_tool_adapters import CoreToolAdapters

# Environment tools
from .environment_tools import check_engine_availability, index_reference_demos
# Workflow tools
from .workflow_tools import (
    search_workflows, generate_astrodynamics_workflow, list_workflow_templates,
)
# Time tools
from .time_tools import convert_time
# Frame tools
from .frame_tools import transform_frame
# Ephemeris tools
from .ephemeris_tools import query_ephemeris_state
# Propagation tools
from .propagation_tools import convert_orbit_representation, propagate_orbit
# Access tools
from .access_tools import compute_ground_access
# GMAT tools
from .gmat_tools import run_gmat_script
# Validation tools
from .validation_tools import cross_validate_results
# SPICE tools
from .spice_tools import (
    spice_query_ephemeris, spice_transform_frame, spice_convert_time,
    spice_load_kernels, spice_list_loaded_kernels,
    spice_compute_observation_geometry, spice_compute_occultation,
    spice_two_line_elements_to_state,
)
# Orekit tools
from .orekit_tools import (
    orekit_propagate_orbit,
    orekit_transform_frame,
    orekit_convert_time,
    orekit_spherical_harmonics,
    orekit_keplerian_to_cartesian,
    orekit_cartesian_to_keplerian,
    orekit_compute_eclipse_times,
    orekit_compute_maneuver,
    orekit_tle_propagation,
    orekit_event_detection,
)
from .space_tools import (
    SPACE_TOOL_SPECS,
    get_space_tool_definitions,
    get_space_tool_specs,
    register_space_tools,
)
# Basilisk tools
from .basilisk_tools import (
    basilisk_propagate_orbit,
    basilisk_attitude_control,
    basilisk_orbit_elements_conversion,
    basilisk_thruster_modeling,
    basilisk_reaction_wheel_modeling,
    basilisk_sun_pointing,
    basilisk_nadir_pointing,
    basilisk_eclipse_detection,
    basilisk_atmospheric_drag,
    basilisk_solar_radiation_pressure,
)

#: Tool registry -- name -> callable
TOOL_REGISTRY: Dict[str, Callable[..., Dict]] = {
    "check_engine_availability": check_engine_availability,
    "index_reference_demos": index_reference_demos,
    "search_workflows": search_workflows,
    "generate_astrodynamics_workflow": generate_astrodynamics_workflow,
    "list_workflow_templates": list_workflow_templates,
    "convert_time": convert_time,
    "transform_frame": transform_frame,
    "query_ephemeris_state": query_ephemeris_state,
    "convert_orbit_representation": convert_orbit_representation,
    "propagate_orbit": propagate_orbit,
    "compute_ground_access": compute_ground_access,
    "run_gmat_script": run_gmat_script,
    "cross_validate_results": cross_validate_results,
    # SPICE tools (8)
    "spice_query_ephemeris": spice_query_ephemeris,
    "spice_transform_frame": spice_transform_frame,
    "spice_convert_time": spice_convert_time,
    "spice_load_kernels": spice_load_kernels,
    "spice_list_loaded_kernels": spice_list_loaded_kernels,
    "spice_compute_observation_geometry": spice_compute_observation_geometry,
    "spice_compute_occultation": spice_compute_occultation,
    "spice_two_line_elements_to_state": spice_two_line_elements_to_state,
    # Basilisk tools (10)
    "basilisk_propagate_orbit": basilisk_propagate_orbit,
    "basilisk_attitude_control": basilisk_attitude_control,
    "basilisk_orbit_elements_conversion": basilisk_orbit_elements_conversion,
    "basilisk_thruster_modeling": basilisk_thruster_modeling,
    "basilisk_reaction_wheel_modeling": basilisk_reaction_wheel_modeling,
    "basilisk_sun_pointing": basilisk_sun_pointing,
    "basilisk_nadir_pointing": basilisk_nadir_pointing,
    "basilisk_eclipse_detection": basilisk_eclipse_detection,
    "basilisk_atmospheric_drag": basilisk_atmospheric_drag,
    "basilisk_solar_radiation_pressure": basilisk_solar_radiation_pressure,
    # Orekit tools (10)
    "orekit_propagate_orbit": orekit_propagate_orbit,
    "orekit_transform_frame": orekit_transform_frame,
    "orekit_convert_time": orekit_convert_time,
    "orekit_spherical_harmonics": orekit_spherical_harmonics,
    "orekit_keplerian_to_cartesian": orekit_keplerian_to_cartesian,
    "orekit_cartesian_to_keplerian": orekit_cartesian_to_keplerian,
    "orekit_compute_eclipse_times": orekit_compute_eclipse_times,
    "orekit_compute_maneuver": orekit_compute_maneuver,
    "orekit_tle_propagation": orekit_tle_propagation,
    "orekit_event_detection": orekit_event_detection,
}

register_space_tools(TOOL_REGISTRY)

#: 12 core MCP tool names (excluding list_workflow_templates helper)
CORE_TOOLS: List[str] = [
    "check_engine_availability",
    "index_reference_demos",
    "search_workflows",
    "generate_astrodynamics_workflow",
    "convert_time",
    "transform_frame",
    "query_ephemeris_state",
    "convert_orbit_representation",
    "propagate_orbit",
    "compute_ground_access",
    "run_gmat_script",
    "cross_validate_results",
    # SPICE tools
    "spice_query_ephemeris",
    "spice_transform_frame",
    "spice_convert_time",
    "spice_load_kernels",
    "spice_list_loaded_kernels",
    "spice_compute_observation_geometry",
    "spice_compute_occultation",
    "spice_two_line_elements_to_state",
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return MCP protocol format tool definitions (JSON Schema).

    Only definitions implemented by this repository are returned.  Optional
    integrations must register through an explicit, validated manifest.
    """

    # Core astrodynamics tool definitions (hardcoded)
    core_defs = [
        {
            "name": "check_engine_availability",
            "description": "Check availability, version, capabilities, data paths and license status for all or specified engines.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "engines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Engine name list; omit to check all 7",
                    },
                },
            },
        },
        {
            "name": "index_reference_demos",
            "description": "Scan configured paths to index examples/tests/tutorials for each engine (read-only).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "scan_paths": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        {
            "name": "search_workflows",
            "description": "Search workflow directory and return matching candidate workflows.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "task_type": {"type": "string"},
                    "preferred_engine": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "generate_astrodynamics_workflow",
            "description": "Generate a complete WorkflowSpec (goal/steps/outputs/validation) from user requirements.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_requirement": {"type": "string"},
                    "candidate_workflow_id": {"type": "string"},
                    "constraints": {"type": "object"},
                },
                "required": ["user_requirement"],
            },
        },
        {
            "name": "convert_time",
            "description": "Cross timescale (UTC/TAI/TT/TDB/ET) and format (ISO/JD/MJD/UNIX) conversion.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "value": {"type": ["string", "number"]},
                    "from_scale": {"type": "string"},
                    "from_format": {"type": "string"},
                    "to_scale": {"type": "string"},
                    "to_format": {"type": "string"},
                },
                "required": ["value"],
            },
        },
        {
            "name": "transform_frame",
            "description": "Transform orbit state to target coordinate system (GCRF/ICRF/EME2000/J2000/ITRF/TEME/BodyFixed).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state_dict": {"type": "object"},
                    "target_frame": {"type": "string"},
                    "target_center": {"type": "string"},
                },
                "required": ["state_dict", "target_frame"],
            },
        },
        {
            "name": "query_ephemeris_state",
            "description": "Query target body position and velocity relative to observer via SPICE kernels.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "observer": {"type": "string"},
                    "epoch_dict": {"type": "object"},
                    "frame": {"type": "string"},
                    "aberration_correction": {"type": "string"},
                    "kernels": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["target", "observer", "epoch_dict"],
            },
        },
        {
            "name": "convert_orbit_representation",
            "description": "Orbit state representation conversion (Cartesian <-> Keplerian), explicitly marking mu and units.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state_dict": {"type": "object"},
                    "target_representation": {"type": "string"},
                    "mu": {"type": "number"},
                },
                "required": ["state_dict", "target_representation"],
            },
        },
        {
            "name": "propagate_orbit",
            "description": "Orbit propagation (two-body + J2 placeholder), supports auto/poliastro/orekit/gmat engines.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "initial_state_dict": {"type": "object"},
                    "force_model_dict": {"type": "object"},
                    "duration_s": {"type": "number"},
                    "output_step_s": {"type": "number"},
                    "engine": {"type": "string"},
                },
                "required": ["initial_state_dict", "force_model_dict",
                             "duration_s"],
            },
        },
        {
            "name": "compute_ground_access",
            "description": "Compute satellite-to-ground-station visibility time windows.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "orbit_state_dict": {"type": "object"},
                    "ground_station_dict": {"type": "object"},
                    "start_epoch_dict": {"type": "object"},
                    "stop_epoch_dict": {"type": "object"},
                    "min_elevation_deg": {"type": "number"},
                },
                "required": ["orbit_state_dict", "ground_station_dict",
                             "start_epoch_dict", "stop_epoch_dict"],
            },
        },
        {
            "name": "run_gmat_script",
            "description": "Run a GMAT script in a sandbox workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "script_text": {"type": "string"},
                    "script_path": {"type": "string"},
                    "workspace": {"type": "string"},
                },
            },
        },
        {
            "name": "cross_validate_results",
            "description": "Multi-engine cross-validation -- compare results of the same task across different engines.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_spec": {"type": "object"},
                    "engines": {"type": "array", "items": {"type": "string"}},
                    "existing_results": {"type": "object"},
                },
                "required": ["task_spec"],
            },
        },
        # ---- SPICE tools (8) ----
        {
            "name": "spice_query_ephemeris",
            "description": "Query target body position/velocity relative to observer using SPICE spkezr (SI units: m, m/s). Requires spiceypy and SPICE kernels.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Target body name or NAIF ID (e.g. Moon/399/MARS)"},
                    "observer": {"type": "string", "description": "Observer body name or NAIF ID (e.g. Earth/10/SUN)"},
                    "epoch_dict": {"type": "object", "description": "Time dict {value, scale, format} e.g. {value:'2026-01-01T00:00:00', scale:'UTC', format:'ISO'}"},
                    "frame": {"type": "string", "description": "Reference frame (J2000/ICRF/GCRF/ITRF93), default J2000"},
                    "aberration_correction": {"type": "string", "description": "Aberration correction (NONE/LT/LT+S), default NONE"},
                    "kernels": {"type": "array", "items": {"type": "string"}, "description": "Optional kernel path list"},
                },
                "required": ["target", "observer", "epoch_dict"],
            },
        },
        {
            "name": "spice_transform_frame",
            "description": "Transform orbit state to target frame using SPICE sxform (6x6 state rotation matrix, includes velocity cross terms).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state_dict": {"type": "object", "description": "OrbitState format with position_m/velocity_mps/epoch/frame"},
                    "target_frame": {"type": "string", "description": "Target frame name (J2000/ICRF/GCRF/ITRF93/IAU_EARTH etc.)"},
                },
                "required": ["state_dict", "target_frame"],
            },
        },
        {
            "name": "spice_convert_time",
            "description": "Convert between UTC/TAI/TDB/ET time scales using SPICE str2et/et2utc/unitim.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "epoch_dict": {"type": "object", "description": "Time dict {value, scale, format} e.g. {value:'2026-01-01T00:00:00', scale:'UTC', format:'ISO'}"},
                    "target_scale": {"type": "string", "description": "Target time scale (UTC/TAI/TDB/ET)"},
                },
                "required": ["epoch_dict", "target_scale"],
            },
        },
        {
            "name": "spice_load_kernels",
            "description": "Load SPICE kernel files (.tls/.bsp/.pck/.tf etc.). Loaded kernels are available for all subsequent ephemeris/frame/geometry queries.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kernel_paths": {"type": "array", "items": {"type": "string"}, "description": "Kernel file path list"},
                },
                "required": ["kernel_paths"],
            },
        },
        {
            "name": "spice_list_loaded_kernels",
            "description": "List all currently loaded SPICE kernel files with filename, type, and source.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "spice_compute_observation_geometry",
            "description": "Compute observation geometry: illumination angles (phase/solar incidence/emission), distances, sub-observer/sub-solar lat/lon. Uses SPICE ilumin/subpnt/subslr. Requires PCK ellipsoid models.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Target body name (e.g. MOON/MARS)"},
                    "observer": {"type": "string", "description": "Observer body name (e.g. EARTH/SUN)"},
                    "epoch_dict": {"type": "object", "description": "Time dict {value, scale, format}"},
                    "frame": {"type": "string", "description": "Reference frame, default J2000"},
                },
                "required": ["target", "observer", "epoch_dict"],
            },
        },
        {
            "name": "spice_compute_occultation",
            "description": "Compute occultation: check if target is occulted by another body. Uses angular separation comparison of apparent disks. Returns occultation_code (0=none/1=partial/2=full). Requires PCK ellipsoid models.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Occulted body name (e.g. MOON/MARS/IO)"},
                    "occulting_body": {"type": "string", "description": "Occulting body name (e.g. EARTH/JUPITER)"},
                    "observer": {"type": "string", "description": "Observer body name (e.g. SUN/EARTH)"},
                    "epoch_dict": {"type": "object", "description": "Time dict {value, scale, format}"},
                    "frame": {"type": "string", "description": "Reference frame, default J2000"},
                },
                "required": ["target", "occulting_body", "observer", "epoch_dict"],
            },
        },
        {
            "name": "spice_two_line_elements_to_state",
            "description": "TLE (Two-Line Elements) to orbital state conversion. Parses TLE format, extracts Keplerian elements, solves Kepler equation to get Cartesian position/velocity (SI units). Two-body model, no SGP4 secular terms. Accuracy ~km level.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "line1": {"type": "string", "description": "TLE line 1"},
                    "line2": {"type": "string", "description": "TLE line 2"},
                    "frame": {"type": "string", "description": "Target frame, default J2000"},
                },
                "required": ["line1", "line2"],
            },
        },
    ]

    # Unavailable SpaceBasic descriptors remain discoverable through
    # get_space_tool_definitions(), but are not MCP execution candidates.
    core_defs.extend(
        item for item in get_space_tool_definitions()
        if item.get("status") == "available"
    )

    # Basilisk tool definitions (10)
    core_defs.extend([
        {
            "name": "basilisk_propagate_orbit",
            "description": "Orbit propagation using Basilisk engine (two-body + optional J2), via BSIL simulation task modules.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "initial_state_dict": {"type": "object", "description": "Initial OrbitState dict with position_m, velocity_mps, epoch, frame"},
                    "force_model_dict": {"type": "object", "description": "ForceModel dict"},
                    "duration_s": {"type": "number", "description": "Propagation duration (seconds)"},
                    "output_step_s": {"type": "number", "description": "Output sampling interval (seconds), None for final state only"},
                    "engine": {"type": "string", "description": "Engine name, fixed as basilisk"},
                },
                "required": ["initial_state_dict", "force_model_dict", "duration_s"],
            },
        },
        {
            "name": "basilisk_attitude_control",
            "description": "Attitude control simulation using Basilisk engine (MRP feedback), via BSIL FSW modules.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "initial_attitude_dict": {"type": "object", "description": "Initial AttitudeState dict with quaternion, angular_velocity_radps"},
                    "controller": {"type": "string", "description": "Controller type, default MRP_feedback"},
                    "K": {"type": "number", "description": "Proportional gain, default 3.5"},
                    "P": {"type": "number", "description": "Derivative gain, default 35.0"},
                    "duration_s": {"type": "number", "description": "Simulation duration (seconds), default 600"},
                    "step_s": {"type": "number", "description": "Integration step (seconds), default 0.1"},
                    "output_step_s": {"type": "number", "description": "Output sampling interval (seconds)"},
                },
                "required": ["initial_attitude_dict"],
            },
        },
        {
            "name": "basilisk_orbit_elements_conversion",
            "description": "Orbit elements conversion: Cartesian <-> Keplerian <-> Equinoctial, pure numpy implementation, no Basilisk simulation dependency.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_representation": {"type": "string", "description": "Source representation: cartesian/keplerian/equinoctial"},
                    "source_data": {"type": "object", "description": "Source data dict"},
                    "target_representation": {"type": "string", "description": "Target representation: cartesian/keplerian/equinoctial"},
                    "mu": {"type": "number", "description": "Gravitational parameter m^3/s^2 (default Earth 3.986004418e14)"},
                },
                "required": ["source_representation", "source_data", "target_representation"],
            },
        },
        {
            "name": "basilisk_thruster_modeling",
            "description": "Thruster modeling: compute thrust vector, specific impulse, mass change, delta-v via rocket equation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "thrust_N": {"type": "number", "description": "Thrust magnitude (N)"},
                    "isp_s": {"type": "number", "description": "Specific impulse (seconds)"},
                    "spacecraft_mass_kg": {"type": "number", "description": "Initial spacecraft mass (kg)"},
                    "thrust_direction_body": {"type": "array", "items": {"type": "number"}, "description": "Thrust direction in body frame [x,y,z], default +x"},
                    "thrust_position_body": {"type": "array", "items": {"type": "number"}, "description": "Thruster mounting position [x,y,z] m, default origin"},
                    "burn_duration_s": {"type": "number", "description": "Burn duration (seconds), default 1.0"},
                    "g0": {"type": "number", "description": "Sea-level gravity acceleration m/s^2, default 9.80665"},
                },
                "required": ["thrust_N", "isp_s", "spacecraft_mass_kg"],
            },
        },
        {
            "name": "basilisk_reaction_wheel_modeling",
            "description": "Reaction wheel modeling: angular momentum storage, saturation detection, torque allocation (pseudo-inverse).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "wheel_speeds_rpm": {"type": "array", "items": {"type": "number"}, "description": "Current wheel speeds RPM"},
                    "wheel_inertia_kgm2": {"type": "number", "description": "Single wheel moment of inertia kg*m^2, default 0.001"},
                    "max_rpm": {"type": "number", "description": "Maximum RPM, default 6000"},
                    "wheel_axes": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Wheel axis directions Nx3, default 4-wheel pyramid"},
                    "torque_cmd_Nm": {"type": "array", "items": {"type": "number"}, "description": "Desired torque [Tx,Ty,Tz] N*m"},
                },
                "required": ["wheel_speeds_rpm"],
            },
        },
        {
            "name": "basilisk_sun_pointing",
            "description": "Sun pointing control: compute sun direction vector + desired attitude quaternion, Meeus 1998 simplified sun position.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spacecraft_position_m": {"type": "array", "items": {"type": "number"}, "description": "Spacecraft geocentric position [x,y,z] m"},
                    "epoch_iso": {"type": "string", "description": "ISO 8601 time string"},
                    "body_x_axis": {"type": "array", "items": {"type": "number"}, "description": "Body +x axis direction in ECI [x,y,z], default [1,0,0]"},
                },
                "required": ["spacecraft_position_m"],
            },
        },
        {
            "name": "basilisk_nadir_pointing",
            "description": "Nadir pointing control: compute nadir direction vector + desired attitude quaternion, LVLH reference frame.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spacecraft_position_m": {"type": "array", "items": {"type": "number"}, "description": "Spacecraft geocentric position [x,y,z] m"},
                    "spacecraft_velocity_mps": {"type": "array", "items": {"type": "number"}, "description": "Spacecraft geocentric velocity [vx,vy,vz] m/s"},
                },
                "required": ["spacecraft_position_m"],
            },
        },
        {
            "name": "basilisk_eclipse_detection",
            "description": "Eclipse detection: determine if spacecraft is in Earth shadow (umbra/penumbra/sunlight), supports cylindrical/conical shadow models.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spacecraft_position_m": {"type": "array", "items": {"type": "number"}, "description": "Spacecraft geocentric position [x,y,z] m"},
                    "sun_position_m": {"type": "array", "items": {"type": "number"}, "description": "Sun geocentric position [x,y,z] m"},
                    "epoch_iso": {"type": "string", "description": "ISO 8601 time string"},
                    "shadow_model": {"type": "string", "description": "Shadow model: cylindrical/conical, default cylindrical"},
                },
                "required": ["spacecraft_position_m"],
            },
        },
        {
            "name": "basilisk_atmospheric_drag",
            "description": "Atmospheric drag model: exponential atmosphere or simplified NRLMSISE-00, compute atmospheric density and drag acceleration.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spacecraft_position_m": {"type": "array", "items": {"type": "number"}, "description": "Spacecraft geocentric position [x,y,z] m"},
                    "spacecraft_velocity_mps": {"type": "array", "items": {"type": "number"}, "description": "Spacecraft velocity [vx,vy,vz] m/s"},
                    "drag_coefficient": {"type": "number", "description": "Drag coefficient Cd, default 2.2"},
                    "area_m2": {"type": "number", "description": "Cross-sectional area m^2, default 1.0"},
                    "spacecraft_mass_kg": {"type": "number", "description": "Spacecraft mass kg, default 100"},
                    "atmosphere_model": {"type": "string", "description": "Atmosphere model: exponential/nrlmsise_simplified, default exponential"},
                    "F10_7": {"type": "number", "description": "10.7cm solar radio flux sfu, default 150"},
                    "Ap": {"type": "number", "description": "Geomagnetic index, default 15"},
                },
                "required": ["spacecraft_position_m"],
            },
        },
        {
            "name": "basilisk_solar_radiation_pressure",
            "description": "Solar radiation pressure model: cannonball spherical model + cylindrical shadow, compute SRP acceleration and eclipse factor.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spacecraft_position_m": {"type": "array", "items": {"type": "number"}, "description": "Spacecraft geocentric position [x,y,z] m"},
                    "spacecraft_mass_kg": {"type": "number", "description": "Spacecraft mass kg, default 100"},
                    "area_m2": {"type": "number", "description": "Sun-facing area m^2, default 1.0"},
                    "reflectivity_coefficient": {"type": "number", "description": "Reflectivity (0=full absorption, 1=full specular reflection), default 0.2"},
                    "sun_position_m": {"type": "array", "items": {"type": "number"}, "description": "Sun geocentric position [x,y,z] m"},
                    "epoch_iso": {"type": "string", "description": "ISO 8601 time string"},
                    "shadow_model": {"type": "string", "description": "Shadow model: cylindrical/none, default cylindrical"},
                },
                "required": ["spacecraft_position_m"],
            },
        },
    ])

    return core_defs


__all__ = [
    # Tool functions
    "check_engine_availability", "index_reference_demos",
    "search_workflows", "generate_astrodynamics_workflow",
    "list_workflow_templates",
    "convert_time", "transform_frame", "query_ephemeris_state",
    "convert_orbit_representation", "propagate_orbit",
    "compute_ground_access", "run_gmat_script",
    "cross_validate_results",
    # SPICE tools (8)
    "spice_query_ephemeris", "spice_transform_frame",
    "spice_convert_time", "spice_load_kernels",
    "spice_list_loaded_kernels", "spice_compute_observation_geometry",
    "spice_compute_occultation", "spice_two_line_elements_to_state",
    # Orekit tools
    "orekit_propagate_orbit", "orekit_transform_frame",
    "orekit_convert_time", "orekit_spherical_harmonics",
    "orekit_keplerian_to_cartesian", "orekit_cartesian_to_keplerian",
    "orekit_compute_eclipse_times", "orekit_compute_maneuver",
    "orekit_tle_propagation", "orekit_event_detection",
    # Basilisk tools
    "basilisk_propagate_orbit", "basilisk_attitude_control",
    "basilisk_orbit_elements_conversion", "basilisk_thruster_modeling",
    "basilisk_reaction_wheel_modeling", "basilisk_sun_pointing",
    "basilisk_nadir_pointing", "basilisk_eclipse_detection",
    "basilisk_atmospheric_drag", "basilisk_solar_radiation_pressure",
    # Registry
    "TOOL_REGISTRY", "CORE_TOOLS", "get_tool_definitions",
    "SPACE_TOOL_SPECS", "get_space_tool_specs",
    "get_space_tool_definitions", "register_space_tools",
    "CoreToolAdapters",
]
