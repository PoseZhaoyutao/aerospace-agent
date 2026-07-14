"""Aerospace domain discovery descriptors.

All six domains are deliberately ``interface_only`` until a separately
verified implementation is approved.
"""

from .control_planning import DESCRIPTOR as CONTROL_PLANNING
from .fault_diagnosis_maintenance import DESCRIPTOR as FAULT_DIAGNOSIS_MAINTENANCE
from .mechanical_thermal_electrical import DESCRIPTOR as MECHANICAL_THERMAL_ELECTRICAL
from .navigation_orbit_determination import DESCRIPTOR as NAVIGATION_ORBIT_DETERMINATION
from .orbit_design import DESCRIPTOR as ORBIT_DESIGN
from .simulation import DESCRIPTOR as SIMULATION


DOMAIN_DESCRIPTORS = (
    SIMULATION,
    NAVIGATION_ORBIT_DETERMINATION,
    CONTROL_PLANNING,
    ORBIT_DESIGN,
    MECHANICAL_THERMAL_ELECTRICAL,
    FAULT_DIAGNOSIS_MAINTENANCE,
)


__all__ = [
    "CONTROL_PLANNING",
    "DOMAIN_DESCRIPTORS",
    "FAULT_DIAGNOSIS_MAINTENANCE",
    "MECHANICAL_THERMAL_ELECTRICAL",
    "NAVIGATION_ORBIT_DETERMINATION",
    "ORBIT_DESIGN",
    "SIMULATION",
]

