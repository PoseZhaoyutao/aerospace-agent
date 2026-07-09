"""Seed corpus for an orbit-dynamics expert RAG library."""

from __future__ import annotations

from typing import Any, Dict, List


ORBIT_DYNAMICS_SEED_DOCUMENTS: List[Dict[str, str]] = [
    {
        "topic": "two_body_dynamics",
        "title": "Two-body orbital dynamics",
        "text": (
            "Two-body dynamics models spacecraft motion under a central gravity field. "
            "The governing acceleration is -mu r / |r|^3. Keplerian elements are useful "
            "for compact orbit description, while Cartesian states are preferred for "
            "numerical propagation and covariance operations. Assumptions: point-mass "
            "gravity, no drag, no third-body perturbation, no finite burn."
        ),
    },
    {
        "topic": "frames_and_time",
        "title": "Reference frames and time scales",
        "text": (
            "Orbit software must state frame and time scale explicitly. Common frames "
            "include ECI/GCRF, ECEF/ITRF, TEME, LVLH, and sensor frames. Common time "
            "scales include UTC, TAI, TT, and TDB. Frame transforms and time conversions "
            "are not optional metadata; they affect position, velocity, pointing, and "
            "truth/image alignment."
        ),
    },
    {
        "topic": "perturbations",
        "title": "Perturbed orbit propagation",
        "text": (
            "High-fidelity orbit propagation should declare force models: gravity degree "
            "and order, J2/J3 terms, atmospheric drag, solar radiation pressure, third-body "
            "gravity, solid tides, maneuvers, and integrator tolerances. Unsupported force "
            "models must be reported as unavailable rather than silently mocked."
        ),
    },
    {
        "topic": "orbit_determination",
        "title": "Orbit determination and measurements",
        "text": (
            "Orbit determination estimates state and uncertainty from observations such "
            "as range, range-rate, optical angles, bearings, or image detections. A useful "
            "RAG answer should separate measurement model, dynamic model, estimator, "
            "prior assumptions, residuals, and validation data."
        ),
    },
    {
        "topic": "validation",
        "title": "Propagation validation",
        "text": (
            "Orbit propagation validation should compare independent engines or analytic "
            "limits when possible. Minimal checks include units, epoch consistency, frame "
            "consistency, conserved energy for two-body cases, expected nodal precession "
            "for J2 cases, and bounded interpolation error."
        ),
    },
    {
        "topic": "sensor_truth_mapping",
        "title": "Truth to sensor mapping",
        "text": (
            "Space-based image simulation must preserve traceability from propagated truth "
            "states to camera-frame line-of-sight vectors, detector coordinates, PSF, SNR, "
            "exposure, gain, noise model, and generated image pixels. Claims about weak "
            "target detectability require explicit photometric assumptions."
        ),
    },
]


def build_orbit_dynamics_seed_texts() -> List[str]:
    return [
        f"[{doc['topic']}] {doc['title']}\n{doc['text']}"
        for doc in ORBIT_DYNAMICS_SEED_DOCUMENTS
    ]


def index_orbit_dynamics_corpus(rag: Any) -> Dict[str, Any]:
    """Index the orbit-dynamics seed corpus into a RAG object."""

    if rag is None or not hasattr(rag, "index"):
        return {
            "status": "unavailable",
            "error_code": "RAG_NOT_AVAILABLE",
            "indexed_count": 0,
            "topics": [doc["topic"] for doc in ORBIT_DYNAMICS_SEED_DOCUMENTS],
        }

    indexed = 0
    for doc in ORBIT_DYNAMICS_SEED_DOCUMENTS:
        text = f"[{doc['topic']}] {doc['title']}\n{doc['text']}"
        try:
            rag.index(text, source=f"orbit_dynamics:{doc['topic']}")
            indexed += 1
        except TypeError:
            rag.index(text)
            indexed += 1

    return {
        "status": "ok",
        "indexed_count": indexed,
        "topics": [doc["topic"] for doc in ORBIT_DYNAMICS_SEED_DOCUMENTS],
    }
