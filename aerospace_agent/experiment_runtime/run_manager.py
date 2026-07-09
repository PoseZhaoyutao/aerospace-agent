"""Run manager for minimal evidence-backed orbit experiments."""

from __future__ import annotations

import csv
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import yaml

from aerospace_agent.mcp.tools.propagation_tools import propagate_orbit

from .invariants import check_aerospace_invariants


MU_EARTH_M3_S2 = 3.986004418e14
DEFAULT_RADIUS_M = 6_778_137.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{stamp}"


@dataclass
class LedgerWriter:
    path: Path | str
    run_id: str

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, payload: Mapping[str, Any] | None = None, status: str = "ok") -> Dict[str, Any]:
        entry = {
            "timestamp_utc": _utc_now_iso(),
            "run_id": self.run_id,
            "event": event,
            "status": status,
            "payload": dict(payload or {}),
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        return entry


def _deep_merge(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def default_orbit_experiment_config(
    task: str,
    overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    velocity_mps = math.sqrt(MU_EARTH_M3_S2 / DEFAULT_RADIUS_M)
    config: Dict[str, Any] = {
        "task": task,
        "run_mode": "minimal_closed_loop",
        "model": {
            "name": "builtin_two_body",
            "status": "assumption_not_cross_validated",
            "mu_m3_s2": MU_EARTH_M3_S2,
        },
        "duration_s": 86_400.0,
        "output_step_s": 900.0,
        "engine": "builtin",
        "units": {
            "position": "m",
            "velocity": "m/s",
            "duration": "s",
            "mu": "m^3/s^2",
        },
        "initial_state": {
            "epoch": {
                "value": "2026-01-01T00:00:00",
                "scale": "UTC",
                "format": "ISO",
            },
            "frame": {
                "name": "GCRF",
                "center": "Earth",
                "realization": "IERS2010",
            },
            "representation": "cartesian",
            "position_m": [DEFAULT_RADIUS_M, 0.0, 0.0],
            "velocity_mps": [0.0, velocity_mps, 0.0],
        },
        "force_model": {
            "gravity": "point_mass",
            "drag": {"enabled": False},
            "srp": {"enabled": False},
            "third_body": [],
            "relativity": False,
        },
    }
    if overrides:
        _deep_merge(config, dict(overrides))
    config["task"] = str(config.get("task") or task)
    return config


def load_config_file(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(data), fh, allow_unicode=True, sort_keys=False)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _state_rows(state_history: Iterable[Mapping[str, Any]]) -> Iterable[Dict[str, Any]]:
    for item in state_history:
        position = item.get("position_m") or [None, None, None]
        velocity = item.get("velocity_mps") or [None, None, None]
        epoch = item.get("epoch") or {}
        frame = item.get("frame") or {}
        yield {
            "elapsed_s": item.get("elapsed_s"),
            "epoch_value": epoch.get("value"),
            "time_scale": epoch.get("scale"),
            "frame": frame.get("name"),
            "x_m": position[0],
            "y_m": position[1],
            "z_m": position[2],
            "vx_mps": velocity[0],
            "vy_mps": velocity[1],
            "vz_mps": velocity[2],
        }


def _write_state_history_csv(path: Path, state_history: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(_state_rows(state_history))
    fieldnames = [
        "elapsed_s",
        "epoch_value",
        "time_scale",
        "frame",
        "x_m",
        "y_m",
        "z_m",
        "vx_mps",
        "vy_mps",
        "vz_mps",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _history_xyz_km(state_history: Iterable[Mapping[str, Any]]) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for item in state_history:
        position = item.get("position_m") or []
        if len(position) >= 3:
            xs.append(float(position[0]) / 1000.0)
            ys.append(float(position[1]) / 1000.0)
            zs.append(float(position[2]) / 1000.0)
    return xs, ys, zs


def plot_orbit_png(state_history: List[Mapping[str, Any]], output_path: str | os.PathLike[str]) -> Path:
    """Generate an orbit plot. Matplotlib is preferred; a small PNG fallback is used."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xs, ys, zs = _history_xyz_km(state_history)
    if not xs:
        raise ValueError("state_history contains no cartesian positions")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, color="#1f77b4", linewidth=1.6)
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color="#2ca02c", s=28, label="start")
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="#d62728", s=28, label="end")
        ax.set_xlabel("x km")
        ax.set_ylabel("y km")
        ax.set_zlabel("z km")
        ax.set_title("Builtin two-body orbit propagation")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path
    except Exception:
        _write_simple_png_plot(path, xs, ys)
        return path


def _write_simple_png_plot(path: Path, xs: list[float], ys: list[float], size: int = 512) -> None:
    margin = 36
    width = height = size
    pixels = bytearray([255] * width * height * 3)

    def put_pixel(px: int, py: int, color: tuple[int, int, int]) -> None:
        if 0 <= px < width and 0 <= py < height:
            idx = (py * width + px) * 3
            pixels[idx:idx + 3] = bytes(color)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    def project(x: float, y: float) -> tuple[int, int]:
        px = margin + int((x - min_x) / span_x * (width - 2 * margin))
        py = height - margin - int((y - min_y) / span_y * (height - 2 * margin))
        return px, py

    def line(a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int]) -> None:
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            put_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    points = [project(x, y) for x, y in zip(xs, ys)]
    for a, b in zip(points, points[1:]):
        line(a, b, (31, 119, 180))
    for point, color in ((points[0], (44, 160, 44)), (points[-1], (214, 39, 40))):
        px, py = point
        for ox in range(-3, 4):
            for oy in range(-3, 4):
                put_pixel(px + ox, py + oy, color)

    raw_rows = []
    for y in range(height):
        start = y * width * 3
        raw_rows.append(b"\x00" + bytes(pixels[start:start + width * 3]))
    raw = b"".join(raw_rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def build_reproduce_script(
    run_dir: str | os.PathLike[str],
    task: str | None = None,
    config_name: str = "config.yaml",
) -> Path:
    path = Path(run_dir) / "reproduce.sh"
    task_comment = f"# task: {task}\n" if task else ""
    content = (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        'cd "$(dirname "$0")"\n'
        f"{task_comment}"
        f'python -m aerospace_agent.cli experiment run --config "{config_name}" --output-dir "reproduced"\n'
    )
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def _bullet(items: Iterable[str]) -> str:
    values = [str(item) for item in items if str(item)]
    if not values:
        return "- None recorded.\n"
    return "".join(f"- {item}\n" for item in values)


def write_experiment_report(
    run_dir: str | os.PathLike[str],
    result: Mapping[str, Any],
    report_name: str = "report.md",
) -> Path:
    run_path = Path(run_dir)
    report_path = run_path / report_name
    artifacts = result.get("artifacts") or {}
    invariant_checks = result.get("invariants", {}).get("checks", [])
    risk_lines = [
        f"[{item.get('status')}] {item.get('code')}: {item.get('message')}"
        for item in invariant_checks
        if item.get("status") != "passed"
    ]
    artifact_lines = [
        f"{name}: {value}"
        for name, value in sorted(artifacts.items())
        if value
    ]
    report = (
        f"# Minimal Orbit Experiment Report\n\n"
        f"Run ID: `{result.get('run_id')}`\n\n"
        f"Status: `{result.get('status')}`\n\n"
        "## Completed\n"
        + _bullet(result.get("completed", []))
        + "\n## Verified\n"
        + _bullet(result.get("verified", []))
        + "\n## Unverified\n"
        + _bullet(result.get("unverified", []))
        + "\n## Assumptions\n"
        + _bullet(result.get("assumptions", []))
        + "\n## Inference\n"
        + _bullet(result.get("inference", []))
        + "\n## Failures\n"
        + _bullet(result.get("failures", []))
        + "\n## Risks\n"
        + _bullet(risk_lines)
        + "\n## Artifact Paths\n"
        + _bullet(artifact_lines)
        + "\n## Reproduce\n"
        + "- From the run directory: `sh reproduce.sh`\n"
        + f"- Script path: `{artifacts.get('reproduce_path', run_path / 'reproduce.sh')}`\n"
        + "\n## Next Minimal Experiment\n"
        + "- Cross-validate the same initial state against one independent engine or analytical reference.\n"
    )
    report_path.write_text(report, encoding="utf-8", newline="\n")
    return report_path


def run_minimal_orbit_experiment(
    task: str | None = None,
    output_root: str | os.PathLike[str] = "data/runs",
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    task_text = str(task or (config or {}).get("task") or "minimal orbit propagation")
    merged_config = default_orbit_experiment_config(task_text, config)
    run_id = create_run_id()
    run_dir = Path(output_root) / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    config_path = run_dir / "config.yaml"
    ledger_path = run_dir / "ledger.jsonl"
    state_json_path = artifacts_dir / "state_history.json"
    state_csv_path = artifacts_dir / "state_history.csv"
    orbit_plot_path = artifacts_dir / "orbit.png"

    _write_yaml(config_path, merged_config)
    ledger = LedgerWriter(ledger_path, run_id=run_id)
    ledger.record("run_started", {"task": task_text})
    ledger.record("config_saved", {"path": str(config_path)})

    failures: list[str] = []
    completed: list[str] = [
        "Created run directory.",
        "Saved config.yaml.",
        "Opened JSONL ledger.",
    ]
    verified: list[str] = [
        "run_id was created by the runtime.",
        "config.yaml was written before propagation.",
        "ledger.jsonl contains timestamped events.",
    ]
    unverified: list[str] = [
        "No external astrodynamics engine cross-validation was performed.",
        "The built-in two-body result was not compared against tracking data.",
    ]
    assumptions: list[str] = [
        "two-body point-mass Earth model is used unless config overrides it.",
        "Initial circular LEO state is synthetic when no user state is provided.",
    ]
    inference: list[str] = []

    ledger.record(
        "propagation_started",
        {
            "engine": merged_config.get("engine"),
            "duration_s": merged_config.get("duration_s"),
            "output_step_s": merged_config.get("output_step_s"),
        },
    )
    propagation = propagate_orbit(
        initial_state_dict=merged_config["initial_state"],
        force_model_dict=merged_config["force_model"],
        duration_s=float(merged_config["duration_s"]),
        output_step_s=float(merged_config.get("output_step_s") or merged_config["duration_s"]),
        engine=str(merged_config.get("engine") or "builtin"),
    )

    if propagation.get("status") == "error":
        failures.append(str(propagation.get("reason", "propagation failed")))
        state_history: list[dict[str, Any]] = []
        ledger.record("propagation_failed", {"reason": propagation.get("reason")}, status="error")
    else:
        state_history = list(propagation.get("state_history") or [])
        _write_json(state_json_path, state_history)
        _write_state_history_csv(state_csv_path, state_history)
        plot_orbit_png(state_history, orbit_plot_path)
        metadata = propagation.get("metadata") or {}
        completed.extend([
            "Ran local Python orbit propagator.",
            "Saved state_history.json.",
            "Saved state_history.csv.",
            "Generated orbit.png.",
        ])
        verified.extend([
            f"Propagator returned {len(state_history)} state samples.",
            f"Propagation metadata reports engine={metadata.get('engine')}.",
            f"Propagation metadata reports frame={metadata.get('frame')}.",
        ])
        if state_history:
            first = state_history[0].get("elapsed_s")
            last = state_history[-1].get("elapsed_s")
            inference.append(f"State history covers elapsed_s {first} to {last}.")
        ledger.record(
            "propagation_completed",
            {
                "state_count": len(state_history),
                "metadata": metadata,
            },
        )

    reproduce_path = build_reproduce_script(run_dir, task=task_text)
    completed.append("Saved reproduce.sh.")
    verified.append("reproduce.sh was written by the runtime.")
    ledger.record("reproduce_saved", {"path": str(reproduce_path)})

    artifacts = {
        "config_path": str(config_path),
        "ledger_path": str(ledger_path),
        "state_history_json": str(state_json_path),
        "state_history_csv": str(state_csv_path),
        "orbit_plot": str(orbit_plot_path),
        "reproduce_path": str(reproduce_path),
    }
    invariants = check_aerospace_invariants(merged_config, artifacts=artifacts)

    result: Dict[str, Any] = {
        "status": "failed" if failures else "completed",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "artifacts": artifacts,
        "completed": completed,
        "verified": verified,
        "unverified": unverified,
        "assumptions": assumptions,
        "inference": inference,
        "failures": failures,
        "risks": [
            item
            for item in invariants["checks"]
            if item.get("status") != "passed"
        ],
        "invariants": invariants,
    }
    report_path = write_experiment_report(run_dir, result)
    result["report_path"] = str(report_path)
    result["artifacts"]["report_path"] = str(report_path)
    ledger.record("report_saved", {"path": str(report_path)})
    ledger.record(
        "run_completed" if not failures else "run_failed",
        {
            "status": result["status"],
            "report_path": str(report_path),
            "risk_count": len(result["risks"]),
        },
        status="ok" if not failures else "error",
    )
    return result
