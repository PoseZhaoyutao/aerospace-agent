import json
from pathlib import Path

import yaml

from aerospace_agent.experiment_runtime import (
    LedgerWriter,
    check_aerospace_invariants,
    run_minimal_orbit_experiment,
)


def test_ledger_writes_jsonl_events(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = LedgerWriter(ledger_path, run_id="run_test")

    ledger.record("started", {"task": "orbit smoke"})
    ledger.record("finished", {"status": "ok"})

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["run_id"] == "run_test"
    assert first["event"] == "started"
    assert first["payload"]["task"] == "orbit smoke"
    assert "timestamp_utc" in first
    assert second["event"] == "finished"


def test_aerospace_invariants_marks_missing_domain_metadata_as_risk():
    result = check_aerospace_invariants(
        {
            "initial_state": {
                "epoch": {"value": "2026-01-01T00:00:00"},
                "frame": {"name": "GCRF"},
                "position_m": [6778137.0, 0.0, 0.0],
                "velocity_mps": [0.0, 7668.6, 0.0],
            }
        },
        artifacts={},
    )

    codes = {item["code"]: item["status"] for item in result["checks"]}
    assert codes["time_system"] == "risk"
    assert codes["frame"] == "risk"
    assert codes["units"] == "risk"
    assert codes["attitude"] == "risk"
    assert codes["camera_model"] == "risk"
    assert codes["snr_photometry"] == "risk"
    assert result["summary"]["risk"] >= 6


def test_aerospace_invariants_require_minimum_sensor_and_truth_fields():
    result = check_aerospace_invariants(
        {
            "units": {"position": "m", "velocity": "m/s"},
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
                "position_m": [6778137.0, 0.0, 0.0],
                "velocity_mps": [0.0, 7668.6, 0.0],
            },
            "attitude": {"representation": "quaternion"},
            "camera_model": {"focal_length_m": 1.0},
            "image_origin": {"row": "top"},
            "photometry": {"exposure_s": 1.0},
            "snr_model": {"read_noise_e": 3.0},
            "psf_model": {"type": "gaussian"},
            "truth_image_mapping": {"truth_path": "truth.csv"},
        },
        artifacts={},
    )

    codes = {item["code"]: item["status"] for item in result["checks"]}
    assert codes["time_system"] == "passed"
    assert codes["frame"] == "passed"
    assert codes["units"] == "passed"
    assert codes["attitude"] == "risk"
    assert codes["camera_model"] == "risk"
    assert codes["image_origin"] == "risk"
    assert codes["snr_photometry"] == "risk"
    assert codes["psf_model"] == "risk"
    assert codes["truth_image_consistency"] == "risk"


def test_minimal_orbit_experiment_produces_reproducible_run_artifacts(tmp_path):
    result = run_minimal_orbit_experiment(
        task="propagate one circular LEO orbit for one hour",
        output_root=tmp_path,
        config={
            "duration_s": 3600.0,
            "output_step_s": 600.0,
        },
    )

    run_dir = Path(result["run_dir"])
    artifacts_dir = run_dir / "artifacts"

    assert result["status"] == "completed"
    assert result["assumptions"]
    assert any("two-body" in item.lower() for item in result["assumptions"])
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "ledger.jsonl").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "reproduce.sh").is_file()
    assert (artifacts_dir / "state_history.json").is_file()
    assert (artifacts_dir / "state_history.csv").is_file()
    assert (artifacts_dir / "orbit.png").is_file()

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["task"] == "propagate one circular LEO orbit for one hour"
    assert config["model"]["name"] == "builtin_two_body"
    assert config["initial_state"]["epoch"]["scale"] == "UTC"
    assert config["initial_state"]["frame"]["name"] == "GCRF"

    history = json.loads((artifacts_dir / "state_history.json").read_text(encoding="utf-8"))
    assert len(history) == 7
    assert history[0]["elapsed_s"] == 0.0
    assert history[-1]["elapsed_s"] == 3600.0

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    for heading in [
        "## Completed",
        "## Verified",
        "## Unverified",
        "## Assumptions",
        "## Inference",
        "## Failures",
        "## Risks",
        "## Artifact Paths",
        "## Reproduce",
        "## Next Minimal Experiment",
    ]:
        assert heading in report

    ledger_events = [
        json.loads(line)["event"]
        for line in (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert ledger_events[0] == "run_started"
    assert "propagation_completed" in ledger_events
    assert ledger_events[-1] == "run_completed"
