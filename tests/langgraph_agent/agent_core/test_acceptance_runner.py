from __future__ import annotations

import json
import stat
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts import run_agent_core_acceptance as acceptance_runner
from scripts.run_agent_core_acceptance import ACCEPTANCE_ITEMS, run_acceptance


def _write_junit(
    artifact_root: Path,
    outcomes: dict[str, str],
    *,
    filename: str = "offline-results.xml",
) -> None:
    suite = ET.Element("testsuite", name="pytest", tests=str(len(outcomes)))
    for node_id, outcome in outcomes.items():
        path, name = node_id.split("::", 1)
        classname = path.removesuffix(".py").replace("/", ".")
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        if outcome != "passed":
            ET.SubElement(case, outcome, message=f"synthetic {outcome}")
    document = ET.Element("testsuites", name="pytest tests")
    document.append(suite)
    ET.ElementTree(document).write(
        artifact_root / filename,
        encoding="utf-8",
        xml_declaration=True,
    )


def test_each_acceptance_item_has_specific_offline_evidence_selectors() -> None:
    mapping = acceptance_runner.ACCEPTANCE_EVIDENCE

    assert set(mapping) == {item_id for item_id, _ in ACCEPTANCE_ITEMS}
    assert all(mapping[item_id] for item_id, _ in ACCEPTANCE_ITEMS)
    assert all(
        selector.startswith("tests/") and ".py::test_" in selector
        for selectors in mapping.values()
        for selector in selectors
    )


def test_runtime_report_fails_closed_when_an_acceptance_item_is_unmapped(
    monkeypatch,
) -> None:
    monkeypatch.delitem(acceptance_runner.ACCEPTANCE_EVIDENCE, "AC-15")

    items = acceptance_runner._acceptance_results({})
    by_id = {item["acceptance_id"]: item for item in items}

    assert len(items) == len(ACCEPTANCE_ITEMS)
    assert by_id["AC-15"]["status"] == "failed"
    assert by_id["AC-15"]["evidence"]["return_code"] == 5
    assert by_id["AC-15"]["evidence"]["missing_selectors"] == [
        "unmapped_acceptance"
    ]


def test_per_item_status_comes_only_from_its_mapped_collected_nodes(
    tmp_path: Path, monkeypatch
) -> None:
    mapping = acceptance_runner.ACCEPTANCE_EVIDENCE
    failed_selector = mapping["AC-02"][0]
    missing_selector = mapping["AC-03"][0]
    outcomes = {
        selector: "passed"
        for selectors in mapping.values()
        for selector in selectors
        if selector != missing_selector
    }
    outcomes[failed_selector] = "failure"

    def mixed(command, *, cwd, env, text, capture_output):
        artifact = Path(env["AEROSPACE_TEST_ARTIFACT_ROOT"])
        _write_junit(artifact, outcomes)
        return subprocess.CompletedProcess(command, 1, stdout="mixed result", stderr="")

    monkeypatch.setattr(subprocess, "run", mixed)

    report = run_acceptance(tmp_path, run_id="per-item")
    items = {item["acceptance_id"]: item for item in report["acceptance_items"]}

    assert items["AC-01"]["status"] == "passed"
    assert items["AC-01"]["evidence"]["return_code"] == 0
    assert items["AC-02"]["status"] == "failed"
    assert items["AC-02"]["evidence"]["summary"]["failed"] == 1
    assert items["AC-03"]["status"] == "failed"
    assert missing_selector in items["AC-03"]["evidence"]["missing_selectors"]
    assert items["AC-03"]["evidence"]["return_code"] == 5
    assert report["status"] == "failed"


def test_live_qwen_is_reported_separately_and_cannot_supply_offline_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    mapping = acceptance_runner.ACCEPTANCE_EVIDENCE
    offline_outcomes = {
        selector: "passed"
        for selectors in mapping.values()
        for selector in selectors
    }
    calls = 0

    def completed(command, *, cwd, env, text, capture_output):
        nonlocal calls
        calls += 1
        if calls == 1:
            _write_junit(Path(env["AEROSPACE_TEST_ARTIFACT_ROOT"]), offline_outcomes)
            return subprocess.CompletedProcess(command, 0, stdout="offline passed", stderr="")
        _write_junit(
            Path(env["AEROSPACE_TEST_ARTIFACT_ROOT"]),
            {
                f"tests/langgraph_agent/test_qwen_acceptance.py::test_live_{index}": "passed"
                for index in range(5)
            },
            filename="live-qwen-results.xml",
        )
        return subprocess.CompletedProcess(command, 0, stdout="5 passed", stderr="")

    monkeypatch.setattr(subprocess, "run", completed)

    report = run_acceptance(tmp_path, run_id="live-separated", live_qwen=True)

    assert calls == 2
    assert report["live_qwen"]["status"] == "passed"
    assert report["live_qwen"]["return_code"] == 0
    assert all(
        "test_qwen_acceptance.py" not in node_id
        for item in report["acceptance_items"]
        for node_id in item["evidence"]["node_ids"]
    )


def test_live_qwen_all_skipped_is_not_reported_as_passed(
    tmp_path: Path, monkeypatch
) -> None:
    offline_outcomes = {
        selector: "passed"
        for selectors in acceptance_runner.ACCEPTANCE_EVIDENCE.values()
        for selector in selectors
    }
    calls = 0

    def skipped(command, *, cwd, env, text, capture_output):
        nonlocal calls
        calls += 1
        artifact = Path(env["AEROSPACE_TEST_ARTIFACT_ROOT"])
        if calls == 1:
            _write_junit(artifact, offline_outcomes)
            return subprocess.CompletedProcess(command, 0, stdout="offline passed", stderr="")
        _write_junit(
            artifact,
            {
                f"tests/langgraph_agent/test_qwen_acceptance.py::test_live_{index}": "skipped"
                for index in range(5)
            },
            filename="live-qwen-results.xml",
        )
        return subprocess.CompletedProcess(command, 0, stdout="5 skipped", stderr="")

    monkeypatch.setattr(subprocess, "run", skipped)

    report = run_acceptance(tmp_path, run_id="live-skipped", live_qwen=True)

    assert report["status"] == "failed"
    assert report["live_qwen"]["status"] == "blocked"
    assert report["live_qwen"]["summary"] == {
        "passed": 0,
        "failed": 0,
        "error": 0,
        "skipped": 5,
    }


def test_runner_uses_unique_artifact_root_writes_one_report_and_cleans_success(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def passed(command, *, cwd, env, text, capture_output):
        calls.append((list(command), dict(env)))
        artifact = Path(env["AEROSPACE_TEST_ARTIFACT_ROOT"])
        artifact.mkdir(parents=True, exist_ok=True)
        readonly_probe = artifact / "probe.tmp"
        readonly_probe.write_text("test", encoding="utf-8")
        readonly_probe.chmod(stat.S_IREAD)
        _write_junit(
            artifact,
            {
                selector: "passed"
                for selectors in acceptance_runner.ACCEPTANCE_EVIDENCE.values()
                for selector in selectors
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="42 passed", stderr="")

    monkeypatch.setattr(subprocess, "run", passed)

    first = run_acceptance(tmp_path, run_id="run-a")
    second = run_acceptance(tmp_path, run_id="run-b", keep_artifacts=True)

    assert first["status"] == "passed"
    assert len(first["acceptance_items"]) == len(ACCEPTANCE_ITEMS) == 15
    assert not Path(first["artifact_root"]).exists()
    assert Path(second["artifact_root"]).exists()
    reports = sorted((tmp_path / "reports").glob("agent_core_acceptance_*.json"))
    assert len(reports) == 2
    assert json.loads(reports[0].read_text(encoding="utf-8"))["run_id"] == "run-a"
    assert calls[0][1]["AEROSPACE_TEST_ARTIFACT_ROOT"].endswith("run-a")
    artifact_root = Path(calls[0][1]["AEROSPACE_TEST_ARTIFACT_ROOT"])
    assert calls[0][1]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert Path(calls[0][1]["TEMP"]).is_relative_to(artifact_root)
    assert Path(calls[0][1]["TMP"]).is_relative_to(artifact_root)
    basetemp_index = calls[0][0].index("--basetemp") + 1
    assert Path(calls[0][0][basetemp_index]).is_relative_to(artifact_root)


def test_runner_keeps_failed_artifacts_for_diagnosis(tmp_path: Path, monkeypatch) -> None:
    def failed(command, *, cwd, env, text, capture_output):
        artifact = Path(env["AEROSPACE_TEST_ARTIFACT_ROOT"])
        artifact.mkdir(parents=True, exist_ok=True)
        _write_junit(
            artifact,
            {
                selector: "failure"
                for selectors in acceptance_runner.ACCEPTANCE_EVIDENCE.values()
                for selector in selectors
            },
        )
        return subprocess.CompletedProcess(command, 1, stdout="1 failed", stderr="failure")

    monkeypatch.setattr(subprocess, "run", failed)

    report = run_acceptance(tmp_path, run_id="failed-run")

    assert report["status"] == "failed"
    assert Path(report["artifact_root"]).exists()
    assert all(item["status"] == "failed" for item in report["acceptance_items"])
