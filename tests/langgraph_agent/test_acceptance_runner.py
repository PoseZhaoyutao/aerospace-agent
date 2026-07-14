from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def test_acceptance_script_bootstraps_repository_imports(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "run_langgraph_acceptance.py"
    code = (
        "import runpy; "
        f"ns = runpy.run_path({str(script)!r}, run_name='acceptance_import_probe'); "
        "print(ns['load_settings'].__globals__['__file__'])"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).resolve() == (
        root / "aerospace_agent" / "langgraph_agent" / "config.py"
    ).resolve()


def test_qwen_conditional_rag_smoke_is_blocked_without_endpoint(tmp_path):
    from scripts import run_langgraph_acceptance as runner

    result = runner.qwen_conditional_rag_smoke(
        {"status": "blocked", "error": "offline"},
        tmp_path,
    )

    assert result == {"status": "blocked", "reason": "offline"}


def test_acceptance_markdown_separates_offline_and_live_observations():
    from scripts import run_langgraph_acceptance as runner

    markdown = runner._markdown({
        "tests": {
            "counts": {"passed": 1},
            "qwen": {
                "status": "available",
                "model": "qwythos",
                "conditional_rag_smoke": {"status": "observed", "policy_satisfied": True},
            },
        },
    })

    assert "## Verification" in markdown
    assert "### Offline pytest" in markdown
    assert "### Live Qwen conditional RAG" in markdown
    assert "## Offline tests" not in markdown


def test_available_qwen_policy_failure_makes_acceptance_exit_nonzero(tmp_path, monkeypatch):
    from scripts import run_langgraph_acceptance as runner

    monkeypatch.setattr(
        runner,
        "run_tests",
        lambda _workspace: {
            "exit_code": 0,
            "qwen": {
                "status": "available",
                "conditional_rag_smoke": {
                    "status": "observed",
                    "policy_satisfied": False,
                },
            },
        },
    )

    exit_code = runner.main(["--workspace", str(tmp_path), "--run-tests"])

    assert exit_code == 1


def test_acl_fallback_does_not_mask_full_pytest_failure(tmp_path, monkeypatch):
    from scripts import run_langgraph_acceptance as runner

    monkeypatch.setattr(
        runner,
        "run_tests",
        lambda _workspace: {
            "exit_code": 1,
            "fixture_acl": {"status": "blocked"},
            "manual_fallback": {"status": "ok"},
            "qwen": {"status": "blocked", "error": "offline"},
        },
    )

    assert runner.main(["--workspace", str(tmp_path), "--run-tests"]) == 1


def test_blocked_qwen_does_not_report_live_acceptance_success(tmp_path, monkeypatch):
    from scripts import run_langgraph_acceptance as runner

    monkeypatch.setattr(
        runner,
        "run_tests",
        lambda _workspace: {
            "exit_code": 0,
            "qwen": {"status": "blocked", "error": "offline"},
        },
    )

    assert runner.main(["--workspace", str(tmp_path), "--run-tests"]) == 1


def test_agent_smoke_records_both_conditional_rag_paths(tmp_path):
    from scripts import run_langgraph_acceptance as runner

    result = runner._agent_smoke(tmp_path)

    assert type(result["general"]["status"]) is str
    assert result["general"]["status"] == "success"
    assert result["general"]["retrieval_required"] is False
    assert result["general"]["rag_hits"] == 0
    assert result["general"]["citation_count"] == 0
    assert result["general"]["search_call_count"] == 0
    assert type(result["evidence"]["status"]) is str
    assert result["evidence"]["status"] == "success"
    assert result["evidence"]["retrieval_required"] is True
    assert result["evidence"]["retrieval_reason"] == "explicit_evidence"
    assert result["evidence"]["rag_hits"] > 0
    assert result["evidence"]["citation_count"] > 0
    assert result["evidence"]["search_call_count"] == 1
    assert result["claim_support"]["unsupported_claims"] == []


def test_successful_acceptance_run_removes_centralized_test_artifacts(tmp_path, monkeypatch):
    from scripts import run_langgraph_acceptance as runner

    monkeypatch.setattr(
        runner,
        "probe_qwen_endpoint",
        lambda *_args, **_kwargs: {"status": "blocked", "error": "offline"},
    )
    monkeypatch.setattr(
        runner,
        "qwen_claim_smoke",
        lambda _qwen: {"status": "blocked", "reason": "offline"},
    )
    monkeypatch.setattr(runner, "_agent_smoke", lambda _workspace: {"status": "ok"})

    def completed_process(_command, **kwargs):
        artifact_root = Path(kwargs["env"]["AEROSPACE_TEST_ARTIFACT_ROOT"])
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "temporary.txt").write_text("temporary", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="1 passed in 0.01s", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", completed_process)

    result = runner.run_tests(tmp_path)

    artifact_root = tmp_path / ".test-artifacts" / "acceptance-pytest"
    assert result["fixture_acl"]["basetemp_removed"] is True
    assert not artifact_root.exists()
