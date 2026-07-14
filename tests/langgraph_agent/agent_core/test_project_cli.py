from __future__ import annotations

import json

from aerospace_agent.langgraph_agent.agent_core.project_memory import ProjectIdentityService
from start_langgraph_agent import main


def test_project_initialization_status_and_reindex_cli(tmp_path, capsys) -> None:
    assert main(["--workspace", str(tmp_path), "--init-project", "--json"]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["state"] == "ready"

    assert main(["--workspace", str(tmp_path), "--project-memory-status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["state"] == "ready"
    assert status["project_id"] == initialized["project_id"]

    assert main(["--workspace", str(tmp_path), "--project-memory-reindex", "--json"]) == 0
    reindexed = json.loads(capsys.readouterr().out)
    assert reindexed["state"] == "ready"
    assert reindexed["indexed_documents"] >= 3


def test_legacy_repl_status_does_not_implicitly_initialize_project(tmp_path, capsys) -> None:
    assert main(["--workspace", str(tmp_path), "--json"]) == 0
    ready = json.loads(capsys.readouterr().out)

    assert ready == {"status": "ready", "thread_id": "default"}
    assert ProjectIdentityService(tmp_path).status().state == "uninitialized"

