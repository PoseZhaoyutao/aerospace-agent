from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import pytest
import yaml

from aerospace_agent.langgraph_agent.agent_core.project_memory import ProjectIdentityService


def test_status_is_uninitialized_without_creating_files(tmp_path) -> None:
    service = ProjectIdentityService(tmp_path)

    status = service.status()

    assert status.state == "uninitialized"
    assert not (tmp_path / "memory" / "project").exists()


def test_initialize_is_idempotent_preserves_user_files_and_creates_required_layout(tmp_path) -> None:
    project_memory = tmp_path / "memory" / "project"
    project_memory.mkdir(parents=True)
    project_md = project_memory / "PROJECT.md"
    project_md.write_text("# User-owned project memory\n", encoding="utf-8")
    service = ProjectIdentityService(tmp_path)

    first = service.initialize()
    second = service.initialize()

    assert UUID(first.project_id)
    assert second.project_id == first.project_id
    assert project_md.read_text(encoding="utf-8") == "# User-owned project memory\n"
    assert (project_memory / "constraints.yaml").is_file()
    assert (project_memory / "manifest.yaml").is_file()
    assert (project_memory / "decisions").is_dir()
    assert (project_memory / "workflows").is_dir()
    assert (tmp_path / "data" / "langgraph" / "session_memory.sqlite").is_file()
    assert (tmp_path / "data" / "langgraph" / "project_memory_index.sqlite").is_file()
    assert not (project_memory / ".init.lock").exists()


def test_manifest_identity_is_random_uuid_not_path_derived(tmp_path) -> None:
    service = ProjectIdentityService(tmp_path)
    initialized = service.initialize()
    manifest = yaml.safe_load(
        (tmp_path / "memory" / "project" / "manifest.yaml").read_text(encoding="utf-8")
    )

    assert manifest["project_id"] == initialized.project_id
    assert initialized.project_id not in str(tmp_path)
    assert manifest["schema_version"] == "1.0"


def test_concurrent_initialization_produces_one_stable_project_id(tmp_path) -> None:
    def initialize() -> str:
        return ProjectIdentityService(tmp_path, lock_timeout_seconds=5).initialize().project_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        project_ids = list(pool.map(lambda _: initialize(), range(2)))

    assert len(set(project_ids)) == 1


def test_partial_initialization_is_repaired_without_changing_identity(tmp_path) -> None:
    service = ProjectIdentityService(tmp_path)
    first = service.initialize()
    (tmp_path / "data" / "langgraph" / "project_memory_index.sqlite").unlink()

    repaired = service.initialize()

    assert repaired.project_id == first.project_id
    assert service.status().state == "ready"


def test_project_index_scans_allowlist_and_excludes_data_secrets_and_test_cache(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("agent rules", encoding="utf-8")
    (tmp_path / "README.md").write_text("read me", encoding="utf-8")
    config = tmp_path / "config"
    config.mkdir()
    (config / "agent.yaml").write_text("safe: true", encoding="utf-8")
    (config / "secret-token.yaml").write_text("token: forbidden", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "private.md").write_text("must not index", encoding="utf-8")
    cache = tmp_path / ".test-artifacts"
    cache.mkdir()
    (cache / "temporary.md").write_text("must not index", encoding="utf-8")
    service = ProjectIdentityService(tmp_path)
    initialized = service.initialize()

    indexed = service.indexed_paths()

    assert "AGENTS.md" in indexed
    assert "README.md" in indexed
    assert "config/agent.yaml" in indexed
    assert "config/secret-token.yaml" not in indexed
    assert "data/private.md" not in indexed
    assert ".test-artifacts/temporary.md" not in indexed
    assert all(not path.startswith("data/") for path in indexed)
    assert service.status().project_id == initialized.project_id


def test_project_databases_use_separate_versioned_schemas(tmp_path) -> None:
    ProjectIdentityService(tmp_path).initialize()

    for name in ("session_memory.sqlite", "project_memory_index.sqlite"):
        with sqlite3.connect(tmp_path / "data" / "langgraph" / name) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


def test_invalid_manifest_reports_migration_failed_without_guessing_project_id(tmp_path) -> None:
    manifest_dir = tmp_path / "memory" / "project"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.yaml").write_text("project_id: not-a-uuid\n", encoding="utf-8")

    status = ProjectIdentityService(tmp_path).status()

    assert status.state == "migration_failed"
    assert status.project_id is None


def test_project_memory_search_is_project_scoped_and_reads_only_indexed_sources(tmp_path) -> None:
    service = ProjectIdentityService(tmp_path)
    service.initialize()
    (tmp_path / "memory/project/PROJECT.md").write_text(
        "# Project\n\nApproved orbit convention is GCRF.\n", encoding="utf-8"
    )
    (tmp_path / "data/langgraph/private.txt").write_text(
        "private orbit token", encoding="utf-8"
    )
    service.reindex()

    matches = service.search("orbit convention")

    assert matches[0]["relative_path"] == "memory/project/PROJECT.md"
    assert "GCRF" in matches[0]["content"]
    assert all("private" not in item["content"] for item in matches)
    assert service.read_indexed("memory/project/PROJECT.md").startswith("# Project")
    with pytest.raises(KeyError):
        service.read_indexed("data/langgraph/private.txt")

