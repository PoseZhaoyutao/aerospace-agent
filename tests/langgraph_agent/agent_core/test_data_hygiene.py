from __future__ import annotations

import os
from pathlib import Path


def test_project_tree_has_no_stray_test_or_bytecode_artifacts() -> None:
    root = Path(__file__).resolve().parents[3]
    allowed_run_raw = os.environ.get("AEROSPACE_TEST_ARTIFACT_ROOT", "").strip()
    allowed_run = Path(allowed_run_raw).resolve() if allowed_run_raw else None
    violations: list[str] = []

    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in {".git", ".agents", ".codex"}:
            continue
        if allowed_run is not None and (path == allowed_run or path.is_relative_to(allowed_run)):
            continue
        if path.is_dir() and (
            path.name in {"__pycache__", ".pytest_cache"}
            or path.name.startswith("pytest-cache-files-")
        ):
            violations.append(relative.as_posix())
            continue
        if path.is_file() and (
            path.suffix.casefold() in {".pyc", ".pyo", ".tmp", ".temp"}
            or "rollback-quarantine-" in path.name
        ):
            violations.append(relative.as_posix())

    artifacts = root / ".test-artifacts"
    if artifacts.exists():
        for child in artifacts.iterdir():
            if allowed_run is None or not (
                child == allowed_run or allowed_run.is_relative_to(child)
            ):
                violations.append(child.relative_to(root).as_posix())

    assert not sorted(set(violations)), "stray generated artifacts:\n" + "\n".join(
        sorted(set(violations))
    )
