from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import aerospace_agent.langgraph_agent.agent_core.git_service as git_service_module
from aerospace_agent.langgraph_agent.agent_core.git_service import GitService


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_EDITOR": "true",
            "GIT_SEQUENCE_EDITOR": "true",
        }
    )
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=environment,
        shell=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "agent-core@example.invalid")
    _git(repo, "config", "user.name", "Agent Core Tests")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "a.txt").write_text("a0\n", encoding="utf-8", newline="")
    (repo / "b.txt").write_text("b0\n", encoding="utf-8", newline="")
    _git(repo, "add", "--", "a.txt", "b.txt")
    _git(repo, "commit", "--quiet", "--message=initial")
    return repo


def test_empty_dot_git_is_truthfully_unavailable(tmp_path: Path) -> None:
    workspace = tmp_path / "invalid"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    service = GitService(workspace)

    availability = service.availability()
    status = service.status()

    assert availability.status == "unavailable"
    assert availability.result["available"] is False
    assert availability.error is not None
    assert availability.error.code == "unavailable"
    assert status.status == "unavailable"
    assert status.recovery_class == "read_only"


def test_detects_valid_repo_and_read_operations_are_read_only(repository: Path) -> None:
    service = GitService(repository)
    (repository / "a.txt").write_text("a1\n", encoding="utf-8", newline="")
    (repository / "untracked.txt").write_text("new\n", encoding="utf-8", newline="")

    availability = service.availability()
    status = service.status()
    diff = service.diff(paths=["a.txt"])
    log = service.log(max_count=1)
    branch = service.branch_info()

    assert availability.status == "success"
    assert availability.result == {"available": True, "root": str(repository.resolve())}
    assert status.status == "success"
    assert status.recovery_class == "read_only"
    assert any("a.txt" in entry for entry in status.result["entries"])
    assert any("untracked.txt" in entry for entry in status.result["entries"])
    assert diff.status == "success"
    assert "-a0" in diff.result["diff"] and "+a1" in diff.result["diff"]
    assert log.status == "success"
    assert len(log.result["commits"]) == 1
    assert log.result["commits"][0]["subject"] == "initial"
    assert branch.status == "success"
    assert branch.result["head"]
    assert branch.result["oid"] == _git(repository, "rev-parse", "HEAD").stdout.strip()


def test_all_git_invocations_use_argv_no_shell_workspace_cwd_and_noninteractive_env(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_run = subprocess.run
    calls: list[tuple[list[str], dict[str, object]]] = []

    def spy(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((list(argv), dict(kwargs)))
        return original_run(argv, **kwargs)

    monkeypatch.setattr(git_service_module.subprocess, "run", spy)
    service = GitService(repository)

    assert service.status().status == "success"
    assert service.diff().status == "success"
    assert service.log(max_count=1).status == "success"
    assert service.branch_info().status == "success"

    assert calls
    for argv, kwargs in calls:
        assert isinstance(argv, list)
        assert kwargs["shell"] is False
        assert Path(kwargs["cwd"]).resolve() == repository.resolve()
        assert kwargs["stdin"] is subprocess.DEVNULL
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["GIT_TERMINAL_PROMPT"] == "0"
        assert environment["GIT_EDITOR"] == "true"
        assert environment["GIT_SEQUENCE_EDITOR"] == "true"


def test_create_checkpoint_requires_consumed_confirmation_and_commits_only_scoped_paths(
    repository: Path,
) -> None:
    service = GitService(repository)
    old_head = _git(repository, "rev-parse", "HEAD").stdout.strip()
    (repository / "a.txt").write_text("a1\n", encoding="utf-8", newline="")
    (repository / "b.txt").write_text("b1\n", encoding="utf-8", newline="")

    blocked = service.create_checkpoint(message="checkpoint a", paths=["a.txt"])
    created = service.create_checkpoint(
        message="checkpoint a",
        paths=["a.txt"],
        confirmation_consumed=True,
    )

    assert blocked.status == "blocked"
    assert blocked.error is not None
    assert blocked.error.code == "confirmation_required"
    assert created.status == "success"
    assert created.recovery_class == "manual_recovery"
    assert created.result["commit_sha"] != old_head
    changed = _git(
        repository, "show", "--pretty=format:", "--name-only", "HEAD"
    ).stdout.splitlines()
    assert [item for item in changed if item] == ["a.txt"]
    assert "b.txt" in _git(repository, "status", "--porcelain").stdout


def test_git_write_paths_are_workspace_scoped(repository: Path, tmp_path: Path) -> None:
    service = GitService(repository)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8", newline="")
    old_head = _git(repository, "rev-parse", "HEAD").stdout.strip()

    result = service.create_checkpoint(
        message="escape",
        paths=[outside],
        confirmation_consumed=True,
    )

    assert result.status == "blocked"
    assert result.error is not None
    assert result.error.code == "path_outside_workspace"
    assert _git(repository, "rev-parse", "HEAD").stdout.strip() == old_head


def test_confirmation_consumed_must_be_literal_true(repository: Path) -> None:
    service = GitService(repository)
    (repository / "a.txt").write_text("a1\n", encoding="utf-8", newline="")
    old_head = _git(repository, "rev-parse", "HEAD").stdout.strip()

    result = service.create_checkpoint(
        message="not confirmed",
        paths=["a.txt"],
        confirmation_consumed=1,  # type: ignore[arg-type]
    )

    assert result.status == "blocked"
    assert result.error is not None
    assert result.error.code == "confirmation_required"
    assert _git(repository, "rev-parse", "HEAD").stdout.strip() == old_head


def test_restore_paths_requires_confirmation_is_scoped_and_manual_recovery(
    repository: Path,
) -> None:
    service = GitService(repository)
    (repository / "a.txt").write_text("a1\n", encoding="utf-8", newline="")
    (repository / "b.txt").write_text("b1\n", encoding="utf-8", newline="")

    blocked = service.restore_paths(paths=["a.txt"])
    restored = service.restore_paths(
        paths=["a.txt"], confirmation_consumed=True
    )

    assert blocked.status == "blocked"
    assert blocked.error is not None
    assert blocked.error.code == "confirmation_required"
    assert restored.status == "success"
    assert restored.recovery_class == "manual_recovery"
    assert (repository / "a.txt").read_text(encoding="utf-8") == "a0\n"
    assert (repository / "b.txt").read_text(encoding="utf-8") == "b1\n"


def test_invalid_restore_revision_is_invalid_arguments(repository: Path) -> None:
    service = GitService(repository)

    result = service.restore_paths(
        paths=["a.txt"],
        source="--hard",
        confirmation_consumed=True,
    )

    assert result.status == "invalid_arguments"
    assert result.error is not None
    assert result.error.code == "invalid_arguments"


def test_revert_head_is_reversible_only_with_clean_tree_exact_scope_and_confirmation(
    repository: Path,
) -> None:
    service = GitService(repository)
    (repository / "a.txt").write_text("a1\n", encoding="utf-8", newline="")
    checkpoint = service.create_checkpoint(
        message="change a", paths=["a.txt"], confirmation_consumed=True
    )
    commit_sha = checkpoint.result["commit_sha"]
    head_before_revert = _git(repository, "rev-parse", "HEAD").stdout.strip()

    blocked = service.revert_commit(commit_sha, paths=["a.txt"])
    head_after_blocked = _git(repository, "rev-parse", "HEAD").stdout.strip()
    reverted = service.revert_commit(
        commit_sha,
        paths=["a.txt"],
        confirmation_consumed=True,
    )

    assert blocked.status == "blocked"
    assert head_after_blocked == head_before_revert
    assert _git(repository, "rev-parse", "HEAD").stdout.strip() != head_before_revert
    assert reverted.status == "success"
    assert reverted.recovery_class == "reversible"
    assert reverted.result["reverted_commit"] == commit_sha
    assert (repository / "a.txt").read_text(encoding="utf-8") == "a0\n"


def test_revert_rejects_partial_scope_without_changing_head(repository: Path) -> None:
    service = GitService(repository)
    (repository / "a.txt").write_text("a1\n", encoding="utf-8", newline="")
    (repository / "b.txt").write_text("b1\n", encoding="utf-8", newline="")
    checkpoint = service.create_checkpoint(
        message="change both",
        paths=["a.txt", "b.txt"],
        confirmation_consumed=True,
    )
    head = checkpoint.result["commit_sha"]

    result = service.revert_commit(
        head,
        paths=["a.txt"],
        confirmation_consumed=True,
    )

    assert result.status == "invalid_arguments"
    assert _git(repository, "rev-parse", "HEAD").stdout.strip() == head


def test_service_never_offers_or_runs_banned_destructive_git_operations(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_run = subprocess.run
    calls: list[list[str]] = []

    def spy(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return original_run(argv, **kwargs)

    monkeypatch.setattr(git_service_module.subprocess, "run", spy)
    service = GitService(repository)
    (repository / "a.txt").write_text("a1\n", encoding="utf-8", newline="")
    checkpoint = service.create_checkpoint(
        message="safe checkpoint",
        paths=["a.txt"],
        confirmation_consumed=True,
    )
    assert checkpoint.status == "success"
    assert service.revert_commit(
        checkpoint.result["commit_sha"],
        paths=["a.txt"],
        confirmation_consumed=True,
    ).status == "success"

    assert service.supported_operations == (
        "status",
        "diff",
        "log",
        "branch_info",
        "create_checkpoint",
        "revert_commit",
        "restore_paths",
    )
    assert not hasattr(service, "reset_hard")
    assert not hasattr(service, "force_push")
    assert not hasattr(service, "delete_branch")
    assert all(not ("reset" in argv and "--hard" in argv) for argv in calls)
    assert all(not ("push" in argv and "--force" in argv) for argv in calls)
    assert all(not ("branch" in argv and ("-d" in argv or "-D" in argv)) for argv in calls)

