"""Scoped, non-interactive Git operations for one workspace repository."""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .models import ToolError, ToolResult


class _PathOutsideWorkspace(ValueError):
    pass


class GitService:
    """Expose only the approved read and confirmation-gated Git operations."""

    supported_operations = (
        "status",
        "diff",
        "log",
        "branch_info",
        "create_checkpoint",
        "revert_commit",
        "restore_paths",
    )

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        git_executable: str | Path = "git",
        timeout_s: float = 30,
    ) -> None:
        self.root = Path(workspace_root).resolve()
        if not self.root.is_dir():
            raise ValueError("workspace_root must be an existing directory")
        if timeout_s <= 0 or timeout_s > 120:
            raise ValueError("timeout_s must be greater than zero and at most 120")
        self.git_executable = str(git_executable)
        self.timeout_s = timeout_s

    def _operation_id(self, supplied: str | None) -> str:
        return supplied or uuid.uuid4().hex

    def _environment(self) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("GIT_")
        }
        environment.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_EDITOR": "true",
                "GIT_SEQUENCE_EDITOR": "true",
                "GIT_PAGER": "cat",
            }
        )
        return environment

    def _run(self, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = [self.git_executable, *arguments]
        try:
            return subprocess.run(
                argv,
                cwd=self.root,
                env=self._environment(),
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                argv,
                124,
                stdout=exc.stdout or "",
                stderr="git command timed out",
            )
        except OSError as exc:
            return subprocess.CompletedProcess(argv, 127, stdout="", stderr=str(exc))

    def _result(
        self,
        *,
        status: str,
        operation_id: str,
        recovery_class: str,
        result: Mapping[str, object] | None = None,
        error_code: str | None = None,
        message: str = "",
        recoverability: str = "not_applicable",
    ) -> ToolResult:
        error = None
        if error_code is not None:
            error = ToolError(
                code=error_code,
                message=message,
                recoverability=recoverability,
            )
        return ToolResult(
            status=status,
            result=dict(result or {}),
            error=error,
            audit_id=uuid.uuid4().hex,
            operation_id=operation_id,
            recovery_class=recovery_class,
        )

    def _probe(self) -> tuple[bool, str]:
        completed = self._run(["rev-parse", "--show-toplevel"])
        if completed.returncode != 0:
            reason = completed.stderr.strip() or "workspace is not a valid Git repository"
            return False, reason
        reported = completed.stdout.strip()
        if not reported:
            return False, "Git did not report a repository root"
        try:
            repository_root = Path(reported).resolve()
        except OSError as exc:
            return False, str(exc)
        if repository_root != self.root:
            return False, "repository root is outside or above the workspace root"
        return True, ""

    def _unavailable(self, operation_id: str, reason: str) -> ToolResult:
        return self._result(
            status="unavailable",
            operation_id=operation_id,
            recovery_class="read_only",
            result={"available": False},
            error_code="unavailable",
            message=reason,
        )

    def _ensure_repository(self, operation_id: str) -> ToolResult | None:
        available, reason = self._probe()
        if available:
            return None
        return self._unavailable(operation_id, reason)

    def availability(self, *, operation_id: str | None = None) -> ToolResult:
        op_id = self._operation_id(operation_id)
        available, reason = self._probe()
        if not available:
            return self._unavailable(op_id, reason)
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class="read_only",
            result={"available": True, "root": str(self.root)},
        )

    def _normalize_paths(
        self,
        paths: Iterable[str | Path] | None,
        *,
        required: bool = False,
    ) -> list[str]:
        if paths is None:
            if required:
                raise ValueError("at least one scoped path is required")
            return []
        if isinstance(paths, (str, bytes, Path)):
            raise ValueError("paths must be a sequence, not a single path")
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_path in paths:
            if not isinstance(raw_path, (str, Path)) or not str(raw_path):
                raise ValueError("each scoped path must be non-empty")
            raw_text = str(raw_path).replace("\\", "/")
            if raw_text.startswith(":") or any(character in raw_text for character in "*?[]"):
                raise ValueError("Git pathspec magic and wildcard paths are forbidden")
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            resolved = candidate.resolve(strict=False)
            try:
                relative = resolved.relative_to(self.root)
            except ValueError as exc:
                raise _PathOutsideWorkspace(
                    f"path is outside workspace: {raw_path}"
                ) from exc
            if not relative.parts or relative.parts[0].casefold() == ".git":
                raise ValueError("workspace root and Git metadata are not valid scoped paths")
            value = relative.as_posix()
            if value not in seen:
                seen.add(value)
                normalized.append(value)
        if required and not normalized:
            raise ValueError("at least one scoped path is required")
        return normalized

    def _path_error(self, operation_id: str, exc: Exception) -> ToolResult:
        if isinstance(exc, _PathOutsideWorkspace):
            return self._result(
                status="blocked",
                operation_id=operation_id,
                recovery_class="read_only",
                error_code="path_outside_workspace",
                message=str(exc),
            )
        return self._result(
            status="invalid_arguments",
            operation_id=operation_id,
            recovery_class="read_only",
            error_code="invalid_arguments",
            message=str(exc),
        )

    def _command_failure(
        self,
        operation_id: str,
        completed: subprocess.CompletedProcess[str],
        *,
        recovery_class: str,
        error_code: str = "failed",
    ) -> ToolResult:
        message = completed.stderr.strip() or completed.stdout.strip() or "Git command failed"
        if completed.returncode == 124:
            status = "timeout"
            code = "timeout"
        else:
            code = error_code
            status = {
                "invalid_arguments": "invalid_arguments",
                "unavailable": "unavailable",
                "conflict": "blocked",
            }.get(error_code, "failed")
        recoverability = (
            "manual_recovery" if recovery_class == "manual_recovery" else "not_applicable"
        )
        return self._result(
            status=status,
            operation_id=operation_id,
            recovery_class=recovery_class,
            error_code=code,
            message=message,
            recoverability=recoverability,
        )

    def _confirmation_required(self, operation_id: str) -> ToolResult:
        return self._result(
            status="blocked",
            operation_id=operation_id,
            recovery_class="manual_recovery",
            error_code="confirmation_required",
            message="a consumed confirmation is required for Git writes",
            recoverability="manual_recovery",
        )

    def status(
        self,
        *,
        paths: Iterable[str | Path] | None = None,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        unavailable = self._ensure_repository(op_id)
        if unavailable is not None:
            return unavailable
        try:
            scoped_paths = self._normalize_paths(paths)
        except Exception as exc:
            return self._path_error(op_id, exc)
        arguments = ["-c", "core.fsmonitor=false", "status", "--porcelain=v1", "--branch", "--untracked-files=all"]
        if scoped_paths:
            arguments.extend(["--", *scoped_paths])
        completed = self._run(arguments)
        if completed.returncode != 0:
            return self._command_failure(op_id, completed, recovery_class="read_only")
        lines = completed.stdout.splitlines()
        branch = lines[0][3:] if lines and lines[0].startswith("## ") else None
        entries = lines[1:] if branch is not None else lines
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class="read_only",
            result={
                "branch": branch,
                "clean": not entries,
                "entries": entries,
                "raw": completed.stdout,
            },
        )

    def _resolve_commit(self, revision: str) -> tuple[str | None, subprocess.CompletedProcess[str]]:
        if not isinstance(revision, str) or not revision or revision.startswith("-"):
            return None, subprocess.CompletedProcess([], 2, "", "invalid revision")
        completed = self._run(
            ["rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"]
        )
        if completed.returncode != 0:
            return None, completed
        return completed.stdout.strip(), completed

    def diff(
        self,
        *,
        paths: Iterable[str | Path] | None = None,
        staged: bool = False,
        revision: str | None = None,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        unavailable = self._ensure_repository(op_id)
        if unavailable is not None:
            return unavailable
        try:
            scoped_paths = self._normalize_paths(paths)
        except Exception as exc:
            return self._path_error(op_id, exc)
        arguments = ["diff", "--no-ext-diff", "--no-textconv"]
        if staged:
            arguments.append("--cached")
        if revision is not None:
            resolved_revision, resolution = self._resolve_commit(revision)
            if resolved_revision is None:
                return self._command_failure(
                    op_id, resolution, recovery_class="read_only", error_code="invalid_arguments"
                )
            arguments.append(resolved_revision)
        if scoped_paths:
            arguments.extend(["--", *scoped_paths])
        completed = self._run(arguments)
        if completed.returncode != 0:
            return self._command_failure(op_id, completed, recovery_class="read_only")
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class="read_only",
            result={"diff": completed.stdout},
        )

    def log(
        self,
        *,
        max_count: int = 20,
        paths: Iterable[str | Path] | None = None,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        unavailable = self._ensure_repository(op_id)
        if unavailable is not None:
            return unavailable
        if isinstance(max_count, bool) or not isinstance(max_count, int) or not 1 <= max_count <= 100:
            return self._path_error(op_id, ValueError("max_count must be an integer from 1 to 100"))
        try:
            scoped_paths = self._normalize_paths(paths)
        except Exception as exc:
            return self._path_error(op_id, exc)
        arguments = [
            "log",
            "--no-decorate",
            f"--max-count={max_count}",
            "--format=%H%x1f%an%x1f%aI%x1f%s%x1e",
        ]
        if scoped_paths:
            arguments.extend(["--", *scoped_paths])
        completed = self._run(arguments)
        if completed.returncode != 0:
            return self._command_failure(op_id, completed, recovery_class="read_only")
        commits: list[dict[str, str]] = []
        for record in completed.stdout.split("\x1e"):
            record = record.strip("\r\n")
            if not record:
                continue
            fields = record.split("\x1f", maxsplit=3)
            if len(fields) == 4:
                commits.append(
                    {
                        "sha": fields[0],
                        "author": fields[1],
                        "authored_at": fields[2],
                        "subject": fields[3],
                    }
                )
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class="read_only",
            result={"commits": commits},
        )

    def branch_info(self, *, operation_id: str | None = None) -> ToolResult:
        op_id = self._operation_id(operation_id)
        unavailable = self._ensure_repository(op_id)
        if unavailable is not None:
            return unavailable
        completed = self._run(
            ["-c", "core.fsmonitor=false", "status", "--porcelain=v2", "--branch"]
        )
        if completed.returncode != 0:
            return self._command_failure(op_id, completed, recovery_class="read_only")
        data: dict[str, object] = {
            "head": None,
            "oid": None,
            "upstream": None,
            "ahead": 0,
            "behind": 0,
            "detached": False,
            "clean": True,
        }
        for line in completed.stdout.splitlines():
            if line.startswith("# branch.oid "):
                data["oid"] = line.removeprefix("# branch.oid ")
            elif line.startswith("# branch.head "):
                head = line.removeprefix("# branch.head ")
                data["head"] = head
                data["detached"] = head == "(detached)"
            elif line.startswith("# branch.upstream "):
                data["upstream"] = line.removeprefix("# branch.upstream ")
            elif line.startswith("# branch.ab "):
                ahead, behind = line.removeprefix("# branch.ab ").split()
                data["ahead"] = int(ahead[1:])
                data["behind"] = int(behind[1:])
            elif not line.startswith("# "):
                data["clean"] = False
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class="read_only",
            result=data,
        )

    def create_checkpoint(
        self,
        *,
        message: str,
        paths: Iterable[str | Path],
        confirmation_consumed: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        unavailable = self._ensure_repository(op_id)
        if unavailable is not None:
            return unavailable
        if not isinstance(message, str) or not message.strip() or len(message) > 500:
            return self._path_error(
                op_id, ValueError("checkpoint message must contain 1 to 500 characters")
            )
        try:
            scoped_paths = self._normalize_paths(paths, required=True)
        except Exception as exc:
            return self._path_error(op_id, exc)
        if confirmation_consumed is not True:
            return self._confirmation_required(op_id)
        status = self._run(["status", "--porcelain=v1", "--", *scoped_paths])
        if status.returncode != 0:
            return self._command_failure(op_id, status, recovery_class="manual_recovery")
        if not status.stdout.strip():
            return self._result(
                status="failed",
                operation_id=op_id,
                recovery_class="manual_recovery",
                error_code="conflict",
                message="scoped paths contain no changes to checkpoint",
                recoverability="manual_recovery",
            )
        add = self._run(["add", "--", *scoped_paths])
        if add.returncode != 0:
            return self._command_failure(op_id, add, recovery_class="manual_recovery")
        commit = self._run(
            [
                "-c",
                f"core.hooksPath={os.devnull}",
                "-c",
                "commit.gpgSign=false",
                "commit",
                "--only",
                "--no-verify",
                f"--message={message.strip()}",
                "--",
                *scoped_paths,
            ]
        )
        if commit.returncode != 0:
            return self._command_failure(
                op_id, commit, recovery_class="manual_recovery", error_code="conflict"
            )
        head = self._run(["rev-parse", "HEAD"])
        if head.returncode != 0:
            return self._command_failure(op_id, head, recovery_class="manual_recovery")
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class="manual_recovery",
            result={"commit_sha": head.stdout.strip(), "paths": scoped_paths},
        )

    def restore_paths(
        self,
        *,
        paths: Iterable[str | Path],
        source: str = "HEAD",
        confirmation_consumed: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        unavailable = self._ensure_repository(op_id)
        if unavailable is not None:
            return unavailable
        try:
            scoped_paths = self._normalize_paths(paths, required=True)
        except Exception as exc:
            return self._path_error(op_id, exc)
        resolved_source, resolution = self._resolve_commit(source)
        if resolved_source is None:
            return self._command_failure(
                op_id, resolution, recovery_class="read_only", error_code="invalid_arguments"
            )
        if confirmation_consumed is not True:
            return self._confirmation_required(op_id)
        restored = self._run(
            [
                "restore",
                f"--source={resolved_source}",
                "--worktree",
                "--staged",
                "--",
                *scoped_paths,
            ]
        )
        if restored.returncode != 0:
            return self._command_failure(op_id, restored, recovery_class="manual_recovery")
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class="manual_recovery",
            result={"source": resolved_source, "paths": scoped_paths},
        )

    def _scope_covers(self, scope: Sequence[str], changed_paths: Sequence[str]) -> bool:
        if not changed_paths:
            return False
        covered_scope: set[str] = set()
        for changed in changed_paths:
            matches = [
                scoped
                for scoped in scope
                if changed == scoped or changed.startswith(scoped.rstrip("/") + "/")
            ]
            if not matches:
                return False
            covered_scope.update(matches)
        return covered_scope == set(scope)

    def revert_commit(
        self,
        commit: str,
        *,
        paths: Iterable[str | Path],
        confirmation_consumed: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        unavailable = self._ensure_repository(op_id)
        if unavailable is not None:
            return unavailable
        try:
            scoped_paths = self._normalize_paths(paths, required=True)
        except Exception as exc:
            return self._path_error(op_id, exc)
        resolved_commit, resolution = self._resolve_commit(commit)
        if resolved_commit is None:
            return self._command_failure(
                op_id, resolution, recovery_class="read_only", error_code="invalid_arguments"
            )
        if confirmation_consumed is not True:
            return self._confirmation_required(op_id)
        clean = self._run(["status", "--porcelain=v1", "--untracked-files=all"])
        if clean.returncode != 0:
            return self._command_failure(op_id, clean, recovery_class="read_only")
        if clean.stdout.strip():
            return self._result(
                status="blocked",
                operation_id=op_id,
                recovery_class="read_only",
                error_code="conflict",
                message="revert requires a clean working tree and index",
            )
        head = self._run(["rev-parse", "HEAD"])
        if head.returncode != 0:
            return self._command_failure(op_id, head, recovery_class="read_only")
        if head.stdout.strip() != resolved_commit:
            return self._result(
                status="blocked",
                operation_id=op_id,
                recovery_class="read_only",
                error_code="conflict",
                message="only the current HEAD commit can be reversibly reverted",
            )
        parents = self._run(["rev-list", "--parents", "--max-count=1", resolved_commit])
        if parents.returncode != 0:
            return self._command_failure(op_id, parents, recovery_class="read_only")
        if len(parents.stdout.split()) > 2:
            return self._result(
                status="invalid_arguments",
                operation_id=op_id,
                recovery_class="read_only",
                error_code="invalid_arguments",
                message="merge commits require a mainline and are not supported",
            )
        changed = self._run(
            [
                "diff-tree",
                "--root",
                "--no-commit-id",
                "--name-only",
                "-r",
                "-z",
                resolved_commit,
            ]
        )
        if changed.returncode != 0:
            return self._command_failure(op_id, changed, recovery_class="read_only")
        changed_paths = [path for path in changed.stdout.split("\0") if path]
        if not self._scope_covers(scoped_paths, changed_paths):
            return self._result(
                status="invalid_arguments",
                operation_id=op_id,
                recovery_class="read_only",
                error_code="invalid_arguments",
                message="scoped paths must cover exactly the paths changed by the commit",
            )
        reverted = self._run(
            [
                "-c",
                f"core.hooksPath={os.devnull}",
                "-c",
                "commit.gpgSign=false",
                "revert",
                "--no-edit",
                resolved_commit,
            ]
        )
        if reverted.returncode != 0:
            self._run(["revert", "--abort"])
            return self._command_failure(op_id, reverted, recovery_class="manual_recovery")
        new_head = self._run(["rev-parse", "HEAD"])
        if new_head.returncode != 0:
            return self._command_failure(op_id, new_head, recovery_class="manual_recovery")
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class="reversible",
            result={
                "reverted_commit": resolved_commit,
                "revert_commit_sha": new_head.stdout.strip(),
                "paths": scoped_paths,
            },
        )


__all__ = ["GitService"]
