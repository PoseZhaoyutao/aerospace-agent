"""Restricted argv-only subprocess execution for one workspace root."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from ..models import ToolError, ToolResult


_SHELL_TOKENS = frozenset({">", ">>", "<", "<<", "|", "||", "&&", ";"})
_PATH_VALUE_OPTIONS = frozenset(
    {"-C", "--git-dir", "--work-tree", "--output", "--src-prefix", "--dst-prefix"}
)
_SYSTEM_ENV_ALLOWLIST = frozenset(
    {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP"}
)


@dataclass
class _ManagedProcess:
    process: subprocess.Popen[str]
    recovery_class: str
    max_output_chars: int
    stdout: str | None = None
    stderr: str | None = None


class TerminalService:
    """Run allowlisted commands without a shell or ambient environment leakage."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        allowed_commands: Iterable[str | Path],
        env_allowlist: Iterable[str] = (),
    ) -> None:
        self.root = Path(workspace_root).resolve()
        if not self.root.is_dir():
            raise ValueError("workspace_root must be an existing directory")
        self._allowed_paths: set[str] = set()
        self._allowed_names: set[str] = set()
        for command in allowed_commands:
            text = str(command)
            path = Path(text)
            if path.is_absolute():
                self._allowed_paths.add(os.path.normcase(str(path.resolve())))
            else:
                self._allowed_names.add(text.casefold())
        self._env_allowlist = _SYSTEM_ENV_ALLOWLIST | frozenset(env_allowlist)
        self._processes: dict[str, _ManagedProcess] = {}
        self._lock = threading.Lock()

    def _operation_id(self, supplied: str | None) -> str:
        return supplied or uuid.uuid4().hex

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

    def _invalid(self, operation_id: str, message: str) -> ToolResult:
        return self._result(
            status="invalid_arguments",
            operation_id=operation_id,
            recovery_class="read_only",
            error_code="invalid_arguments",
            message=message,
        )

    def _resolve_cwd(self, cwd: str | Path | None) -> Path:
        candidate = self.root if cwd is None else Path(cwd)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("cwd is outside workspace") from exc
        if not resolved.is_dir():
            raise ValueError("cwd must be an existing directory")
        return resolved

    def _allowed(self, command: str) -> bool:
        path = Path(command)
        if path.is_absolute():
            return (
                os.path.normcase(str(path.resolve())) in self._allowed_paths
                or path.name.casefold() in self._allowed_names
            )
        if command.casefold() in self._allowed_names:
            return True
        located = shutil.which(command)
        return bool(
            located
            and os.path.normcase(str(Path(located).resolve())) in self._allowed_paths
        )

    def _validate_argv(self, argv: Sequence[str] | object) -> list[str]:
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
            raise ValueError("argv must be a non-empty sequence of strings")
        values = list(argv)
        if not values or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("argv must be a non-empty sequence of non-empty strings")
        if any(
            value in _SHELL_TOKENS or "$(" in value or "`" in value
            for value in values
        ):
            raise ValueError("shell operators, redirection, and subshells are forbidden")
        return values

    def _assert_workspace_path(self, value: str, *, cwd: Path) -> None:
        parsed = urlsplit(value)
        if parsed.scheme.casefold() == "file":
            raw_path = unquote(parsed.path)
            if parsed.netloc and parsed.netloc.casefold() != "localhost":
                raw_path = f"//{parsed.netloc}{raw_path}"
            elif os.name == "nt" and len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
                raw_path = raw_path[1:]
            candidate = Path(raw_path)
        else:
            candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"argument path is outside workspace: {value}") from exc

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return (
            urlsplit(value).scheme.casefold() == "file"
            or Path(value).is_absolute()
            or "/" in value
            or "\\" in value
            or value in {".", ".."}
            or value.startswith(("~/", "~\\"))
        )

    def _validate_argument_paths(self, argv: Sequence[str], *, cwd: Path) -> None:
        after_separator = False
        force_next_path = False
        for value in argv[1:]:
            if force_next_path:
                self._assert_workspace_path(value, cwd=cwd)
                force_next_path = False
                continue
            if value == "--":
                after_separator = True
                continue
            option, separator, option_value = value.partition("=")
            if option in _PATH_VALUE_OPTIONS:
                if separator:
                    self._assert_workspace_path(option_value, cwd=cwd)
                else:
                    force_next_path = True
                continue
            if after_separator or self._looks_like_path(option_value if separator else value):
                self._assert_workspace_path(option_value if separator else value, cwd=cwd)

        executable = Path(argv[0]).name.casefold()
        arguments = list(argv[1:])
        if executable not in {"git", "git.exe"} or arguments[:1] != ["diff"]:
            return
        if "--no-index" not in arguments:
            return
        skip_next = False
        for value in arguments[1:]:
            if skip_next:
                skip_next = False
                continue
            if value == "--":
                continue
            option, separator, _ = value.partition("=")
            if option in _PATH_VALUE_OPTIONS:
                skip_next = not separator
                continue
            if value.startswith("-"):
                continue
            self._assert_workspace_path(value, cwd=cwd)

    def _environment(self, supplied: Mapping[str, str] | None) -> dict[str, str]:
        supplied = supplied or {}
        rejected = set(supplied) - self._env_allowlist
        if rejected:
            raise ValueError(
                "environment variable is not allowlisted: " + ", ".join(sorted(rejected))
            )
        environment = {
            key: value for key, value in os.environ.items() if key in _SYSTEM_ENV_ALLOWLIST
        }
        environment.update(supplied)
        return environment

    def _classify(self, argv: Sequence[str], *, background: bool) -> str:
        if background:
            return "manual_recovery"
        arguments = tuple(value.casefold() for value in argv[1:])
        if arguments and all(
            value in {"--version", "-v", "-vv", "--help", "-h"} for value in arguments
        ):
            return "read_only"
        executable = Path(argv[0]).name.casefold()
        if executable in {"git", "git.exe"} and arguments[:1] in {
            ("status",),
            ("diff",),
            ("log",),
        }:
            return "read_only"
        return "manual_recovery"

    def _popen(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.Popen[str]:
        process_options: dict[str, object] = {}
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True
        return subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(env),
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **process_options,
        )

    def _terminate(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    def _cap(self, stdout: str, stderr: str, maximum: int) -> tuple[str, str, bool]:
        combined_length = len(stdout) + len(stderr)
        if combined_length <= maximum:
            return stdout, stderr, False
        shown_stdout = stdout[:maximum]
        remaining = maximum - len(shown_stdout)
        shown_stderr = stderr[:remaining]
        return shown_stdout, shown_stderr, True

    def run(
        self,
        argv: Sequence[str] | object,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 120,
        max_output_chars: int = 100_000,
        background: bool = False,
        confirmed: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            values = self._validate_argv(argv)
            if timeout_s <= 0 or timeout_s > 120:
                return self._invalid(op_id, "timeout_s must be greater than zero and at most 120")
            if max_output_chars < 1:
                return self._invalid(op_id, "max_output_chars must be positive")
            working_directory = self._resolve_cwd(cwd)
            self._validate_argument_paths(values, cwd=working_directory)
            environment = self._environment(env)
        except PermissionError as exc:
            return self._result(
                status="blocked",
                operation_id=op_id,
                recovery_class="read_only",
                error_code="path_outside_workspace",
                message=str(exc),
            )
        except ValueError as exc:
            return self._invalid(op_id, str(exc))

        if not self._allowed(values[0]):
            return self._result(
                status="unavailable",
                operation_id=op_id,
                recovery_class="read_only",
                error_code="unavailable",
                message="command is not allowlisted",
            )
        recovery_class = self._classify(values, background=background)
        if recovery_class != "read_only" and not confirmed:
            return self._result(
                status="blocked",
                operation_id=op_id,
                recovery_class="manual_recovery",
                error_code="confirmation_required",
                message="unknown, writing, or long-running command requires confirmation",
                recoverability="manual_recovery",
            )
        try:
            process = self._popen(values, cwd=working_directory, env=environment)
        except OSError as exc:
            return self._result(
                status="failed",
                operation_id=op_id,
                recovery_class=recovery_class,
                error_code="failed",
                message=str(exc),
                recoverability=(
                    "manual_recovery" if recovery_class == "manual_recovery" else "not_applicable"
                ),
            )

        if background:
            process_id = uuid.uuid4().hex
            with self._lock:
                self._processes[process_id] = _ManagedProcess(
                    process=process,
                    recovery_class=recovery_class,
                    max_output_chars=max_output_chars,
                )
            return self._result(
                status="success",
                operation_id=op_id,
                recovery_class=recovery_class,
                result={"process_id": process_id, "pid": process.pid, "running": True},
            )

        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._terminate(process)
            stdout, stderr = process.communicate()
            shown_stdout, shown_stderr, truncated = self._cap(
                stdout or "", stderr or "", max_output_chars
            )
            return self._result(
                status="timeout",
                operation_id=op_id,
                recovery_class=recovery_class,
                result={
                    "stdout": shown_stdout,
                    "stderr": shown_stderr,
                    "truncated": truncated,
                },
                error_code="timeout",
                message="command exceeded timeout_s",
                recoverability=(
                    "manual_recovery" if recovery_class == "manual_recovery" else "retryable"
                ),
            )
        shown_stdout, shown_stderr, truncated = self._cap(
            stdout or "", stderr or "", max_output_chars
        )
        if process.returncode != 0:
            return self._result(
                status="failed",
                operation_id=op_id,
                recovery_class=recovery_class,
                result={
                    "returncode": process.returncode,
                    "stdout": shown_stdout,
                    "stderr": shown_stderr,
                    "truncated": truncated,
                },
                error_code="failed",
                message=f"command exited with code {process.returncode}",
                recoverability=(
                    "manual_recovery" if recovery_class == "manual_recovery" else "not_applicable"
                ),
            )
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class=recovery_class,
            result={
                "returncode": process.returncode,
                "stdout": shown_stdout,
                "stderr": shown_stderr,
                "truncated": truncated,
            },
        )

    def _managed(self, process_id: str) -> _ManagedProcess | None:
        with self._lock:
            return self._processes.get(process_id)

    def status(
        self, process_id: str, *, operation_id: str | None = None
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        managed = self._managed(process_id)
        if managed is None:
            return self._invalid(op_id, "unknown process_id")
        returncode = managed.process.poll()
        if returncode is None:
            return self._result(
                status="success",
                operation_id=op_id,
                recovery_class="read_only",
                result={"process_id": process_id, "running": True},
            )
        if managed.stdout is None or managed.stderr is None:
            stdout, stderr = managed.process.communicate()
            managed.stdout, managed.stderr, truncated = self._cap(
                stdout or "", stderr or "", managed.max_output_chars
            )
        else:
            truncated = False
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class="read_only",
            result={
                "process_id": process_id,
                "running": False,
                "returncode": returncode,
                "stdout": managed.stdout,
                "stderr": managed.stderr,
                "truncated": truncated,
            },
        )

    def cancel(
        self,
        process_id: str,
        *,
        confirmed: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        managed = self._managed(process_id)
        if managed is None:
            return self._invalid(op_id, "unknown process_id")
        if not confirmed:
            return self._result(
                status="blocked",
                operation_id=op_id,
                recovery_class="manual_recovery",
                error_code="confirmation_required",
                message="process cancellation requires confirmation",
                recoverability="manual_recovery",
            )
        was_running = managed.process.poll() is None
        self._terminate(managed.process)
        stdout, stderr = managed.process.communicate()
        managed.stdout, managed.stderr, truncated = self._cap(
            stdout or "", stderr or "", managed.max_output_chars
        )
        return self._result(
            status="success",
            operation_id=op_id,
            recovery_class="manual_recovery",
            result={
                "process_id": process_id,
                "cancelled": was_running,
                "returncode": managed.process.returncode,
                "stdout": managed.stdout,
                "stderr": managed.stderr,
                "truncated": truncated,
            },
        )


__all__ = ["TerminalService"]
