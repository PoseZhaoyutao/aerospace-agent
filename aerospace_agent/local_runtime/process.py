"""Subprocess wrapper with a stable Unicode decode policy."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence


DEFAULT_ENCODING = "utf-8"
DEFAULT_ERRORS = "replace"


@dataclass(frozen=True)
class CommandResult:
    """Structured result returned by :func:`run_command`."""

    cmd: list[str]
    cwd: str | None
    returncode: int
    stdout: str
    stderr: str
    timeout: bool
    encoding: str = DEFAULT_ENCODING

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timeout


def _to_text(value: str | bytes | None, encoding: str = DEFAULT_ENCODING) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(encoding, errors=DEFAULT_ERRORS)
    return value


def _truncate(value: str, max_output_chars: int | None) -> str:
    if max_output_chars is None or max_output_chars < 0:
        return value
    if len(value) <= max_output_chars:
        return value
    return value[-max_output_chars:]


def run_command(
    cmd: Sequence[str | os.PathLike[str]],
    cwd: str | os.PathLike[str] | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    max_output_chars: int | None = None,
) -> CommandResult:
    """Run a command without using the platform default text encoding.

    Windows often defaults to cp936/GBK. Many developer tools emit UTF-8 or raw
    bytes, which can crash ``subprocess`` reader threads when ``text=True`` uses
    the locale codec. This wrapper always decodes as UTF-8 with replacement.
    """

    normalized_cmd = [os.fspath(part) for part in cmd]
    normalized_cwd = os.fspath(cwd) if cwd is not None else None
    try:
        proc = subprocess.run(
            normalized_cmd,
            cwd=normalized_cwd,
            env=dict(env) if env is not None else None,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding=DEFAULT_ENCODING,
            errors=DEFAULT_ERRORS,
        )
        return CommandResult(
            cmd=normalized_cmd,
            cwd=normalized_cwd,
            returncode=proc.returncode,
            stdout=_truncate(proc.stdout or "", max_output_chars),
            stderr=_truncate(proc.stderr or "", max_output_chars),
            timeout=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _truncate(_to_text(exc.stdout), max_output_chars)
        stderr = _truncate(_to_text(exc.stderr), max_output_chars)
        message = f"Command timed out after {timeout}s"
        if stderr:
            stderr = f"{stderr}\n{message}"
        else:
            stderr = message
        return CommandResult(
            cmd=normalized_cmd,
            cwd=normalized_cwd,
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            timeout=True,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            cmd=normalized_cmd,
            cwd=normalized_cwd,
            returncode=-1,
            stdout="",
            stderr=str(exc),
            timeout=False,
        )
    except Exception as exc:
        return CommandResult(
            cmd=normalized_cmd,
            cwd=normalized_cwd,
            returncode=-1,
            stdout="",
            stderr=str(exc),
            timeout=False,
        )
