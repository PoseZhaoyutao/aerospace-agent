"""Workspace-contained file tools with durable preimage journaling."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Mapping

from ..journal import OperationJournal, PreimageConflict, PreparedOperation
from ..models import ToolError, ToolResult


DEFAULT_IMPORTANT_PATHS = (
    "AGENTS.md",
    ".gitignore",
    "requirements*.txt",
    "pyproject.toml",
    "setup.py",
    "config/**",
    "memory/project/**",
    "workflows/**",
    "evolved_skills/**",
    "docs/**",
)


class _PathOutsideWorkspace(ValueError):
    pass


class _PathLockConflict(RuntimeError):
    pass


class FileService:
    """File operations constrained to one resolved workspace root."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        journal: OperationJournal | object | None = None,
        important_paths: Iterable[str] = DEFAULT_IMPORTANT_PATHS,
        excluded_directories: Iterable[str] = (".git", ".agent_core", "__pycache__"),
    ) -> None:
        self.root = Path(workspace_root).resolve()
        if not self.root.is_dir():
            raise ValueError("workspace_root must be an existing directory")
        self.journal = journal or OperationJournal(
            self.root / ".agent_core" / "operation_journal.sqlite3"
        )
        self.important_paths = tuple(important_paths)
        self.excluded_directories = frozenset(excluded_directories)

    def _operation_id(self, supplied: str | None) -> str:
        return supplied or uuid.uuid4().hex

    def _resolve(self, path: str | Path) -> Path:
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError("path must be a non-empty string or Path")
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise _PathOutsideWorkspace(f"path is outside workspace: {path}") from exc
        return resolved

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _is_important(self, path: Path) -> bool:
        relative = self._relative(path)
        for pattern in self.important_paths:
            normalized = pattern.replace("\\", "/")
            if normalized.endswith("/**"):
                prefix = normalized[:-3].rstrip("/")
                if relative == prefix or relative.startswith(prefix + "/"):
                    return True
            if fnmatch.fnmatchcase(relative, normalized):
                return True
        return False

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
        audit_id: str | None = None,
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
            audit_id=audit_id or uuid.uuid4().hex,
            operation_id=operation_id,
            recovery_class=recovery_class,
        )

    def _path_failure(self, operation_id: str, exc: Exception) -> ToolResult:
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

    def _read_only_failure(self, operation_id: str, exc: Exception) -> ToolResult:
        if isinstance(exc, (_PathOutsideWorkspace, ValueError)):
            return self._path_failure(operation_id, exc)
        return self._result(
            status="failed",
            operation_id=operation_id,
            recovery_class="read_only",
            error_code="failed",
            message=str(exc),
        )

    def read(
        self,
        path: str | Path,
        *,
        max_bytes: int = 1_000_000,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            if max_bytes < 1:
                raise ValueError("max_bytes must be positive")
            target = self._resolve(path)
            if not target.is_file():
                raise FileNotFoundError(f"file does not exist: {path}")
            data = target.read_bytes()
            shown = data[:max_bytes]
            return self._result(
                status="success",
                operation_id=op_id,
                recovery_class="read_only",
                result={
                    "path": self._relative(target),
                    "content": shown.decode("utf-8", errors="replace"),
                    "byte_length": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "truncated": len(shown) < len(data),
                },
            )
        except Exception as exc:
            return self._read_only_failure(op_id, exc)

    def read_lines(
        self,
        path: str | Path,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        max_bytes: int = 1_000_000,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            if start_line < 1 or (end_line is not None and end_line < start_line):
                raise ValueError("line range is invalid")
            if max_bytes < 1:
                raise ValueError("max_bytes must be positive")
            target = self._resolve(path)
            if not target.is_file():
                raise FileNotFoundError(f"file does not exist: {path}")
            data = target.read_bytes()
            lines = data.decode("utf-8", errors="replace").splitlines(keepends=True)
            selected = "".join(lines[start_line - 1 : end_line])
            encoded = selected.encode("utf-8")
            shown = encoded[:max_bytes]
            return self._result(
                status="success",
                operation_id=op_id,
                recovery_class="read_only",
                result={
                    "path": self._relative(target),
                    "content": shown.decode("utf-8", errors="replace"),
                    "start_line": start_line,
                    "end_line": min(end_line or len(lines), len(lines)),
                    "truncated": len(shown) < len(encoded),
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
            )
        except Exception as exc:
            return self._read_only_failure(op_id, exc)

    def list(
        self,
        path: str | Path = ".",
        *,
        max_results: int = 1_000,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            if max_results < 1:
                raise ValueError("max_results must be positive")
            directory = self._resolve(path)
            if not directory.is_dir():
                raise NotADirectoryError(f"directory does not exist: {path}")
            children = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
            visible = children[:max_results]
            entries = [
                {
                    "name": child.name,
                    "path": child.relative_to(self.root).as_posix(),
                    "type": "symlink"
                    if child.is_symlink()
                    else "directory"
                    if child.is_dir()
                    else "file",
                }
                for child in visible
            ]
            return self._result(
                status="success",
                operation_id=op_id,
                recovery_class="read_only",
                result={"entries": entries, "truncated": len(visible) < len(children)},
            )
        except Exception as exc:
            return self._read_only_failure(op_id, exc)

    def info(
        self, path: str | Path, *, operation_id: str | None = None
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            target = self._resolve(path)
            if not target.exists():
                raise FileNotFoundError(f"path does not exist: {path}")
            stat = target.stat()
            kind = "directory" if target.is_dir() else "file"
            payload: dict[str, object] = {
                "path": self._relative(target),
                "type": kind,
                "byte_length": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            }
            if target.is_file():
                payload["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            return self._result(
                status="success",
                operation_id=op_id,
                recovery_class="read_only",
                result=payload,
            )
        except Exception as exc:
            return self._read_only_failure(op_id, exc)

    def stat(
        self, path: str | Path, *, operation_id: str | None = None
    ) -> ToolResult:
        """Compatibility name for the file metadata operation."""
        return self.info(path, operation_id=operation_id)

    def search(
        self,
        path: str | Path,
        query: str,
        *,
        max_results: int = 100,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            if not query or max_results < 1:
                raise ValueError("query must be non-empty and max_results positive")
            base = self._resolve(path)
            candidates = [base] if base.is_file() else base.rglob("*")
            matches: list[dict[str, object]] = []
            truncated = False
            for candidate in candidates:
                if any(part in self.excluded_directories for part in candidate.parts):
                    continue
                try:
                    resolved = self._resolve(candidate)
                except _PathOutsideWorkspace:
                    continue
                if not resolved.is_file():
                    continue
                try:
                    lines = resolved.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeError):
                    continue
                for line_number, line in enumerate(lines, start=1):
                    if query not in line:
                        continue
                    if len(matches) == max_results:
                        truncated = True
                        break
                    matches.append(
                        {
                            "path": self._relative(resolved),
                            "line": line_number,
                            "text": line,
                        }
                    )
                if truncated:
                    break
            return self._result(
                status="success",
                operation_id=op_id,
                recovery_class="read_only",
                result={"matches": matches, "truncated": truncated},
            )
        except Exception as exc:
            return self._read_only_failure(op_id, exc)

    def _confirmed_result(self, operation_id: str, message: str) -> ToolResult:
        return self._result(
            status="blocked",
            operation_id=operation_id,
            recovery_class="read_only",
            error_code="confirmation_required",
            message=message,
        )

    def _conflict(self, operation_id: str, message: str) -> ToolResult:
        return self._result(
            status="blocked",
            operation_id=operation_id,
            recovery_class="read_only",
            error_code="conflict",
            message=message,
        )

    @contextmanager
    def _lock_paths(self, paths: Mapping[str, Path]):
        """Coordinate mutations across local FileService processes."""

        lock_root = self.root / ".agent_core" / "path_locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        descriptors: list[tuple[int, Path]] = []
        try:
            for target in sorted({str(path) for path in paths.values()}):
                digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
                lock_path = lock_root / f"{digest}.lock"
                deadline = time.monotonic() + 5.0
                while True:
                    try:
                        descriptor = os.open(
                            lock_path,
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                            0o600,
                        )
                        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
                        descriptors.append((descriptor, lock_path))
                        break
                    except FileExistsError as exc:
                        if time.monotonic() >= deadline:
                            raise _PathLockConflict(
                                f"target is locked by another operation: {target}"
                            ) from exc
                        time.sleep(0.01)
            yield
        finally:
            for descriptor, lock_path in reversed(descriptors):
                try:
                    os.close(descriptor)
                finally:
                    lock_path.unlink(missing_ok=True)

    def _mutate(
        self,
        *,
        operation_id: str,
        action: str,
        paths: Mapping[str, Path],
        metadata: Mapping[str, object] | None,
        mutation: Callable[[], None],
        result: Callable[[], Mapping[str, object]],
    ) -> ToolResult:
        prepared: PreparedOperation | None = None
        try:
            with self._lock_paths(paths):
                prepared = self.journal.prepare(
                    operation_id=operation_id,
                    action=action,
                    paths=paths,
                    metadata=metadata,
                )
                for path in paths.values():
                    # Re-resolve immediately before mutation so a replaced
                    # symlink or junction cannot redirect an earlier path.
                    if self._resolve(path) != path:
                        raise PreimageConflict("target path identity changed before mutation")
                self.journal.assert_preimages_unchanged(operation_id)
                mutation()
                self.journal.complete(operation_id)
                return self._result(
                    status="success",
                    operation_id=operation_id,
                    recovery_class="reversible",
                    result=result(),
                    audit_id=prepared.audit_id,
                )
        except (PreimageConflict, _PathLockConflict) as exc:
            if prepared is not None:
                try:
                    self.journal.fail(operation_id, str(exc))
                except Exception:
                    pass
            return self._conflict(operation_id, str(exc))
        except Exception as exc:
            if prepared is None:
                return self._result(
                    status="failed",
                    operation_id=operation_id,
                    recovery_class="read_only",
                    error_code="failed",
                    message=f"operation journal preparation failed: {exc}",
                )
            try:
                self.journal.fail(operation_id, str(exc))
            except Exception:
                pass
            return self._result(
                status="failed",
                operation_id=operation_id,
                recovery_class="reversible",
                error_code="failed",
                message=str(exc),
                recoverability="reversible",
                audit_id=prepared.audit_id,
            )

    def _atomic_bytes(self, target: Path, data: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def write(
        self,
        path: str | Path,
        content: str | bytes,
        *,
        overwrite: bool = False,
        confirmed: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            target = self._resolve(path)
            if not isinstance(content, (str, bytes)):
                raise ValueError("content must be str or bytes")
            if not target.parent.is_dir():
                raise ValueError("parent directory does not exist")
            if target.exists() and target.is_dir():
                raise ValueError("target is a directory")
            if target.exists() and not overwrite:
                return self._conflict(op_id, "target exists and overwrite is false")
            if target.exists() and self._is_important(target) and not confirmed:
                return self._confirmed_result(op_id, "important file overwrite requires confirmation")
            data = content.encode("utf-8") if isinstance(content, str) else content
            return self._mutate(
                operation_id=op_id,
                action="file.write",
                paths={"target": target},
                metadata={"overwrite": overwrite},
                mutation=lambda: self._atomic_bytes(target, data),
                result=lambda: {
                    "path": self._relative(target),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "byte_length": target.stat().st_size,
                },
            )
        except Exception as exc:
            return self._path_failure(op_id, exc)

    def append(
        self,
        path: str | Path,
        content: str | bytes,
        *,
        confirmed: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            target = self._resolve(path)
            if not isinstance(content, (str, bytes)):
                raise ValueError("content must be str or bytes")
            if not target.parent.is_dir():
                raise ValueError("parent directory does not exist")
            if target.exists() and not target.is_file():
                raise ValueError("target is not a file")
            if target.exists() and self._is_important(target) and not confirmed:
                return self._confirmed_result(
                    op_id, "important file append requires confirmation"
                )
            before = target.read_bytes() if target.exists() else b""
            suffix = content.encode("utf-8") if isinstance(content, str) else content
            metadata = {
                "original_length": len(before),
                "original_sha256": hashlib.sha256(before).hexdigest(),
            }
            return self._mutate(
                operation_id=op_id,
                action="file.append",
                paths={"target": target},
                metadata=metadata,
                mutation=lambda: self._atomic_bytes(
                    target,
                    (target.read_bytes() if target.exists() else b"") + suffix,
                ),
                result=lambda: {
                    "path": self._relative(target),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "byte_length": target.stat().st_size,
                },
            )
        except Exception as exc:
            return self._path_failure(op_id, exc)

    def mkdir(
        self, path: str | Path, *, operation_id: str | None = None
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            target = self._resolve(path)
            if target.is_dir():
                return self._result(
                    status="success",
                    operation_id=op_id,
                    recovery_class="read_only",
                    result={"path": self._relative(target), "created": False},
                )
            if target.exists():
                return self._conflict(op_id, "target exists and is not a directory")
            if not target.parent.is_dir():
                raise ValueError("parent directory does not exist")
            return self._mutate(
                operation_id=op_id,
                action="file.mkdir",
                paths={"target": target},
                metadata=None,
                mutation=target.mkdir,
                result=lambda: {"path": self._relative(target), "created": True},
            )
        except Exception as exc:
            return self._path_failure(op_id, exc)

    def _atomic_copy(self, source: Path, destination: Path) -> None:
        if source.is_dir():
            temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
            try:
                shutil.rmtree(temporary)
                shutil.copytree(source, temporary, symlinks=True)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def copy(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
        confirmed: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            source_path = self._resolve(source)
            destination_path = self._resolve(destination)
            if not source_path.exists():
                raise ValueError("source does not exist")
            if not destination_path.parent.is_dir():
                raise ValueError("destination parent does not exist")
            if destination_path.exists() and not overwrite:
                return self._conflict(op_id, "destination exists and overwrite is false")
            if (
                destination_path.exists()
                and self._is_important(destination_path)
                and not confirmed
            ):
                return self._confirmed_result(
                    op_id, "important file overwrite requires confirmation"
                )
            return self._mutate(
                operation_id=op_id,
                action="file.copy",
                paths={"destination": destination_path},
                metadata={"source": self._relative(source_path), "overwrite": overwrite},
                mutation=lambda: self._atomic_copy(source_path, destination_path),
                result=lambda: {
                    "source": self._relative(source_path),
                    "destination": self._relative(destination_path),
                },
            )
        except Exception as exc:
            return self._path_failure(op_id, exc)

    def move(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
        confirmed: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            source_path = self._resolve(source)
            destination_path = self._resolve(destination)
            if not source_path.exists():
                raise ValueError("source does not exist")
            if self._is_important(source_path) and not confirmed:
                return self._confirmed_result(op_id, "important file move requires confirmation")
            if not destination_path.parent.is_dir():
                raise ValueError("destination parent does not exist")
            if destination_path.exists() and not overwrite:
                return self._conflict(op_id, "destination exists and overwrite is false")
            if (
                destination_path.exists()
                and self._is_important(destination_path)
                and not confirmed
            ):
                return self._confirmed_result(
                    op_id, "important file overwrite requires confirmation"
                )
            return self._mutate(
                operation_id=op_id,
                action="file.move",
                paths={"source": source_path, "destination": destination_path},
                metadata={"overwrite": overwrite},
                mutation=lambda: os.replace(source_path, destination_path),
                result=lambda: {
                    "source": self._relative(source_path),
                    "destination": self._relative(destination_path),
                },
            )
        except Exception as exc:
            return self._path_failure(op_id, exc)

    def delete(
        self,
        path: str | Path,
        *,
        recursive: bool = False,
        confirmed: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        op_id = self._operation_id(operation_id)
        try:
            target = self._resolve(path)
            if not target.exists():
                raise ValueError("target does not exist")
            if not confirmed:
                return self._confirmed_result(op_id, "delete requires confirmation")
            if target.is_dir() and not recursive:
                raise ValueError("recursive=True is required to delete a directory")

            def mutation() -> None:
                quarantine = target.parent / f".{target.name}.{op_id}.delete"
                os.replace(target, quarantine)
                if quarantine.is_dir():
                    shutil.rmtree(quarantine)
                else:
                    quarantine.unlink()

            return self._mutate(
                operation_id=op_id,
                action="file.delete",
                paths={"target": target},
                metadata={"recursive": recursive, "important": self._is_important(target)},
                mutation=mutation,
                result=lambda: {"path": self._relative(target), "deleted": True},
            )
        except Exception as exc:
            return self._path_failure(op_id, exc)


__all__ = ["DEFAULT_IMPORTANT_PATHS", "FileService"]
