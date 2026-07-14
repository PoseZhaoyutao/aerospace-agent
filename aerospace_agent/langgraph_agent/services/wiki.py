"""Small, deterministic Markdown wiki storage primitives."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Iterable


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WikiPage:
    topic: str
    slug: str
    title: str
    text: str
    related_topics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def page_id(self) -> str:
        return f"seed:{self.topic}"

    @property
    def page_path(self) -> str:
        return f"knowledge/orbital-dynamics/{self.slug}.md"

    @property
    def content_sha256(self) -> str:
        return sha256_text(self.text)


class WikiStore:
    """A root-constrained store with atomic, content-addressed writes."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        workspace: str | os.PathLike[str] | None = None,
    ):
        if root is None:
            root = workspace
        if root is None:
            raise TypeError("WikiStore requires a root")
        self.root = Path(root).resolve()

    def resolve_relative(self, relative: str | os.PathLike[str]) -> Path:
        value = str(relative).strip()
        if not value or value in {".", ".."}:
            raise ValueError("knowledge root: wiki path must be non-empty and relative")
        # Validate both path syntaxes even when running on the other OS.
        if (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or PureWindowsPath(value).drive
            or value.startswith(("/", "\\"))
            or re.match(r"^[A-Za-z]:", value)
        ):
            raise ValueError("knowledge root: wiki path must be relative")
        parts = re.split(r"[\\/]", value)
        if any(part == ".." for part in parts):
            raise ValueError("knowledge root: wiki path cannot contain '..'")
        # Keep path identity deterministic.  In particular, ``a//b`` must not
        # silently normalize to ``a/b`` while a caller is checking a link.
        if any(part == "" for part in parts):
            raise ValueError("knowledge root: wiki path cannot contain empty components")
        candidate = (self.root / Path(*parts)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("knowledge root: wiki path escapes store root") from exc
        return candidate

    def read_relative(self, relative: str | os.PathLike[str]) -> str:
        path = self.resolve_relative(relative)
        return path.read_text(encoding="utf-8")

    def content_sha256(self, relative: str | os.PathLike[str]) -> str:
        return hashlib.sha256(self.resolve_relative(relative).read_bytes()).hexdigest()

    def exists(self, relative: str | os.PathLike[str]) -> bool:
        return self.resolve_relative(relative).exists()

    def write_relative(self, relative: str | os.PathLike[str], content: str) -> str:
        """Write content atomically and return ``created``, ``updated`` or ``unchanged``."""
        path = self.resolve_relative(relative)
        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).digest()
        if path.exists():
            try:
                old = path.read_bytes()
            except OSError:
                old = None
            if old is not None and hashlib.sha256(old).digest() == digest:
                return "unchanged"
            status = "updated"
        else:
            status = "created"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                temp_path = handle.name
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
        return status

    def list_relative(self, directory: str = "") -> list[str]:
        base = self.root if not directory else self.resolve_relative(directory)
        if not base.exists():
            return []
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in base.rglob("*")
            if p.is_file()
        )

    _MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

    def extract_links(self, page: str | os.PathLike[str] | Path) -> list[str]:
        """Return validated relative Markdown links from a Wiki page.

        ``page`` may be either a path to a page on disk or Markdown text.  The
        links are returned exactly as page-relative POSIX paths so callers can
        resolve them against ``page.parent``.  Absolute and parent-traversal
        links are rejected instead of being allowed to escape the Wiki root.
        External URLs and fragment-only links are ignored.
        """
        candidate = Path(page) if isinstance(page, os.PathLike) else None
        if candidate is not None and candidate.exists():
            text = candidate.read_text(encoding="utf-8")
        else:
            text = str(page)

        links: list[str] = []
        seen: set[str] = set()
        for raw in self._MARKDOWN_LINK.findall(text):
            target = raw.strip()
            # Strip optional title and URL fragments.  This Wiki only emits
            # local links, so a scheme/netloc is not a valid page reference.
            if target.startswith(("#", "/", "\\")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0].strip()
            if not target:
                continue
            parts = re.split(r"[\\/]", target)
            if any(part in {"", ".", ".."} for part in parts):
                raise ValueError(f"Wiki link is not a safe relative path: {raw!r}")
            normalized = "/".join(parts)
            if normalized not in seen:
                seen.add(normalized)
                links.append(normalized)
        return links


def render_page(page: WikiPage, pages_by_topic: dict[str, WikiPage]) -> str:
    links: list[str] = []
    for topic in page.related_topics:
        related = pages_by_topic.get(topic)
        if related is None:
            # A malformed/partial seed must not produce a broken Markdown
            # link; graph synchronization applies the same endpoint rule.
            continue
        links.append(f"- [{related.title}]({related.slug}.md)")
    body = "\n".join(
        [
            f"# {page.title}",
            "",
            "Source: built-in seed corpus",
            f"Page ID: {page.page_id}",
            f"Content SHA256: {page.content_sha256}",
            "",
            page.text,
            "",
            "## Related pages",
            *links,
            "",
        ]
    )
    return body


__all__ = ["WikiPage", "WikiStore", "render_page", "sha256_text"]
