"""Stateful read-only browser facade built on the public WebService."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from uuid import uuid4

from ..models import ToolError, ToolResult
from .web import WebService


def build_playwright_screenshot_adapter(
    *,
    timeout_ms: int = 30_000,
) -> Callable[[str, Path], None] | None:
    """Return a headless Playwright screenshot adapter when installed.

    Playwright and its browser binaries are optional.  Returning ``None`` is
    intentional: the catalog then reports a structured ``unavailable`` tool
    result instead of pretending screenshots are executable.
    """

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    def capture(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.screenshot(path=str(target), full_page=True)
            finally:
                browser.close()

    return capture


def build_playwright_navigation_adapter(
    *,
    timeout_ms: int = 30_000,
) -> Callable[[str], tuple[str, str]] | None:
    """Return an optional rendered-page adapter for JavaScript-heavy sites."""

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    def navigate(url: str) -> tuple[str, str]:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                final_url = str(getattr(response, "url", None) or page.url or url)
                return final_url, page.content()
            finally:
                browser.close()

    return navigate


class _PageParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[dict[str, Any]] = []
        self._in_title = False
        self._link_url: str | None = None
        self._link_text: list[str] = []
        self._suppressed = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag in {"script", "style", "form", "input", "button"}:
            self._suppressed += 1
        if tag == "title":
            self._in_title = True
        if tag == "a" and self._suppressed == 0 and attributes.get("href"):
            self._link_url = urljoin(self.base_url, attributes["href"])
            self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._link_url is not None:
            self.links.append(
                {
                    "link_id": len(self.links),
                    "text": " ".join(self._link_text).strip(),
                    "url": self._link_url,
                }
            )
            self._link_url = None
            self._link_text = []
        if tag in {"script", "style", "form", "input", "button"} and self._suppressed:
            self._suppressed -= 1

    def handle_data(self, data: str) -> None:
        if self._suppressed:
            return
        value = " ".join(data.split())
        if not value:
            return
        self.text_parts.append(value)
        if self._in_title:
            self.title_parts.append(value)
        if self._link_url is not None:
            self._link_text.append(value)


class BrowserService:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        web_service: WebService,
        screenshot_adapter: Callable[[str, Path], None] | None = None,
        navigation_adapter: Callable[[str], tuple[str, str]] | None = None,
    ) -> None:
        if not isinstance(web_service, WebService):
            raise TypeError("web_service must be WebService")
        self.root = Path(workspace_root).resolve()
        self._web = web_service
        self._screenshot_adapter = screenshot_adapter
        self._navigation_adapter = navigation_adapter
        self._pages: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _result(status: str, operation: str, *, result=None, code=None, message="", recovery="read_only"):
        return ToolResult(
            status=status,
            result=result or {},
            error=(
                ToolError(code=code, message=message, recoverability="not_applicable")
                if code is not None
                else None
            ),
            audit_id=uuid4().hex,
            operation_id=operation,
            recovery_class=recovery,
        )

    def open(self, url: str, *, operation_id: str | None = None) -> ToolResult:
        operation = operation_id or uuid4().hex
        if self._navigation_adapter is not None:
            try:
                validated = self._web._validate_public_url(url)
                final_url, html = self._navigation_adapter(validated)
                # Rendered navigation may follow redirects. Re-validate the
                # adapter's final URL so a public entry point cannot pivot to
                # localhost, link-local, or private network targets.
                final_url = self._web._validate_public_url(final_url)
            except PermissionError as exc:
                return self._result(
                    "blocked",
                    operation,
                    code="unavailable",
                    message=str(exc),
                    recovery="manual_recovery",
                )
            except Exception as exc:
                return self._result(
                    "failed",
                    operation,
                    code="failed",
                    message=str(exc),
                )
        else:
            fetched = self._web.fetch(url, max_bytes=2_000_000, hard_limit_bytes=2_000_000)
            if fetched.status != "success":
                data = fetched.model_dump(mode="python")
                data.update({"audit_id": uuid4().hex, "operation_id": operation})
                return ToolResult.model_validate(data)
            final_url = str(fetched.result["url"])
            html = str(fetched.result["content"])
        parser = _PageParser(final_url)
        parser.feed(html)
        page_id = uuid4().hex
        page = {
            "page_id": page_id,
            "url": final_url,
            "title": " ".join(parser.title_parts),
            "text": " ".join(parser.text_parts),
            "links": parser.links,
        }
        self._pages[page_id] = page
        return self._result(
            "success",
            operation,
            result={
                "page_id": page_id,
                "url": final_url,
                "title": page["title"],
                "links": page["links"],
                "source_type": "public_web",
            },
        )

    def follow_link(
        self, page_id: str, link_id: int, *, operation_id: str | None = None
    ) -> ToolResult:
        operation = operation_id or uuid4().hex
        page = self._pages.get(page_id)
        if page is None or not isinstance(link_id, int) or not 0 <= link_id < len(page["links"]):
            return self._result(
                "invalid_arguments",
                operation,
                code="invalid_arguments",
                message="page or link does not exist",
            )
        return self.open(page["links"][link_id]["url"], operation_id=operation)

    def extract(
        self, page_id: str, *, max_chars: int = 100_000, operation_id: str | None = None
    ) -> ToolResult:
        operation = operation_id or uuid4().hex
        page = self._pages.get(page_id)
        if page is None or max_chars < 1:
            return self._result(
                "invalid_arguments",
                operation,
                code="invalid_arguments",
                message="page does not exist or max_chars is invalid",
            )
        text = re.sub(r"\s+", " ", str(page["text"])).strip()
        return self._result(
            "success",
            operation,
            result={
                "page_id": page_id,
                "url": page["url"],
                "text": text[:max_chars],
                "truncated": len(text) > max_chars,
                "source_type": "public_web",
            },
        )

    def screenshot(
        self,
        page_id: str,
        target_path: str | Path,
        *,
        operation_id: str | None = None,
    ) -> ToolResult:
        operation = operation_id or uuid4().hex
        if self._screenshot_adapter is None:
            return self._result(
                "unavailable",
                operation,
                code="unavailable",
                message="browser screenshot adapter is unavailable",
            )
        page = self._pages.get(page_id)
        if page is None:
            return self._result(
                "invalid_arguments", operation, code="invalid_arguments", message="page does not exist"
            )
        target = Path(target_path)
        if not target.is_absolute():
            target = self.root / target
        target = target.resolve(strict=False)
        try:
            target.relative_to(self.root)
        except ValueError:
            return self._result(
                "blocked",
                operation,
                code="path_outside_workspace",
                message="screenshot path is outside workspace",
                recovery="manual_recovery",
            )
        try:
            self._screenshot_adapter(page["url"], target)
            if not target.is_file():
                raise OSError("screenshot adapter did not create the target")
            data = target.read_bytes()
            return self._result(
                "success",
                operation,
                result={
                    "page_id": page_id,
                    "path": target.relative_to(self.root).as_posix(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "byte_length": len(data),
                },
                recovery="manual_recovery",
            )
        except Exception as exc:
            return self._result(
                "failed",
                operation,
                code="failed",
                message=str(exc),
                recovery="manual_recovery",
            )


__all__ = [
    "BrowserService",
    "build_playwright_navigation_adapter",
    "build_playwright_screenshot_adapter",
]
