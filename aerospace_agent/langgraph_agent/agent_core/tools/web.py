"""Public, read-only web retrieval plus confirmation-gated downloads."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlsplit
from uuid import uuid4

import httpx

from ..models import ToolError, ToolResult


Resolver = Callable[[str], list[str]]


@dataclass(frozen=True)
class SearchProvider:
    name: str
    endpoint: str
    kind: str = "json"
    api_key_env: str | None = None
    enabled: bool = True
    timeout_seconds: float = 20.0


class _SearchHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        classes = set(str(attributes.get("class", "")).split())
        if "result__a" in classes:
            href = str(attributes.get("href", ""))
            parsed = urlsplit(href)
            if parsed.query and "uddg" in parse_qs(parsed.query):
                href = unquote(parse_qs(parsed.query)["uddg"][0])
            self._current = {"url": href, "title": ""}
            self._parts = []
        elif "result__snippet" in classes and self.results:
            self._current = {"url": self.results[-1].get("url", ""), "title": ""}
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._current is None:
            return
        text = " ".join(" ".join(self._parts).split())
        if self._current.get("title"):
            self._current["snippet"] = text
            self.results.append(self._current)
        else:
            self._current["title"] = text
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._parts.append(data)


def _default_resolver(host: str) -> list[str]:
    return sorted({item[4][0] for item in socket.getaddrinfo(host, None)})


class WebService:
    """HTTP(S)-only client that blocks credentials and non-public networks."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        search_endpoint: str | None = None,
        search_providers: list[SearchProvider | dict[str, Any]] | None = None,
        default_search_provider: str | None = None,
        transport: httpx.BaseTransport | None = None,
        resolver: Resolver = _default_resolver,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.root = Path(workspace_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.search_endpoint = search_endpoint
        providers: list[SearchProvider] = []
        for item in search_providers or []:
            if isinstance(item, SearchProvider):
                providers.append(item)
            elif isinstance(item, dict):
                providers.append(SearchProvider(**item))
            else:
                raise TypeError("search_providers must contain SearchProvider or mappings")
        if search_endpoint:
            providers.insert(0, SearchProvider(name="legacy", endpoint=search_endpoint))
        self.search_providers = tuple(providers)
        self.default_search_provider = default_search_provider
        self._resolver = resolver
        self._client = httpx.Client(
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": "zyt-agent-core/1.0"},
        )

    @staticmethod
    def _result(
        *,
        status: str,
        operation_id: str,
        result: dict[str, Any] | None = None,
        code: str | None = None,
        message: str = "",
        recovery_class: str = "read_only",
    ) -> ToolResult:
        return ToolResult(
            status=status,
            result=result or {},
            error=(
                ToolError(
                    code=code,
                    message=message,
                    recoverability=(
                        "retryable" if code in {"timeout", "unavailable", "failed"} else "not_applicable"
                    ),
                )
                if code is not None
                else None
            ),
            audit_id=uuid4().hex,
            operation_id=operation_id,
            recovery_class=recovery_class,
        )

    def _validate_public_url(self, raw_url: str) -> str:
        parsed = urlsplit(str(raw_url))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("only absolute HTTP(S) URLs are allowed")
        if parsed.username is not None or parsed.password is not None:
            raise PermissionError("URL credentials are forbidden")
        try:
            addresses = self._resolver(parsed.hostname)
        except OSError as exc:
            raise ConnectionError(f"URL host resolution failed: {exc}") from exc
        if not addresses:
            raise ConnectionError("URL host has no resolved address")
        for value in addresses:
            address = ipaddress.ip_address(value)
            if not address.is_global:
                raise PermissionError(f"URL resolves to a non-public address: {address}")
        return str(httpx.URL(raw_url))

    def _get(self, url: str, *, max_redirects: int = 5) -> tuple[httpx.Response, str]:
        current = self._validate_public_url(url)
        for _ in range(max_redirects + 1):
            response = self._client.get(current)
            if response.status_code not in {301, 302, 303, 307, 308}:
                response.raise_for_status()
                return response, current
            location = response.headers.get("location")
            if not location:
                raise ValueError("redirect response has no Location")
            current = self._validate_public_url(urljoin(current, location))
        raise ValueError("too many redirects")

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        operation_id: str | None = None,
    ) -> ToolResult:
        operation = operation_id or uuid4().hex
        providers = [item for item in self.search_providers if item.enabled]
        if self.default_search_provider:
            providers.sort(key=lambda item: 0 if item.name == self.default_search_provider else 1)
        if not providers:
            return self._result(
                status="unavailable",
                operation_id=operation,
                code="unavailable",
                message="web search provider is not configured",
            )
        try:
            if not query.strip() or not 1 <= max_results <= 20:
                raise ValueError("query is required and max_results must be between 1 and 20")
            failures: list[str] = []
            for provider in providers:
                try:
                    endpoint = self._validate_public_url(provider.endpoint)
                    headers: dict[str, str] = {}
                    if provider.api_key_env:
                        secret = os.environ.get(provider.api_key_env)
                        if secret:
                            headers["Authorization"] = f"Bearer {secret}"
                    response = self._client.get(
                        endpoint,
                        params={"q": query, "limit": max_results},
                        headers=headers,
                    )
                    response.raise_for_status()
                    if provider.kind == "duckduckgo_html":
                        parser = _SearchHTMLParser()
                        parser.feed(response.text)
                        raw_results = parser.results
                    else:
                        payload = response.json()
                        raw_results = payload.get("results", [])
                        if not isinstance(raw_results, list):
                            raise ValueError("search provider returned an invalid result list")
                    results = []
                    for raw in raw_results:
                        if not isinstance(raw, dict):
                            continue
                        result_url = self._validate_public_url(str(raw.get("url", "")))
                        results.append(
                            {
                                "url": result_url,
                                "title": str(raw.get("title", "")),
                                "snippet": str(raw.get("snippet", "")),
                                "published_at": raw.get("published_at"),
                            }
                        )
                        if len(results) == max_results:
                            break
                    return self._result(
                        status="success",
                        operation_id=operation,
                        result={
                            "query": query,
                            "provider": provider.name,
                            "results": results,
                            "truncated": len(raw_results) > len(results),
                            "source_type": "public_web",
                        },
                    )
                except Exception as exc:
                    failures.append(f"{provider.name}: {exc}")
            raise ConnectionError("; ".join(failures) or "all web search providers failed")
        except (ValueError, PermissionError) as exc:
            return self._result(
                status="invalid_arguments",
                operation_id=operation,
                code="invalid_arguments",
                message=str(exc),
            )
        except Exception as exc:
            return self._result(
                status="failed", operation_id=operation, code="failed", message=str(exc)
            )

    def fetch(
        self,
        url: str,
        *,
        max_bytes: int = 1_000_000,
        hard_limit_bytes: int = 5_000_000,
        operation_id: str | None = None,
    ) -> ToolResult:
        operation = operation_id or uuid4().hex
        try:
            if not 1 <= max_bytes <= hard_limit_bytes:
                raise ValueError("max_bytes must be positive and no larger than hard_limit_bytes")
            response, final_url = self._get(url)
            data = response.content
            if len(data) > hard_limit_bytes:
                raise OSError("web response exceeds hard byte limit")
            shown = data[:max_bytes]
            return self._result(
                status="success",
                operation_id=operation,
                result={
                    "url": final_url,
                    "content": shown.decode("utf-8", errors="replace"),
                    "media_type": response.headers.get("content-type", "application/octet-stream").split(";", 1)[0],
                    "byte_length": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "truncated": len(shown) < len(data),
                    "source_type": "public_web",
                },
            )
        except PermissionError as exc:
            return self._result(
                status="blocked", operation_id=operation, code="unavailable", message=str(exc)
            )
        except ValueError as exc:
            return self._result(
                status="invalid_arguments",
                operation_id=operation,
                code="invalid_arguments",
                message=str(exc),
            )
        except httpx.TimeoutException as exc:
            return self._result(
                status="timeout", operation_id=operation, code="timeout", message=str(exc)
            )
        except Exception as exc:
            return self._result(
                status="failed", operation_id=operation, code="failed", message=str(exc)
            )

    def download(
        self,
        url: str,
        target_path: str | Path,
        *,
        confirmed: bool = False,
        expected_sha256: str | None = None,
        overwrite: bool = False,
        operation_id: str | None = None,
    ) -> ToolResult:
        operation = operation_id or uuid4().hex
        if not confirmed:
            return self._result(
                status="blocked",
                operation_id=operation,
                code="confirmation_required",
                message="web.download requires confirmation",
                recovery_class="manual_recovery",
            )
        temporary: Path | None = None
        try:
            target = Path(target_path)
            if not target.is_absolute():
                target = self.root / target
            target = target.resolve(strict=False)
            try:
                target.relative_to(self.root)
            except ValueError as exc:
                raise PermissionError("download path is outside workspace") from exc
            if target.exists() and not overwrite:
                raise FileExistsError("download target already exists")
            response, final_url = self._get(url)
            data = response.content
            digest = hashlib.sha256(data).hexdigest()
            if expected_sha256 is not None and digest != expected_sha256.lower():
                raise OSError("download SHA-256 does not match expected digest")
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            temporary = None
            return self._result(
                status="success",
                operation_id=operation,
                result={
                    "url": final_url,
                    "target_path": target.relative_to(self.root).as_posix(),
                    "sha256": digest,
                    "byte_length": len(data),
                    "source_type": "public_web",
                },
                recovery_class="manual_recovery",
            )
        except PermissionError as exc:
            return self._result(
                status="blocked",
                operation_id=operation,
                code="path_outside_workspace",
                message=str(exc),
                recovery_class="manual_recovery",
            )
        except FileExistsError as exc:
            return self._result(
                status="blocked",
                operation_id=operation,
                code="conflict",
                message=str(exc),
                recovery_class="manual_recovery",
            )
        except Exception as exc:
            return self._result(
                status="failed",
                operation_id=operation,
                code="failed",
                message=str(exc),
                recovery_class="manual_recovery",
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


__all__ = ["SearchProvider", "WebService"]
