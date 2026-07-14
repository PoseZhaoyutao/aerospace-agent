from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

from aerospace_agent.langgraph_agent.agent_core.tools.web import WebService


def _resolver(host: str):
    return ["93.184.216.34"]


def test_search_and_fetch_are_bounded_and_marked_as_public_web(tmp_path) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"url": "https://example.test/a", "title": "A", "snippet": "one"},
                        {"url": "https://example.test/b", "title": "B", "snippet": "two"},
                    ]
                },
            )
        return httpx.Response(200, content=b"public body", headers={"content-type": "text/plain"})

    service = WebService(
        tmp_path,
        search_endpoint="https://search.test/search",
        transport=httpx.MockTransport(transport),
        resolver=_resolver,
    )

    searched = service.search("orbit", max_results=1, operation_id="search")
    fetched = service.fetch("https://example.test/a", max_bytes=6, operation_id="fetch")

    assert searched.status == "success"
    assert searched.result["source_type"] == "public_web"
    assert searched.result["truncated"] is True
    assert len(searched.result["results"]) == 1
    assert fetched.result["content"] == "public"
    assert fetched.result["truncated"] is True
    assert fetched.result["sha256"] == hashlib.sha256(b"public body").hexdigest()


def test_private_hosts_credentials_and_private_redirects_are_blocked(tmp_path) -> None:
    calls = []

    def redirect(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    service = WebService(
        tmp_path,
        transport=httpx.MockTransport(redirect),
        resolver=lambda host: ["127.0.0.1"]
        if host in {"localhost", "127.0.0.1"}
        else _resolver(host),
    )

    direct = service.fetch("http://localhost/private")
    credentialed = service.fetch("https://user:pass@example.test/")
    redirected = service.fetch("https://example.test/start")

    assert direct.status == "blocked"
    assert credentialed.status == "blocked"
    assert redirected.status == "blocked"
    assert len(calls) == 1


def test_download_requires_confirmation_and_expected_hash_before_atomic_replace(tmp_path) -> None:
    payload = b"downloaded"
    service = WebService(
        tmp_path,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=payload)),
        resolver=_resolver,
    )

    blocked = service.download("https://example.test/a", "downloads/a.bin")
    mismatch = service.download(
        "https://example.test/a",
        "downloads/a.bin",
        confirmed=True,
        expected_sha256="0" * 64,
    )
    existed_after_mismatch = (tmp_path / "downloads/a.bin").exists()
    success = service.download(
        "https://example.test/a",
        "downloads/a.bin",
        confirmed=True,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert blocked.status == "blocked"
    assert mismatch.status == "failed"
    assert existed_after_mismatch is False
    assert (tmp_path / "downloads/a.bin").read_bytes() == payload
    assert success.recovery_class == "manual_recovery"


def test_download_path_escape_and_response_size_are_blocked(tmp_path) -> None:
    service = WebService(
        tmp_path,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"12345")),
        resolver=_resolver,
    )

    escaped = service.download(
        "https://example.test/a", str(tmp_path.parent / "escape.bin"), confirmed=True
    )
    oversized = service.fetch("https://example.test/a", max_bytes=3, hard_limit_bytes=4)

    assert escaped.status == "blocked"
    assert oversized.status == "failed"


def test_search_rejects_unconfigured_provider(tmp_path) -> None:
    service = WebService(
        tmp_path,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
        resolver=_resolver,
    )
    result = service.search("orbit")
    assert result.status == "unavailable"


def test_search_falls_back_to_next_configured_provider(tmp_path) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.host == "primary.test":
            return httpx.Response(503, content=b"down")
        return httpx.Response(
            200,
            json={"results": [{"url": "https://example.test/fallback", "title": "Fallback"}]},
        )

    service = WebService(
        tmp_path,
        search_providers=[
            {"name": "primary", "endpoint": "https://primary.test/search"},
            {"name": "fallback", "endpoint": "https://fallback.test/search"},
        ],
        transport=httpx.MockTransport(transport),
        resolver=_resolver,
    )

    result = service.search("orbit")

    assert result.status == "success"
    assert result.result["provider"] == "fallback"
    assert result.result["results"][0]["title"] == "Fallback"


def test_duckduckgo_html_provider_extracts_public_result_links(tmp_path) -> None:
    html = b'''
    <a class="result__a" href="https://example.test/a">Orbit result</a>
    <a class="result__snippet">Public snippet</a>
    '''
    service = WebService(
        tmp_path,
        search_providers=[
            {
                "name": "duck",
                "endpoint": "https://html.duck.test/html/",
                "kind": "duckduckgo_html",
            }
        ],
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=html)),
        resolver=_resolver,
    )

    result = service.search("orbit")

    assert result.status == "success"
    assert result.result["results"] == [
        {
            "url": "https://example.test/a",
            "title": "Orbit result",
            "snippet": "Public snippet",
            "published_at": None,
        }
    ]
