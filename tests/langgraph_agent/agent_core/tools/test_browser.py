from __future__ import annotations

from pathlib import Path

import httpx

from aerospace_agent.langgraph_agent.agent_core.tools.browser import BrowserService
from aerospace_agent.langgraph_agent.agent_core.tools.web import WebService


def _web(tmp_path: Path) -> WebService:
    pages = {
        "/": b"<html><head><title>Home</title></head><body><p>Hello orbit</p><a href='/next'>Next</a><form></form></body></html>",
        "/next": b"<html><head><title>Next</title></head><body><main>Second page</main></body></html>",
    }
    return WebService(
        tmp_path,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=pages[request.url.path], headers={"content-type": "text/html"}
            )
        ),
        resolver=lambda _host: ["93.184.216.34"],
    )


def test_open_follow_and_extract_are_read_only(tmp_path) -> None:
    browser = BrowserService(tmp_path, web_service=_web(tmp_path))

    opened = browser.open("https://example.test/")
    page_id = opened.result["page_id"]
    followed = browser.follow_link(page_id, 0)
    extracted = browser.extract(followed.result["page_id"])

    assert opened.status == "success"
    assert opened.result["title"] == "Home"
    assert opened.result["links"] == [
        {"link_id": 0, "text": "Next", "url": "https://example.test/next"}
    ]
    assert extracted.result["text"] == "Next Second page"
    assert not hasattr(browser, "login")
    assert not hasattr(browser, "submit_form")
    assert not hasattr(browser, "upload")


def test_unknown_page_or_link_is_structured_invalid_arguments(tmp_path) -> None:
    browser = BrowserService(tmp_path, web_service=_web(tmp_path))
    assert browser.extract("missing").status == "invalid_arguments"
    opened = browser.open("https://example.test/")
    assert browser.follow_link(opened.result["page_id"], 9).status == "invalid_arguments"


def test_screenshot_is_unavailable_without_adapter_and_workspace_bounded_with_one(tmp_path) -> None:
    browser = BrowserService(tmp_path, web_service=_web(tmp_path))
    opened = browser.open("https://example.test/")
    assert browser.screenshot(opened.result["page_id"], "shots/a.png").status == "unavailable"

    def capture(_url: str, target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"png")

    enabled = BrowserService(tmp_path, web_service=_web(tmp_path), screenshot_adapter=capture)
    page = enabled.open("https://example.test/")
    result = enabled.screenshot(page.result["page_id"], "shots/a.png")
    escaped = enabled.screenshot(page.result["page_id"], str(tmp_path.parent / "bad.png"))

    assert result.status == "success"
    assert (tmp_path / "shots/a.png").read_bytes() == b"png"
    assert escaped.status == "blocked"


def test_open_can_use_optional_rendered_browser_adapter(tmp_path) -> None:
    browser = BrowserService(
        tmp_path,
        web_service=_web(tmp_path),
        navigation_adapter=lambda url: (
            url,
            "<html><head><title>Rendered</title></head><body><main>JS content</main></body></html>",
        ),
    )

    opened = browser.open("https://example.test/")
    extracted = browser.extract(opened.result["page_id"])

    assert opened.result["title"] == "Rendered"
    assert extracted.result["text"] == "Rendered JS content"


def test_rendered_navigation_revalidates_final_redirect_target(tmp_path) -> None:
    web = WebService(
        tmp_path,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, text="unused")),
        resolver=lambda host: ["127.0.0.1"] if host == "private.test" else ["93.184.216.34"],
    )
    browser = BrowserService(
        tmp_path,
        web_service=web,
        navigation_adapter=lambda _url: (
            "http://private.test/internal",
            "<html><body>should not be accepted</body></html>",
        ),
    )

    result = browser.open("https://example.test/")

    assert result.status == "blocked"
