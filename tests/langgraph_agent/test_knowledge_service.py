from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.services.graph_export import export_graph
from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService
from aerospace_agent.langgraph_agent.services.wiki import WikiStore


def test_seed_wiki_is_idempotent_and_search_persists(tmp_path: Path):
    service = KnowledgeService(workspace=tmp_path)
    first = service.initialize_seed_wiki()
    assert first.created == 6
    pages = sorted((tmp_path / "knowledge/orbital-dynamics").glob("*.md"))
    assert len(pages) == 6
    index_before = (tmp_path / "knowledge/index.md").read_bytes()
    log_before = (tmp_path / "knowledge/log.md").read_bytes()
    assert log_before.count(b"| ingest |") == 6

    second = service.initialize_seed_wiki()
    assert second.created == 0
    assert (tmp_path / "knowledge/index.md").read_bytes() == index_before
    assert (tmp_path / "knowledge/log.md").read_bytes() == log_before

    results = service.search("two-body central gravity", top_k=3)
    assert results
    item = results[0]
    assert item.page_path.endswith("two-body-orbital-dynamics.md")
    assert item.chunk_id
    assert 0 <= item.score <= 1
    assert item.metadata["content_sha256"]
    assert item.metadata["page_id"].startswith("seed:")
    assert item.metadata["page_path"] == item.page_path

    reloaded = KnowledgeService(workspace=tmp_path)
    again = reloaded.search("two-body central gravity", top_k=3)
    assert again and again[0].model_dump() == item.model_dump()


def test_wiki_store_rejects_traversal_and_atomic_status(tmp_path: Path):
    store = WikiStore(tmp_path / "knowledge")
    for bad in ("/etc/passwd", "../escape.md", "a/../../escape.md", "C:\\escape.md"):
        with pytest.raises(ValueError):
            store.resolve_relative(bad)
    assert store.write_relative("x.md", "hello") == "created"
    assert store.write_relative("x.md", "hello") == "unchanged"
    assert store.write_relative("x.md", "changed") == "updated"


def test_graph_export_is_deterministic(tmp_path: Path):
    service = KnowledgeService(workspace=tmp_path)
    service.initialize_seed_wiki()
    html_path = tmp_path / "reports" / "graph.html"
    result = export_graph(service, html_path)
    assert result == html_path
    payload = json.loads(html_path.with_suffix(".json").read_text(encoding="utf-8"))
    pages = [n for n in payload["nodes"] if n["type"] == "wiki_page"]
    assert len(pages) == 6
    assert all(n["id"].startswith("wiki:") for n in pages)
    assert any(e["type"] == "related_to" for e in payload["edges"])
    assert "two-body-orbital-dynamics.md" in html_path.read_text(encoding="utf-8")

