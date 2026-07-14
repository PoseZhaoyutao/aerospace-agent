"""Deterministic JSON/HTML export for the Wiki knowledge graph."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class GraphExportResult:
    """Paths produced by :func:`export_graph`.

    The result remains path-like for callers of the original API (which
    returned the HTML ``Path``), while exposing the sibling JSON path needed by
    integrations that consume the deterministic payload.
    """

    html_path: Path
    json_path: Path

    def __fspath__(self) -> str:
        return os.fspath(self.html_path)

    def __str__(self) -> str:
        return str(self.html_path)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GraphExportResult):
            return self.html_path == other.html_path and self.json_path == other.json_path
        try:
            return self.html_path == Path(other)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __getattr__(self, name: str) -> Any:
        # Preserve common Path conveniences (``read_text``, ``parent``, ...)
        # without requiring every legacy caller to unwrap ``html_path``.
        return getattr(self.html_path, name)


def _payload(source: Any) -> tuple[dict[str, Any], Path | None]:
    workspace = getattr(source, "workspace", None)
    graph = getattr(getattr(source, "rag", None), "kb", None)
    graph = getattr(graph, "knowledge_graph", None)
    pages = list(getattr(source, "pages", ()))
    if graph is None:
        graph = source

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    if hasattr(graph, "nodes"):
        for node_id in sorted(graph.nodes):
            node = graph.nodes[node_id]
            metadata = dict(node.get("metadata", {}))
            item = {
                "id": node_id,
                "type": node.get("type", "concept"),
                "content": node.get("content", ""),
                "metadata": metadata,
            }
            if metadata.get("page_path"):
                item["page_path"] = metadata["page_path"]
            if metadata.get("content_sha256"):
                item["content_sha256"] = metadata["content_sha256"]
            if metadata.get("links"):
                item["links"] = list(metadata["links"])
            nodes.append(item)
        for src in sorted(graph.adj):
            for relation in sorted(graph.adj[src]):
                for dst, weight in sorted(graph.adj[src][relation], key=lambda item: item[0]):
                    edges.append(
                        {
                            "source": src,
                            "target": dst,
                            "src": src,
                            "dst": dst,
                            "relation": relation,
                            "type": relation,
                            "weight": weight,
                        }
                    )

    # A service may be exported before a RAG rebuild. Ensure wiki nodes and
    # relation edges are still represented from source pages.
    existing_nodes = {node["id"] for node in nodes}
    existing_edges = {(edge["source"], edge["target"], edge["relation"]) for edge in edges}
    by_topic = {page.topic: page for page in pages}
    by_path = {page.page_path: page for page in pages}

    def page_links(page: Any) -> list[str]:
        renderer = getattr(source, "render_page", None)
        wiki = getattr(source, "wiki", None)
        extractor = getattr(wiki, "extract_links", None)
        if callable(renderer) and callable(extractor):
            return extractor(renderer(page))
        return [
            by_topic[t].slug + ".md"
            for t in page.related_topics
            if t in by_topic
        ]

    for page in sorted(pages, key=lambda item: item.topic):
        node_id = f"wiki:{page.page_id}"
        links = page_links(page)
        link_paths: list[str] = []
        link_targets: list[Any] = []
        for relative in links:
            target_path = (PurePosixPath(page.page_path).parent / relative).as_posix()
            target_page = by_path.get(target_path)
            if target_page is not None:
                link_paths.append(target_page.page_path)
                link_targets.append(target_page)
        if node_id not in existing_nodes:
            nodes.append(
                {
                    "id": node_id,
                    "type": "wiki_page",
                    "content": page.text,
                    "page_path": page.page_path,
                    "content_sha256": page.content_sha256,
                    "links": link_paths,
                    "metadata": {
                        "page_id": page.page_id,
                        "page_path": page.page_path,
                        "content_sha256": page.content_sha256,
                        "title": page.title,
                        "aliases": [page.topic, page.slug, page.title, page.page_id],
                        "links": link_paths,
                    },
                }
            )
        for target_page in link_targets:
            target = f"wiki:{target_page.page_id}"
            key = (node_id, target, "related_to")
            if key not in existing_edges:
                edges.append(
                    {
                        "source": node_id,
                        "target": target,
                        "src": node_id,
                        "dst": target,
                        "relation": "related_to",
                        "type": "related_to",
                        "weight": 1.0,
                    }
                )

    nodes.sort(key=lambda item: item["id"])
    edges.sort(key=lambda item: (item["source"], item["target"], item["relation"]))
    return {"nodes": nodes, "edges": edges}, Path(workspace) if workspace else None


def export_graph(
    source: Any = None,
    output_path: str | os.PathLike[str] | None = None,
    *,
    graph: Any = None,
    html_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Write deterministic ``graph.html`` and sibling ``graph.json``."""
    if source is None:
        source = graph
    if output_path is None:
        output_path = html_path
    if source is None or output_path is None:
        raise TypeError("export_graph requires a graph/service and output path")
    html_path = Path(output_path)
    if html_path.suffix.lower() != ".html":
        html_path = html_path.with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    payload, workspace = _payload(source)
    json_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    json_path = html_path.with_suffix(".json")
    json_path.write_text(json_text, encoding="utf-8")

    links: list[str] = []
    for node in payload["nodes"]:
        page_path = node.get("page_path")
        if not page_path:
            page_path = node.get("metadata", {}).get("page_path")
        if not page_path:
            continue
        href = page_path
        if workspace is not None:
            href = os.path.relpath(workspace / page_path, html_path.parent).replace(os.sep, "/")
        title = node.get("metadata", {}).get("title", node["id"])
        links.append(f'<li><a href="{html.escape(href, quote=True)}">{html.escape(title)}</a></li>')

    html_text = "\n".join(
        [
            "<!doctype html>",
            "<meta charset=\"utf-8\">",
            "<title>Knowledge graph</title>",
            "<h1>Knowledge graph</h1>",
            "<ul>",
            *sorted(set(links)),
            "</ul>",
            '<script id="graph-data" type="application/json">',
            json_text,
            "</script>",
            "",
        ]
    )
    html_path.write_text(html_text, encoding="utf-8")
    return GraphExportResult(html_path=html_path, json_path=json_path)


__all__ = ["GraphExportResult", "export_graph"]
