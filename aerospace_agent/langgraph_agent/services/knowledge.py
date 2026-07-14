"""Markdown Wiki, persistent RAG, and evidence service."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from aerospace_agent.rag.aerospace_rag import AerospaceRAG
from aerospace_agent.rag.orbit_dynamics import ORBIT_DYNAMICS_SEED_DOCUMENTS

from ..schema import EvidenceItem
from .wiki import WikiPage, WikiStore, render_page, sha256_text


@dataclass(frozen=True)
class KnowledgeSummary:
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    indexed: int = 0
    pages: int = 0
    paths: tuple[str, ...] = field(default_factory=tuple)
    status: str = "ok"
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> int:
        return self.created + self.updated

    @property
    def indexed_count(self) -> int:
        return self.indexed

    def __getitem__(self, key: str) -> Any:
        return self.model_dump()[key]

    def model_dump(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "indexed": self.indexed,
            "indexed_count": self.indexed,
            "pages": self.pages,
            "paths": list(self.paths),
            "status": self.status,
            "errors": list(self.errors),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def model_dump_json(self, **kwargs: Any) -> str:
        """Pydantic-compatible JSON helper used by bootstrap scripts."""
        options = {"ensure_ascii": False, "sort_keys": True, "indent": 2}
        options.update(kwargs)
        return json.dumps(self.model_dump(), **options)


class KnowledgeService:
    """Owns the Wiki as source of truth and rebuilds derived indexes."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        wiki_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
    ):
        self.workspace = Path(workspace).resolve()
        self.wiki_root = Path(wiki_dir).resolve() if wiki_dir else self.workspace / "knowledge"
        self.data_dir = Path(data_dir).resolve() if data_dir else self.workspace / "data/langgraph/rag"
        self.store = WikiStore(self.wiki_root)
        # Public Wiki facade name used by graph/evolution integrations.
        self.wiki = self.store
        self.rag = AerospaceRAG(
            data_dir=str(self.data_dir),
            autoload=True,
            auto_default_knowledge=False,
        )
        self._pages: list[WikiPage] = self._seed_pages()
        # Evolution transactions may add small, user-authored Wiki pages.
        # Track only paths explicitly supplied by an evolution transaction;
        # never crawl/import an arbitrary book or directory.
        self._evolved_wiki_paths: set[str] = set()
        for text, metadata in getattr(self.rag.kb, "documents", ()):
            page_path = str((metadata or {}).get("page_path", ""))
            if str((metadata or {}).get("page_id", "")).startswith("evolved:") and page_path:
                self._evolved_wiki_paths.add(page_path)

    @staticmethod
    def _seed_pages() -> list[WikiPage]:
        out: list[WikiPage] = []
        for seed in ORBIT_DYNAMICS_SEED_DOCUMENTS:
            out.append(
                WikiPage(
                    topic=seed["topic"],
                    slug=seed.get("slug", seed["topic"].replace("_", "-")),
                    title=seed["title"],
                    text=seed["text"],
                    related_topics=tuple(seed.get("related_topics", ())),
                )
            )
        return out

    @property
    def pages(self) -> tuple[WikiPage, ...]:
        return tuple(self._pages)

    @property
    def pages_by_topic(self) -> dict[str, WikiPage]:
        return {page.topic: page for page in self._pages}

    def render_page(self, page: WikiPage) -> str:
        return render_page(page, self.pages_by_topic)

    def initialize_seed_wiki(self) -> KnowledgeSummary:
        """Render all six seeds; only then rebuild the derived RAG/graph."""
        pages_by_topic = self.pages_by_topic
        statuses: list[str] = []
        # Any write exception intentionally propagates before RAG rebuild.
        for page in self._pages:
            relative = f"orbital-dynamics/{page.slug}.md"
            statuses.append(self.store.write_relative(relative, render_page(page, pages_by_topic)))

        self._write_index()
        self._ensure_log()
        changed = [
            (status, page)
            for status, page in zip(statuses, self._pages)
            if status in {"created", "updated"}
        ]
        if changed:
            self._append_log(changed)

        # Build a fresh derived RAG only after every source write succeeds.
        self.rag = self._rebuild_rag()
        return KnowledgeSummary(
            created=statuses.count("created"),
            updated=statuses.count("updated"),
            unchanged=statuses.count("unchanged"),
            indexed=len(self._pages),
            pages=len(self._pages),
            paths=tuple(page.page_path for page in self._pages),
        )

    def _write_index(self) -> None:
        lines = [
            "# Orbital Dynamics Knowledge Wiki",
            "",
            "Source: built-in seed corpus",
            "",
            "## Pages",
        ]
        for page in sorted(self._pages, key=lambda item: item.slug):
            relative = f"orbital-dynamics/{page.slug}.md"
            # Re-read the rendered page so the index reflects the Wiki source
            # of truth rather than duplicating seed metadata in Python.
            rendered = self.store.read_relative(relative) if self.store.exists(relative) else ""
            title, body = self._parse_rendered_page(rendered, fallback=page)
            lines.append(
                f"- [{title}]({relative}): {self._summary_sentence(body)}"
            )
        lines.append("")
        self.store.write_relative("index.md", "\n".join(lines))

    @staticmethod
    def _parse_rendered_page(rendered: str, *, fallback: WikiPage) -> tuple[str, str]:
        """Extract title and seed body from deterministic rendered Markdown."""
        lines = rendered.splitlines()
        title = fallback.title
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip() or title
        try:
            marker = next(i for i, line in enumerate(lines) if line.startswith("Content SHA256:"))
            body_start = marker + 1
            while body_start < len(lines) and not lines[body_start].strip():
                body_start += 1
        except StopIteration:
            body_start = 0
        try:
            body_end = next(i for i in range(body_start, len(lines)) if lines[i].strip() == "## Related pages")
        except StopIteration:
            body_end = len(lines)
        body = " ".join(line.strip() for line in lines[body_start:body_end] if line.strip())
        return title, body or fallback.text

    @staticmethod
    def _summary_sentence(body: str) -> str:
        text = " ".join(body.split())
        if not text:
            return "(no summary)"
        # Keep exactly one sentence where possible.  The seed corpus is
        # English, but include common CJK terminators for future seeds.
        match = re.search(r".*?[.!?。！？](?:\s|$)", text)
        if match:
            return match.group(0).strip()
        return text.rstrip(".") + "."

    def _ensure_log(self) -> None:
        if not self.store.exists("log.md"):
            # The first write is itself atomic; subsequent updates append only.
            self.store.write_relative("log.md", "# Knowledge ingest log\n\n")

    def _append_log(self, changed: list[tuple[str, WikiPage]]) -> None:
        from datetime import date

        path = self.store.resolve_relative("log.md")
        with path.open("a", encoding="utf-8", newline="") as handle:
            for status, page in changed:
                action = "ingest" if status == "created" else "update"
                handle.write(
                    f"| {date.today().isoformat()} | {action} | {page.page_path} | "
                    f"{page.page_id} | {page.content_sha256} |\n"
                )

    def _rebuild_rag(self, evolved_paths: Iterable[str] = ()) -> AerospaceRAG:
        rag = AerospaceRAG(
            data_dir=str(self.data_dir),
            autoload=False,
            auto_default_knowledge=False,
        )
        pages_by_topic = self.pages_by_topic
        for page in self._pages:
            # Keep one deterministic chunk per seed.  Use kb.index_text directly
            # because AerospaceRAG.index historically discards metadata kwargs.
            relative = f"orbital-dynamics/{page.slug}.md"
            if self.store.exists(relative):
                rendered = self.store.read_relative(relative)
                if f"Content SHA256: {page.content_sha256}" in rendered:
                    title, chunk = page.title, page.text
                else:
                    title, chunk = self._parse_rendered_page(rendered, fallback=page)
            else:
                title, chunk = page.title, page.text
            metadata = {
                "page_id": page.page_id,
                "page_path": page.page_path,
                "chunk_id": f"{page.page_id}:chunk-0",
                "content_sha256": sha256_text(chunk),
                "title": title,
                "aliases": [page.topic, page.slug, page.title, page.page_id],
                "links": [
                    pages_by_topic[t].page_path
                    for t in page.related_topics
                    if t in pages_by_topic
                ],
            }
            rag.kb.index_text(chunk, source=f"wiki:{page.page_id}", metadata=metadata)

        # Add only transaction-declared Wiki files.  This deliberately avoids
        # indexing the whole ``knowledge`` tree (which could contain a book).
        paths = set(self._evolved_wiki_paths)
        paths.update(str(path).replace("\\", "/") for path in evolved_paths)
        for relative in sorted(paths):
            path = (self.workspace / Path(relative)).resolve()
            try:
                path.relative_to(self.workspace / "knowledge")
            except ValueError:
                continue
            if not path.is_file() or path.suffix.lower() not in {".md", ".markdown", ".txt"}:
                self._evolved_wiki_paths.discard(relative)
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            page_id = f"evolved:{relative}"
            links = []
            try:
                links = self.wiki.extract_links(text)
            except ValueError:
                # Unsafe links are ignored by the derived graph; source Wiki
                # validation remains the caller's responsibility.
                links = []
            metadata = {
                "page_id": page_id,
                "page_path": relative,
                "chunk_id": f"{page_id}:chunk-0",
                "content_sha256": sha256_text(text),
                "title": next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), Path(relative).stem),
                "aliases": [relative, Path(relative).stem],
                "links": links,
            }
            rag.kb.index_text(text, source=f"wiki:{page_id}", metadata=metadata)
            self._evolved_wiki_paths.add(relative)

        rag.kb.vector_store.reindex()
        self._add_wiki_graph(rag)
        for relative in sorted(paths):
            page_id = f"evolved:{relative}"
            matching = next((item for item in rag.kb.documents if item[1].get("page_id") == page_id), None)
            if matching is not None:
                rag.kb.knowledge_graph.add_node(
                    f"wiki:{page_id}", "wiki_page", matching[0], dict(matching[1])
                )
                for target in matching[1].get("links", []):
                    target_path = (Path(relative).parent / target).as_posix()
                    target_id = next(
                        (meta.get("page_id") for _text, meta in rag.kb.documents
                         if meta.get("page_path") == target_path),
                        None,
                    )
                    if target_id:
                        rag.kb.knowledge_graph.add_edge(
                            f"wiki:{page_id}", f"wiki:{target_id}", "related_to"
                        )
        rag.save()
        return rag

    def rebuild_derived(self, affected_paths: Iterable[str] = ()) -> KnowledgeSummary:
        """Rebuild RAG and graph from seeds plus declared evolved Wiki files."""
        normalized = [str(path).replace("\\", "/") for path in affected_paths]
        self._write_index()
        self.rag = self._rebuild_rag(normalized)
        return KnowledgeSummary(
            indexed=len(self._pages) + len(self._evolved_wiki_paths),
            pages=len(self._pages),
            paths=tuple(page.page_path for page in self._pages) + tuple(sorted(self._evolved_wiki_paths)),
        )

    # Explicit aliases make the rebuild boundary discoverable to transaction
    # integrations while keeping one implementation of derived-state logic.
    rebuild_index = rebuild_derived
    rebuild_graph = rebuild_derived

    def _add_wiki_graph(self, rag: AerospaceRAG) -> None:
        graph = rag.kb.knowledge_graph
        pages_by_topic = self.pages_by_topic
        pages_by_path = {page.page_path: page for page in self._pages}
        for page in self._pages:
            node_id = f"wiki:{page.page_id}"
            relative = f"orbital-dynamics/{page.slug}.md"
            if self.store.exists(relative):
                rendered = self.store.read_relative(relative)
                if f"Content SHA256: {page.content_sha256}" in rendered:
                    title, text, content_sha = page.title, page.text, page.content_sha256
                else:
                    title, text = self._parse_rendered_page(rendered, fallback=page)
                    content_sha = sha256_text(text)
            else:
                title, text, content_sha = page.title, page.text, page.content_sha256
            graph.add_node(
                node_id,
                "wiki_page",
                text,
                {
                    "page_id": page.page_id,
                    "page_path": page.page_path,
                    "content_sha256": content_sha,
                    "title": title,
                    "aliases": [page.topic, page.slug, page.title, page.page_id],
                    "links": [
                        pages_by_topic[t].page_path
                        for t in page.related_topics
                        if t in pages_by_topic
                    ],
                },
            )
        for page in self._pages:
            src = f"wiki:{page.page_id}"
            # Derive graph relations from the rendered source-of-truth page,
            # not from the seed Python structure.  This keeps graph edges in
            # lockstep with the links a reader can actually follow.
            for relative in self.wiki.extract_links(self.render_page(page)):
                target_path = (PurePosixPath(page.page_path).parent / relative).as_posix()
                related = pages_by_path.get(target_path)
                if related is not None:
                    graph.add_edge(src, f"wiki:{related.page_id}", "related_to")

    def search(self, query: str, top_k: int = 5) -> list[EvidenceItem]:
        """Return typed, bounded evidence items from persisted query results."""
        results = self.rag.query_results(query, top_k=top_k)
        out: list[EvidenceItem] = []
        for result in results:
            metadata = dict(result.metadata or {})
            page_path = metadata.get("page_path")
            chunk_id = metadata.get("chunk_id")
            page_id = metadata.get("page_id")
            content_sha = metadata.get("content_sha256")
            if not (page_path and chunk_id and page_id and content_sha):
                continue
            score = max(0.0, min(1.0, float(result.score)))
            out.append(
                EvidenceItem(
                    source_id=f"wiki:{page_id}",
                    page_path=page_path,
                    chunk_id=chunk_id,
                    score=score,
                    excerpt=result.text[:4000],
                    page_id=page_id,
                    title=metadata.get("title"),
                    source_uri=page_path,
                    metadata=metadata,
                )
            )
        return out[:top_k]

    def export_graph(self, output_path: str | Path):
        from .graph_export import export_graph

        return export_graph(self, output_path)


__all__ = ["KnowledgeService", "KnowledgeSummary"]
