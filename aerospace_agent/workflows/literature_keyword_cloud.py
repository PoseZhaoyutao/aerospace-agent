"""Local literature keyword cloud workflow."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "into", "using", "used", "use", "can", "may", "model", "models", "method",
    "methods", "result", "results", "study", "paper", "based", "analysis",
}


def _paper_text(paper: Any) -> str:
    if isinstance(paper, str):
        return paper
    if isinstance(paper, dict):
        parts = [
            paper.get("title", ""),
            paper.get("abstract", ""),
            paper.get("summary", ""),
            paper.get("text", ""),
        ]
        return "\n".join(str(part) for part in parts if part)
    return str(paper)


def _extract_keywords(texts: Iterable[str], max_keywords: int = 30) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower()):
            if token in STOPWORDS:
                continue
            counter[token] += 1
    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(max_keywords)
    ]


def _write_keyword_svg(path: Path, keywords: List[Dict[str, Any]], title: str) -> None:
    width, height = 960, 540
    max_count = max([item["count"] for item in keywords] or [1])
    rows = []
    x, y = 48, 92
    for index, item in enumerate(keywords):
        size = 18 + int(42 * item["count"] / max_count)
        color = ["#1b998b", "#2d3047", "#ff9b71", "#e84855", "#577590"][index % 5]
        rows.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-family="Segoe UI, Arial">{item["term"]}</text>'
        )
        x += 120 + size * len(item["term"]) // 3
        if x > width - 220:
            x = 48
            y += 72
    body = "\n  ".join(rows)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8f9fb"/>
  <text x="48" y="44" font-size="24" fill="#111827" font-family="Segoe UI, Arial">{title}</text>
  {body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _search_with_rag(query: str, rag: Any, max_results: int) -> List[Any]:
    if rag is None:
        return []
    for method_name in ("search_literature", "query_results", "retrieve"):
        method = getattr(rag, method_name, None)
        if method is None:
            continue
        try:
            try:
                result = method(query, top_k=max_results)
            except TypeError:
                result = method(query)
        except Exception:
            continue
        if isinstance(result, dict):
            return result.get("results") or result.get("papers") or []
        if isinstance(result, list):
            return result
        if result:
            return [str(result)]
    return []


def run_literature_keyword_cloud_workflow(
    query: str,
    papers: Optional[List[Any]] = None,
    rag: Any = None,
    output_dir: str | Path = "artifacts/literature_keyword_cloud",
    max_results: int = 10,
    max_keywords: int = 30,
) -> Dict[str, Any]:
    """Search literature context and render a deterministic keyword cloud."""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    papers = list(papers or [])
    source = "provided"
    if not papers:
        papers = _search_with_rag(query, rag, max_results=max_results)
        source = "rag" if papers else "none"

    texts = [_paper_text(paper) for paper in papers]
    if not texts:
        texts = [query]

    keywords = _extract_keywords(texts, max_keywords=max_keywords)

    papers_path = output_path / "papers.json"
    keywords_path = output_path / "keywords.json"
    cloud_path = output_path / "keyword_cloud.svg"

    papers_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    keywords_path.write_text(json.dumps(keywords, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_keyword_svg(cloud_path, keywords, title=f"Keyword cloud: {query}")

    return {
        "status": "ok" if papers else "partial",
        "query": query,
        "source": source,
        "paper_count": len(papers),
        "keywords": keywords,
        "artifacts": {
            "papers": str(papers_path),
            "keywords": str(keywords_path),
            "keyword_cloud": str(cloud_path),
        },
        "notes": [
            "No benchmark or literature conclusion is implied by keyword frequency.",
            "When no external search source is available, the workflow still emits reproducible local artifacts.",
        ],
    }
