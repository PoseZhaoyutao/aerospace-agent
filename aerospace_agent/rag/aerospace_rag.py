"""AerospaceRAG —— 航天知识检索增强生成 (RAG) 顶层门面。

对外暴露统一接口, 内部组合「向量库 + 关键词倒排 + 知识图谱」三路混合检索:

    rag = AerospaceRAG()          # 自动 load 已有索引, 否则新建并预填充
    rag.index("/path/to/docs")    # 索引目录 (或单段文本)
    ctx = rag.query("地月转移用什么方法", top_k=5)  # -> 格式化上下文文本
    print(rag.status())

设计为对 ``aerospace_agent`` 的 Agent 友好: ``query`` 返回一段可直接注入
LLM 上下文的格式化文本, 包含命中文档、来源、分数, 以及知识图谱推导链。
"""

from __future__ import annotations

import os
from typing import List, Union

from .knowledge_base import AerospaceKnowledgeBase, DEFAULT_DATA_DIR
from .retriever import RetrievalResult

__all__ = ["AerospaceRAG"]


class AerospaceRAG:
    """RAG 顶层门面。"""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        autoload: bool = True,
        auto_default_knowledge: bool = True,
    ):
        self.data_dir = data_dir
        self.kb = AerospaceKnowledgeBase(
            data_dir=data_dir,
            autoload=autoload,
            auto_default_knowledge=auto_default_knowledge,
        )

    # ------------------------------------------------------------------ 索引
    def index(self, doc_or_dir: Union[str, List[str]], **kwargs) -> int:
        """统一索引入口。

        * 若为目录路径 -> ``index_directory``
        * 若为单段文本 -> ``index_text``
        * 若为字符串列表 -> 逐条 ``index_text``
        """
        if isinstance(doc_or_dir, list):
            n = 0
            for t in doc_or_dir:
                if t and t.strip():
                    self.kb.index_text(t, source=kwargs.get("source", "manual"))
                    n += 1
            self.kb.vector_store.reindex()
            return n
        if isinstance(doc_or_dir, str) and os.path.isdir(doc_or_dir):
            return self.kb.index_directory(
                doc_or_dir, extensions=kwargs.get("extensions", [".md", ".txt", ".py"])
            )
        # 单段文本
        self.kb.index_text(doc_or_dir, source=kwargs.get("source", "manual"))
        self.kb.vector_store.reindex()
        return 1

    # ------------------------------------------------------------------ 检索
    def query(self, query: str, top_k: int = 5) -> str:
        """检索并返回格式化上下文文本 (供 Agent 注入 LLM 上下文)。"""
        results = self.kb.query(query, top_k=top_k, use_reranker=True)
        return self._format(query, results)

    def query_results(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """返回结构化结果 (程序化使用)。"""
        return self.kb.query(query, top_k=top_k, use_reranker=True)

    # ------------------------------------------------------------------ 格式化
    def _format(self, query: str, results: List[RetrievalResult]) -> str:
        lines: List[str] = []
        lines.append(f"[航天知识检索] 查询: {query!r}  (top {len(results)})")
        lines.append("=" * 78)
        if not results:
            lines.append("(无检索结果)")
            return "\n".join(lines)
        for i, r in enumerate(results, 1):
            node = r.metadata.get("node_id") or r.metadata.get("source", "?")
            ntype = r.metadata.get("type") or r.metadata.get("node_type", "")
            head = f"{i}. [{r.source}] score={r.score:.4f} <{node}>"
            if ntype:
                head += f" ({ntype})"
            lines.append(head)
            # 正文: 折行显示
            text = r.text.strip()
            while len(text) > 76:
                cut = text.rfind(" ", 0, 76)
                if cut < 40:
                    cut = 76
                lines.append("   " + text[:cut].strip())
                text = text[cut:].strip()
            if text:
                lines.append("   " + text)
            if r.explanation:
                lines.append("   ↳ " + r.explanation)
        # 附: 若命中图谱概念, 追加一条知识链
        concepts = self.kb.knowledge_graph.match_concepts(query)
        if concepts:
            top_concept = concepts[0][0]
            chain = self.kb.knowledge_graph.explain(top_concept)
            lines.append("-" * 78)
            lines.append(f"[知识图谱推导链: {top_concept}]")
            lines.append(chain)
        return "\n".join(lines)

    # ------------------------------------------------------------------ 文献搜索
    def search_literature(
        self,
        query: str,
        research_topic: str = "",
        max_results: int = 10,
        download_strong: bool = True,
    ) -> dict:
        """搜索最新文献、评估相关性、下载强相关论文并总结全文。

        流程：CSTCloud 登录 → arXiv 搜索 → 摘要相关性评分 →
        strong 相关：下载 PDF + 全文总结 + 索引入 RAG + 更新知识图谱 →
        weak 相关：跳过 → 生成知识云图。

        Args:
            query: arXiv 搜索关键词（如 "lunar transfer orbit"）
            research_topic: 研究主题（用于相关性评分，默认等同 query）
            max_results: 最大搜索结果数
            download_strong: 是否下载强相关论文的 PDF

        Returns:
            管线报告字典，含 total_found / strong_count / weak_count /
            downloaded / papers / knowledge_cloud_path 等字段。
        """
        from .literature_pipeline import LiteraturePipeline

        topic = research_topic or query
        pipeline = LiteraturePipeline(rag=self)
        report = pipeline.run(
            research_topic=query,
            max_papers=max_results,
            min_relevance="strong" if download_strong else "weak",
        )
        return {
            "research_topic": report.research_topic,
            "total_found": report.total_found,
            "strong_count": report.strong_count,
            "weak_count": report.weak_count,
            "downloaded_count": report.downloaded_count,
            "papers": [
                {
                    "title": pr.paper.title,
                    "arxiv_id": pr.paper.id,
                    "authors": pr.paper.authors[:3],
                    "relevance": pr.score.relevance if pr.score else "unknown",
                    "score": pr.score.score if pr.score else 0.0,
                    "status": pr.status,
                    "summary": (pr.summary[:200] + " ...") if pr.summary and len(pr.summary) > 200 else pr.summary,
                    "pdf_path": pr.pdf_path,
                    "concepts": pr.concepts,
                }
                for pr in report.papers
            ],
            "knowledge_graph_snapshot": report.knowledge_graph_snapshot,
        }

    # ------------------------------------------------------------------ 知识云图
    def generate_knowledge_cloud(
        self, output_path: str = "/workspace/reports/knowledge_cloud.html"
    ) -> str:
        """生成动态知识云图 HTML（力导向交互式可视化）。"""
        from .knowledge_cloud import KnowledgeCloudGenerator

        gen = KnowledgeCloudGenerator()
        return gen.generate(self.kb.knowledge_graph, output_path=output_path)

    # ------------------------------------------------------------------ 知识报告
    def generate_knowledge_report(
        self,
        pipeline_report=None,
        output_path: str = "/workspace/reports/knowledge_learning_report.html",
    ) -> str:
        """生成知识学习报告（含概念网络分析、文献记录、论文写作辅助）。"""
        from ..reporting.knowledge_report import KnowledgeReportGenerator

        gen = KnowledgeReportGenerator()
        return gen.generate(
            self.kb.knowledge_graph,
            pipeline_report=pipeline_report,
            output_path=output_path,
        )

    # ------------------------------------------------------------------ 状态
    def status(self) -> dict:
        s = self.kb.status()
        # 补充文献相关状态
        papers_dir = "/workspace/data/papers"
        if os.path.isdir(papers_dir):
            s["downloaded_papers"] = len(
                [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
            )
        else:
            s["downloaded_papers"] = 0
        return s

    # ------------------------------------------------------------------ 持久化
    def save(self) -> None:
        self.kb.save()


# ---------------------------------------------------------------------------
# 自测 / 验收
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        rag = AerospaceRAG(data_dir=d, autoload=False)
        print("[status]", rag.status())

        q = "地月转移用什么方法"
        print("\n" + rag.query(q, top_k=5))

        # 验收: 结果文本应包含 Hohmann / 拼凑圆锥 / vis-viva
        text = rag.query(q, top_k=6)
        low = text.lower()
        for kw in ["hohmann", "拼凑圆锥", "vis-viva"]:
            assert kw in low, f"验收失败: 缺 {kw}"
        print("\n[ok] 验收通过: AerospaceRAG.query 返回包含 Hohmann / 拼凑圆锥 / vis-viva")
