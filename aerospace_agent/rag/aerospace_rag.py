"""AerospaceRAG —— 航天知识检索增强生成 (RAG) 顶层门面。

对外暴露统一接口, 内部组合「向量库 + 关键词倒排 + 知识图谱」三路混合检索,
并集成 RetrieverRouter 多源路由 + EvidenceVerifier 证据验证 + TraceabilityManager 溯源链:

    rag = AerospaceRAG()          # 自动 load 已有索引, 否则新建并预填充
    rag.index("/path/to/docs")    # 索引目录 (或单段文本)
    ctx = rag.query("地月转移用什么方法", top_k=5)  # -> 格式化上下文文本
    # 增强检索（多源路由 + 证据验证 + 溯源链）
    result = rag.query_enhanced("TLI 速度增量约 3.1 km/s", "地月转移 TLI 计算")
    print(result["trace"]["cited_answer"])  # 答案 + 来源引用
    print(rag.status())
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from .knowledge_base import AerospaceKnowledgeBase, DEFAULT_DATA_DIR
from .retriever import RetrievalResult, HybridRetriever

__all__ = ["AerospaceRAG"]


class AerospaceRAG:
    """RAG 顶层门面——可路由、可验证、可追踪的知识工具系统。

    三层架构：
      1. 基础层：HybridRetriever（向量 + BM25 + 知识图谱 三路混合）
      2. 路由层：RetrieverRouter（文档/数据库/代码/记忆/网络 多源路由）
      3. 验证层：EvidenceVerifier + TraceabilityManager（证据验证 + 溯源链）
    """

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
        # 增强组件（懒加载）
        self._router = None
        self._verifier = None
        self._tracer = None

    @property
    def retriever(self) -> HybridRetriever:
        """底层 HybridRetriever 实例。"""
        return self.kb

    @property
    def router(self):
        """RetrieverRouter 多源路由器（懒加载）。"""
        if self._router is None:
            from .router import RetrieverRouter
            self._router = RetrieverRouter(
                doc_retriever=self.kb,  # 文档源用 HybridRetriever
            )
        return self._router

    @property
    def verifier(self):
        """EvidenceVerifier 证据验证器（懒加载）。"""
        if self._verifier is None:
            from .verifier import EvidenceVerifier
            self._verifier = EvidenceVerifier()
        return self._verifier

    @property
    def tracer(self):
        """TraceabilityManager 溯源链管理器（懒加载）。"""
        if self._tracer is None:
            from .trace import TraceabilityManager
            self._tracer = TraceabilityManager()
        return self._tracer

    def register_code_retriever(self, fn):
        """注册代码检索源。"""
        self.router.code_retriever = fn

    def register_memory_retriever(self, fn):
        """注册记忆检索源。"""
        self.router.memory_retriever = fn

    def register_web_retriever(self, fn):
        """注册网络搜索源。"""
        self.router.web_retriever = fn

    def register_db_retriever(self, fn):
        """注册数据库检索源。"""
        self.router.db_retriever = fn

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

    # ------------------------------------------------------------------ 多源路由检索
    def query_routed(self, query: str, top_k: int = 5,
                     force_sources: List[str] = None) -> dict:
        """多源路由检索——根据查询意图自动路由到不同检索源。

        Args:
            query: 查询文本
            top_k: 每个源返回的最大结果数
            force_sources: 强制使用的源列表（覆盖自动检测）
                可选: document / database / code / memory / web

        Returns:
            {
                "query": 原始查询,
                "results": [RetrievalResult...],
                "sources_used": [源名...],
                "routing_reason": 路由原因,
            }
        """
        routed = self.router.retrieve(query, top_k=top_k,
                                      force_sources=force_sources)
        return routed.to_dict()

    # ------------------------------------------------------------------ 增强检索（路由 + 验证 + 溯源）
    def query_enhanced(self, answer: str, query: str,
                       top_k: int = 5) -> dict:
        """增强检索——对答案做证据验证并构建溯源链。

        这是 RAG 的完整闭环：
            1. 用 query 检索证据（多源路由）
            2. 用 EvidenceVerifier 验证 answer 的每个声明是否有证据支撑
            3. 用 TraceabilityManager 构建溯源链（答案 + source_id 引用）

        Args:
            answer: 待验证的答案文本（LLM 生成或人工编写）
            query: 原始查询（用于检索证据）
            top_k: 检索结果数

        Returns:
            {
                "query": 原始查询,
                "evidence": 检索证据列表,
                "verification": 验证报告 (overall_support / confidence / links),
                "trace": 溯源链 (answer + cited_answer + trace_entries),
            }
        """
        # 1. 多源路由检索证据
        evidence = self.query_routed(query, top_k=top_k)
        evidence_results = [
            RetrievalResult(
                text=r.get("text", ""),
                score=r.get("score", 0.0),
                source=r.get("source", "unknown"),
                metadata=r.get("metadata", {}),
                explanation=r.get("explanation", ""),
            )
            for r in evidence.get("results", [])
        ]

        # 2. 证据验证
        report = self.verifier.verify(answer, evidence_results)

        # 3. 构建溯源链
        trace = self.tracer.build_trace(
            answer=answer,
            query=query,
            evidence=evidence_results,
            verification=report,
            retrieval_sources=evidence.get("sources_used", []),
        )

        return {
            "query": query,
            "evidence": evidence,
            "verification": report.to_dict(),
            "trace": {
                "answer": trace.answer,
                "cited_answer": trace.to_citation_string(),
                "trace_entries": [t.to_dict() for t in trace.trace_entries],
                "retrieval_sources": trace.retrieval_sources,
                "timestamp": trace.timestamp,
            },
        }

    def query_with_verification(self, query: str, answer: str,
                                top_k: int = 5) -> str:
        """检索 + 验证 + 溯源，返回带引用的格式化文本（供 Agent 注入 LLM 上下文）。

        与 query() 的区别：
            - query() 只返回检索结果
            - query_with_verification() 返回检索结果 + 答案验证 + 来源引用

        Args:
            query: 检索查询
            answer: 待验证的答案
            top_k: 检索结果数

        Returns:
            格式化文本，包含检索结果、验证报告、溯源引用
        """
        enhanced = self.query_enhanced(answer, query, top_k=top_k)
        lines: List[str] = []
        lines.append(f"[增强 RAG 检索] 查询: {query!r}")
        lines.append(f"答案: {answer[:100]}...")
        lines.append("=" * 78)

        # 检索结果
        ev = enhanced["evidence"]
        lines.append(f"检索源: {', '.join(ev.get('sources_used', ['document']))}")
        lines.append(f"证据数: {len(ev.get('results', []))}")
        for i, r in enumerate(ev.get("results", []), 1):
            lines.append(f"  {i}. [{r.get('source','?')}] score={r.get('score',0):.3f}")
            lines.append(f"     {r.get('text','')[:80]}...")

        # 验证报告
        v = enhanced["verification"]
        lines.append("-" * 78)
        lines.append(f"证据验证: {v['overall_support']} (confidence={v['confidence']:.2f})")
        lines.append(f"  {v['summary']}")
        if v.get("unsupported_claims"):
            lines.append(f"  无支撑声明: {len(v['unsupported_claims'])} 条")

        # 溯源引用
        trace = enhanced["trace"]
        lines.append("-" * 78)
        lines.append("来源引用:")
        for i, t in enumerate(trace.get("trace_entries", []), 1):
            mark = "✓" if t.get("used_in_answer") else " "
            lines.append(f"  {mark} [{i}] ({t.get('source_type','?')}) "
                         f"{t.get('source_ref','?')}")

        return "\n".join(lines)

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
            research_topic=topic,
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
        self, output_path: str = None
    ) -> str:
        """生成动态知识云图 HTML（力导向交互式可视化）。"""
        from .knowledge_cloud import KnowledgeCloudGenerator

        if output_path is None:
            output_path = os.path.join(os.getcwd(), "reports",
                                       "knowledge_cloud.html")
        gen = KnowledgeCloudGenerator()
        return gen.generate(self.kb.knowledge_graph, output_path=output_path)

    # ------------------------------------------------------------------ 知识报告
    def generate_knowledge_report(
        self,
        pipeline_report=None,
        output_path: str = None,
    ) -> str:
        """生成知识学习报告（含概念网络分析、文献记录、论文写作辅助）。"""
        from ..reporting.knowledge_report import KnowledgeReportGenerator

        if output_path is None:
            output_path = os.path.join(os.getcwd(), "reports",
                                       "knowledge_learning_report.html")
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
        papers_dir = os.path.join(os.getcwd(), "data", "papers")
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
