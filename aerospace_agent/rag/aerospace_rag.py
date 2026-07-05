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

    # ------------------------------------------------------------------ 状态
    def status(self) -> dict:
        return self.kb.status()

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
