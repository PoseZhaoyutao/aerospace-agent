"""知识检索技能 —— 多源路由 RAG 检索。

调用 agent.rag (AerospaceRAG) 的 query_routed 方法，根据查询意图
自动路由到文档/代码/记忆/网络等检索源，返回合并去重后的结果。
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import SkillBase


class KnowledgeRetrievalSkill(SkillBase):
    """知识检索技能。

    通过 RetrieverRouter 多源路由检索，根据查询意图自动选择检索源
    （文档/数据库/代码/记忆/网络），合并去重后返回 top_k 结果。
    """

    name: str = "knowledge_retrieval"
    description: str = "知识检索：多源路由 RAG 检索（文档/代码/记忆/网络）"
    category: str = "rag"

    def is_available(self) -> bool:
        """RAG 模块较重，惰性检测是否可导入。"""
        try:
            import aerospace_agent.rag  # noqa: F401
            return True
        except Exception:
            return False

    def execute(self, agent, **kwargs) -> dict:
        """执行多源路由知识检索。

        Args:
            agent: AerospaceAgent 实例
            query: 检索查询文本（必填）
            top_k: 每个源返回的最大结果数（默认 5）
            force_sources: 强制使用的源列表，覆盖自动检测
                可选值: document / database / code / memory / web

        Returns:
            {"success", "result": {"query", "results", "sources_used", ...}, "message"}
        """
        rag = getattr(agent, "rag", None)
        if rag is None:
            return self._error("Agent 未挂载 rag（知识检索引擎）")

        query: str = kwargs.get("query", "")
        top_k: int = kwargs.get("top_k", 5)
        force_sources: List[str] | None = kwargs.get("force_sources")

        if not query or not query.strip():
            return self._error("缺少必填参数 query（检索查询文本）")

        # 优先使用多源路由检索；不支持时回退到普通检索
        if hasattr(rag, "query_routed"):
            try:
                result: Dict[str, Any] = rag.query_routed(
                    query=query, top_k=top_k, force_sources=force_sources)
            except Exception as exc:
                return self._error(f"多源路由检索失败: {exc}")
        elif hasattr(rag, "query_results"):
            # 回退：返回结构化结果列表
            try:
                raw = rag.query_results(query=query, top_k=top_k)
                result = {
                    "query": query,
                    "results": [
                        {"text": getattr(r, "text", str(r)),
                         "score": getattr(r, "score", 0.0),
                         "source": getattr(r, "source", "document")}
                        for r in raw
                    ],
                    "sources_used": ["document"],
                    "routing_reason": "回退到单源文档检索",
                }
            except Exception as exc:
                return self._error(f"回退检索失败: {exc}")
        else:
            return self._error("RAG 引擎不支持 query_routed 或 query_results")

        count = len(result.get("results", []))
        sources = result.get("sources_used", [])
        return {
            "success": True,
            "result": result,
            "message": f"检索完成：{count} 条结果，使用源 {sources}",
        }

    @staticmethod
    def _error(message: str) -> dict:
        """返回标准化错误结果。"""
        return {"success": False, "result": None, "message": message}
