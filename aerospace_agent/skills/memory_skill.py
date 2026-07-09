"""记忆召回技能 —— 从长期记忆中检索与查询相关的历史记录。

调用 agent.memory (LongTermMemory) 的 recall 方法，基于简化向量
相似度（字符袋 + 随机投影）检索最相关的 top_k 条记忆。
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import SkillBase


class MemoryRecallSkill(SkillBase):
    """记忆召回技能。

    从 Agent 的长期记忆中检索与查询语义相关的历史记录，
    返回键名、相似度分数与记忆内容，供 ReAct 推理注入上下文。
    """

    name: str = "memory_recall"
    description: str = "记忆召回：从长期记忆中检索与查询相关的历史记录"
    category: str = "memory"

    def is_available(self) -> bool:
        """长期记忆依赖 numpy，尝试惰性导入检测。"""
        try:
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def execute(self, agent, **kwargs) -> dict:
        """执行记忆召回。

        Args:
            agent: AerospaceAgent 实例
            query: 检索查询文本（必填）
            top_k: 返回的记忆条数上限（默认 5）

        Returns:
            {"success", "result": {"memories", "count"}, "message"}
        """
        memory = getattr(agent, "memory", None)
        if memory is None:
            return self._error("Agent 未挂载 memory（长期记忆）")

        query: str = kwargs.get("query", "")
        top_k: int = kwargs.get("top_k", 5)

        if not query or not query.strip():
            return self._error("缺少必填参数 query（检索查询文本）")

        # recall 返回 List[Tuple[key, similarity, value]]
        raw_results: List[tuple] = []
        try:
            raw_results = memory.recall(query, top_k=top_k)
        except Exception as exc:
            return self._error(f"记忆召回失败: {exc}")

        # 格式化为结构化字典列表
        memories: List[Dict[str, Any]] = []
        for key, similarity, value in raw_results:
            memories.append({
                "key": key,
                "similarity": round(float(similarity), 4),
                "value": value,
            })

        return {
            "success": True,
            "result": {
                "query": query,
                "memories": memories,
                "count": len(memories),
            },
            "message": f"召回 {len(memories)} 条相关记忆",
        }

    @staticmethod
    def _error(message: str) -> dict:
        """返回标准化错误结果。"""
        return {"success": False, "result": None, "message": message}
