"""上下文管理技能 —— CEO 三层上下文的摘要、压缩与卸载。

调用 agent.context_manager 完成以下操作:
    1. 构建上下文摘要 (build_context)
    2. 按需压缩历史 (clear_compressed)
    3. 卸载大块数据到外部文件 (save_offload)
"""
from __future__ import annotations

from typing import Any, Dict

from .base import SkillBase


class ContextManagementSkill(SkillBase):
    """上下文管理技能。

    负责对 Agent 的 CEO 三层上下文（Essential / Compress / Offload）
    进行摘要生成、历史压缩与大块数据卸载，确保 token 预算可控。
    """

    name: str = "context_management"
    description: str = "上下文管理：生成摘要、压缩历史、卸载大块数据"
    category: str = "context"

    def is_available(self) -> bool:
        """上下文管理器是核心组件，默认始终可用。"""
        return True

    def execute(self, agent, **kwargs) -> dict:
        """执行上下文管理操作。

        Args:
            agent: AerospaceAgent 实例
            token_budget: 上下文 token 预算（默认 8000）
            compress: 是否清空已压缩的 Compress 层（默认 False）
            offload_key: 卸载数据的键名（提供 offload_data 时生效）
            offload_data: 待卸载的大块数据

        Returns:
            {"success", "result": {"context", "stats", "offload_path"}, "message"}
        """
        cm = getattr(agent, "context_manager", None)
        if cm is None:
            return self._error("Agent 未挂载 context_manager")

        token_budget: int = kwargs.get("token_budget", 8000)
        compress: bool = kwargs.get("compress", False)
        offload_key: str | None = kwargs.get("offload_key")
        offload_data: Any = kwargs.get("offload_data")

        # 1. 构建上下文摘要
        context_text = ""
        try:
            context_text = cm.build_context(token_budget=token_budget)
        except Exception as exc:
            return self._error(f"构建上下文失败: {exc}")

        # 2. 获取各层统计
        stats: Dict[str, Any] = {}
        if hasattr(cm, "stats"):
            try:
                stats = cm.stats()
            except Exception:
                stats = {}

        # 3. 按需压缩（清空 Compress 层，保留 Essential 与 Offload 索引）
        if compress and hasattr(cm, "clear_compressed"):
            try:
                cm.clear_compressed()
            except Exception as exc:
                return self._error(f"压缩上下文失败: {exc}")

        # 4. 卸载大块数据到外部文件
        offload_path: str | None = None
        if offload_key and offload_data is not None and hasattr(cm, "save_offload"):
            try:
                offload_path = cm.save_offload(offload_key, offload_data)
            except Exception as exc:
                return self._error(f"卸载数据失败: {exc}")

        return {
            "success": True,
            "result": {
                "context": context_text,
                "stats": stats,
                "offload_path": offload_path,
            },
            "message": "上下文管理执行完成",
        }

    @staticmethod
    def _error(message: str) -> dict:
        """返回标准化错误结果。"""
        return {"success": False, "result": None, "message": message}
