"""报告生成技能 —— 将任务结果转为自包含 HTML 报告。

调用 aerospace_agent.reporting.ReportGenerator，把工作流结果
(WorkflowResult) 转换为带内嵌图表、MathJax 公式的深色航天工程报告。
"""
from __future__ import annotations

import os
from typing import Any, Dict

from .base import SkillBase


class ReportGenerationSkill(SkillBase):
    """报告生成技能。

    将 Agent 的工作流执行结果转换为自包含的 HTML 报告，
    包含执行摘要、任务设计、物理公式、发射窗口、轨迹设计、
    Delta-v 预算、任务时间线、工具验证与结论建议等章节。
    """

    name: str = "report_generation"
    description: str = "报告生成：将任务结果转为自包含 HTML 报告（含图表与公式）"
    category: str = "reporting"

    def is_available(self) -> bool:
        """报告生成依赖 reporting 模块与 matplotlib，惰性检测。"""
        try:
            import matplotlib  # noqa: F401
            from aerospace_agent.reporting import ReportGenerator  # noqa: F401
            return True
        except Exception:
            return False

    def execute(self, agent, **kwargs) -> dict:
        """执行报告生成。

        Args:
            agent: AerospaceAgent 实例（本技能未直接使用 agent，但保持接口一致）
            results: 工作流结果（WorkflowResult 或含 result 字典的对象，必填）
            output_path: HTML 报告输出路径（默认 <cwd>/reports/skill_report.html）

        Returns:
            {"success", "result": {"report_path", "report_size_kb"}, "message"}
        """
        # 惰性导入报告生成器（重依赖：matplotlib 等）
        try:
            from aerospace_agent.reporting import ReportGenerator
        except Exception as exc:
            return self._error(f"reporting 模块不可用: {exc}")

        results: Any = kwargs.get("results")
        if results is None:
            return self._error("缺少必填参数 results（工作流结果）")

        output_path: str = kwargs.get(
            "output_path", os.path.join(os.getcwd(), "reports", "skill_report.html"))

        # 确保输出目录存在
        try:
            out_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            return self._error(f"创建输出目录失败: {exc}")

        # 生成报告
        try:
            gen = ReportGenerator()
            report_path = gen.generate_lunar_transfer_report(
                results, output_path=output_path)
        except Exception as exc:
            return self._error(f"报告生成失败: {exc}")

        # 获取文件大小
        size_kb = 0.0
        try:
            size_kb = os.path.getsize(report_path) / 1024.0
        except Exception:
            pass

        return {
            "success": True,
            "result": {
                "report_path": report_path,
                "report_size_kb": round(size_kb, 1),
            },
            "message": f"报告已生成: {report_path} ({size_kb:.1f} KB)",
        }

    @staticmethod
    def _error(message: str) -> dict:
        """返回标准化错误结果。"""
        return {"success": False, "result": None, "message": message}
