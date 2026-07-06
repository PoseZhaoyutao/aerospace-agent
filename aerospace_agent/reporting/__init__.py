"""aerospace_agent.reporting — 报告生成与绘图模块。

子模块:
    formulas          LaTeX 公式集合 (FORMULAS / DERIVATIONS)
    plots             Plotter 统一绘图接口
    report            ReportGenerator HTML 报告生成器 (任务分析报告)
    knowledge_report  KnowledgeReportGenerator 知识学习报告生成器
"""

from __future__ import annotations

from .plots import Plotter
from .report import ReportGenerator
from .formulas import FORMULAS, DERIVATIONS, get_formula_latex, get_derivation
from .knowledge_report import KnowledgeReportGenerator

__all__ = [
    "Plotter",
    "ReportGenerator",
    "KnowledgeReportGenerator",
    "FORMULAS",
    "DERIVATIONS",
    "get_formula_latex",
    "get_derivation",
]
