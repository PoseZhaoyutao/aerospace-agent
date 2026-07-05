"""aerospace_agent.reporting — 报告生成与绘图模块。

子模块:
    formulas  LaTeX 公式集合 (FORMULAS / DERIVATIONS)
    plots     Plotter 统一绘图接口
    report    ReportGenerator HTML 报告生成器
"""

from __future__ import annotations

from .plots import Plotter
from .report import ReportGenerator
from .formulas import FORMULAS, DERIVATIONS, get_formula_latex, get_derivation

__all__ = [
    "Plotter",
    "ReportGenerator",
    "FORMULAS",
    "DERIVATIONS",
    "get_formula_latex",
    "get_derivation",
]
