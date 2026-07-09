"""FixAgent -- 自主修复 - 最小变更 - 验证闭环.

职责边界：
    接收 ArchAgent / TestAgent 发现的问题清单，生成修复计划并自主执行。
    所有修复遵循"最小变更"原则：用定向搜索替换修改文件，不重写整个文件。

修复能力清单：
    1. analyze_issues          -- 分析问题清单，按严重度排序生成修复计划
    2. fix_circular_import     -- 循环依赖修复（sys.path hack -> 包内相对导入）
    3. fix_context_desync      -- 上下文压缩与 LLM 输入脱节修复（messages 截断）
    4. fix_yaml_interpolation  -- YAML 工作流变量插值实现（${inputs.x} 解析）
    5. fix_j2_placeholder      -- J2 摄动占位修复（RK4 积分平均长期效应）
    6. fix_skill_integration   -- 技能系统接入 ReAct 循环（system prompt 注入）
    7. apply_fixes             -- 批量执行修复计划

设计原则（LoopRecursive-CEO Phase A）：
    K2 上下文压缩必须控制 LLM 输入 -- fix_context_desync
    K3 工作流必须先解析变量再执行   -- fix_yaml_interpolation
    K6 最小变更：定向搜索替换，不重写文件
    K7 验证闭环：每个修复先读取确认当前内容，修复后返回变更详情
"""
from __future__ import annotations

import ast
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import AgentBase, AgentResult, AgentRole


class FixAgent(AgentBase):
    """自主修复 Agent -- 最小变更 - 验证闭环.

    继承 AgentBase，role = AgentRole.FIX。
    通过定向搜索替换（_apply_search_replace）修改现有文件，不重写整个文件。
    每个修复方法先读取目标文件确认当前内容，再执行最小变更。

    Attributes:
        role: AgentRole.FIX
        project_root: 项目根目录
        log: 结构化日志器 (agent.fix)
        metrics: Metrics 收集器
    """

    role = AgentRole.FIX

    #: 严重度排序权重（数值越小优先级越高）
    _SEVERITY_ORDER: Dict[str, int] = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
        "info": 4,
    }

    #: 问题关键词 -> 修复策略映射
    _STRATEGY_KEYWORDS: Dict[str, str] = {
        "circular_import": "direct_edit",
        "sys.path": "direct_edit",
        "循环依赖": "direct_edit",
        "context": "refactor",
        "上下文": "refactor",
        "脱节": "refactor",
        "yaml": "config_change",
        "插值": "config_change",
        "variable": "config_change",
        "j2": "refactor",
        "摄动": "refactor",
        "placeholder": "refactor",
        "占位": "refactor",
        "skill": "new_file",
        "技能": "new_file",
    }

    def __init__(self, project_root: str = "."):
        super().__init__(project_root)

    # ==================================================================
    # 1. 问题分析
    # ==================================================================
    def analyze_issues(self, issues: List[Dict]) -> AgentResult:
        """分析问题清单，生成修复计划。

        接收 ArchAgent/TestAgent 发现的问题列表，按严重度排序，
        为每个问题分配修复策略和复杂度估算。

        Args:
            issues: 问题列表，每项格式::
                {"id": "...", "file": "...", "issue": "...", "severity": "..."}

        Returns:
            AgentResult, data 格式::
                {
                    "total_issues": N,
                    "plan": [
                        {"id": ..., "strategy": ..., "priority": ..., "estimated_complexity": "low/medium/high"}
                    ]
                }
        """
        with self._time_operation("analyze_issues"):
            if not issues:
                self.log.info("analyze_issues_empty")
                return self._make_result(
                    "analyze_issues", True,
                    data={"total_issues": 0, "plan": []},
                )

            # 按严重度排序（critical > high > medium > low > info）
            sorted_issues = sorted(
                issues,
                key=lambda x: self._SEVERITY_ORDER.get(
                    str(x.get("severity", "low")).lower(), 3),
            )

            plan: List[Dict[str, Any]] = []
            for idx, issue in enumerate(sorted_issues):
                strategy = self._determine_strategy(issue)
                complexity = self._estimate_complexity(issue, strategy)
                plan.append({
                    "id": issue.get("id", f"issue_{idx}"),
                    "strategy": strategy,
                    "priority": idx + 1,
                    "estimated_complexity": complexity,
                    "file": issue.get("file", ""),
                    "issue": issue.get("issue", ""),
                    "severity": issue.get("severity", "low"),
                })

            self.log.info("issues_analyzed", data={
                "total": len(issues), "plan_size": len(plan),
                "strategies": {p["strategy"] for p in plan},
            })
            self.metrics.gauge("fix.issues_total", len(issues))
            self.metrics.gauge("fix.plan_size", len(plan))

            return self._make_result(
                "analyze_issues", True,
                data={"total_issues": len(issues), "plan": plan},
            )

    # ==================================================================
    # 2. 循环依赖修复
    # ==================================================================
    def fix_circular_import(self, file_path: str) -> AgentResult:
        """修复循环依赖 -- 将 sys.path hack 替换为正确的相对导入。

        读取文件，找到 ``sys.path.insert(0, ...)`` + ``from core.xxx import`` 模式，
        替换为 ``from ..core.xxx import``（正确的包内相对导入），删除 sys.path hack 行。

        Args:
            file_path: 目标文件路径（相对于 project_root 或绝对路径）

        Returns:
            AgentResult, data 格式::
                {"file": file_path, "old_pattern": "...", "new_pattern": "...", "lines_changed": N}
            如果文件不存在该模式：success=True, data={"already_fixed": True}
        """
        with self._time_operation("fix_circular_import"):
            full_path = self._resolve_path(file_path)

            if not os.path.exists(full_path):
                return self._make_result(
                    "fix_circular_import", False,
                    error=f"File not found: {full_path}",
                )

            content = self._read_file(file_path)
            if content is None:
                return self._make_result(
                    "fix_circular_import", False,
                    error=f"Cannot read file: {full_path}",
                )

            # 匹配 sys.path.insert(0, ...) 行
            sys_path_re = re.compile(
                r"^[ \t]*sys\.path\.insert\s*\([^)]+\)[^\n]*\n",
                re.MULTILINE,
            )
            # 匹配 from core.xxx import ... （不含 .. 前缀的裸 core 引用）
            import_re = re.compile(
                r"^([ \t]*)from\s+core\.([\w.]+)\s+import\s+(.+)$",
                re.MULTILINE,
            )

            sys_path_matches = sys_path_re.findall(content)
            import_matches = list(import_re.finditer(content))

            if not sys_path_matches and not import_matches:
                self.log.info("circular_import_already_fixed",
                              data={"file": file_path})
                return self._make_result(
                    "fix_circular_import", True,
                    data={"already_fixed": True},
                )

            lines_changed = 0
            old_patterns: List[str] = []

            # 1. 删除 sys.path hack 行
            if sys_path_matches:
                old_patterns.extend(sys_path_matches)
                lines_changed += len(sys_path_matches)
                content = sys_path_re.sub("", content)

            # 2. 替换 from core.xxx import -> from ..core.xxx import
            new_patterns: List[str] = []
            def _replace_import(m: re.Match) -> str:
                nonlocal lines_changed
                indent = m.group(1)
                module = m.group(2)
                names = m.group(3)
                lines_changed += 1
                new_line = f"{indent}from ..core.{module} import {names}"
                new_patterns.append(new_line)
                return new_line

            content = import_re.sub(_replace_import, content)

            # 清理多余空行（连续 3+ 换行 -> 2 换行）
            content = re.sub(r"\n{3,}", "\n\n", content)

            # 写回文件
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            # 验证闭环：用 ast 确认修改后文件仍是合法 Python
            try:
                ast.parse(content, filename=full_path)
            except SyntaxError as exc:
                self.log.error("circular_import_syntax_error",
                               data={"file": file_path, "error": str(exc)})
                return self._make_result(
                    "fix_circular_import", False,
                    error=f"Syntax error after fix: {exc}",
                )

            old_pattern_str = "; ".join(p.strip() for p in old_patterns) if old_patterns else "(none)"
            new_pattern_str = "; ".join(new_patterns) if new_patterns else "(sys.path removed)"

            self.log.info("circular_import_fixed", data={
                "file": file_path, "lines_changed": lines_changed,
            })
            self.metrics.inc("fix.circular_import_fixed")

            return self._make_result(
                "fix_circular_import", True,
                data={
                    "file": file_path,
                    "old_pattern": old_pattern_str,
                    "new_pattern": new_pattern_str,
                    "lines_changed": lines_changed,
                },
            )

    # ==================================================================
    # 3. 上下文压缩与 LLM 输入脱节修复
    # ==================================================================
    def fix_context_desync(self) -> AgentResult:
        """修复上下文压缩与 LLM 输入脱节问题。

        读取 ``aerospace_agent/core/agent.py`` 的 ``run_react_stream`` 方法，
        在每步循环开始处添加 messages 截断逻辑：
        如果 len(messages) > max_context_messages（50 条），则保留 system + 最近 N 条。

        用 SearchReplace 修改，不重写整个文件。

        Returns:
            AgentResult, data 格式::
                {"file": "agent.py", "fix": "context_truncation", "details": "..."}
        """
        with self._time_operation("fix_context_desync"):
            rel_path = os.path.join("aerospace_agent", "core", "agent.py")
            content = self._read_file(rel_path)
            if content is None:
                return self._make_result(
                    "fix_context_desync", False,
                    error=f"Cannot read file: {rel_path}",
                )

            # 检查是否已修复
            if "_max_context_messages" in content or "context_truncated" in content:
                self.log.info("context_desync_already_fixed")
                return self._make_result(
                    "fix_context_desync", True,
                    data={
                        "file": "agent.py",
                        "fix": "context_truncation",
                        "details": "already_fixed",
                    },
                )

            # 定向搜索：在 run_react_stream 的循环开头插入截断逻辑
            old_str = (
                "        for step in range(1, max_steps + 1):\n"
                "            _steps_used = step\n"
                "            # 检查上下文是否需要压缩"
            )

            new_str = (
                "        for step in range(1, max_steps + 1):\n"
                "            _steps_used = step\n"
                "            # --- 上下文截断：防止 messages 列表无限增长导致 LLM 输入脱节 ---\n"
                "            _max_context_messages = 50\n"
                "            if len(messages) > _max_context_messages:\n"
                "                _system_msg = messages[0] if messages and messages[0].get(\"role\") == \"system\" else None\n"
                "                _keep_recent = _max_context_messages - (1 if _system_msg else 0)\n"
                "                _recent = messages[-_keep_recent:] if _keep_recent > 0 else []\n"
                "                messages = ([_system_msg] + list(_recent)) if _system_msg else list(_recent)\n"
                "                log.info(\"context_truncated\",\n"
                "                         data={\"max\": _max_context_messages, \"remaining\": len(messages)})\n"
                "            # 检查上下文是否需要压缩"
            )

            success, error = self._apply_search_replace(rel_path, old_str, new_str)
            if not success:
                return self._make_result(
                    "fix_context_desync", False,
                    error=error,
                )

            self.log.info("context_desync_fixed", data={"file": rel_path})
            self.metrics.inc("fix.context_desync_fixed")

            return self._make_result(
                "fix_context_desync", True,
                data={
                    "file": "agent.py",
                    "fix": "context_truncation",
                    "details": (
                        "Added messages truncation at loop start: "
                        "if len(messages) > 50, keep system + recent 49 messages. "
                        "Prevents context compression desync with LLM input."
                    ),
                },
            )

    # ==================================================================
    # 4. YAML 工作流变量插值
    # ==================================================================
    def fix_yaml_interpolation(self) -> AgentResult:
        """实现 YAML 工作流变量插值。

        读取 ``aerospace_agent/mcp/schemas/workflow.py`` 的 ``from_yaml_dict`` 方法，
        添加 ``_resolve_variables`` 静态方法：递归遍历 dict/list，
        将 ``${inputs.x}`` / ``${steps.y.outputs.z}`` 替换为实际值。
        用正则 ``r"\\$\\{([^}]+)\\}"`` 匹配变量引用。

        Returns:
            AgentResult, data 格式::
                {"file": "workflow.py", "method_added": "_resolve_variables", "pattern": "..."}
        """
        with self._time_operation("fix_yaml_interpolation"):
            rel_path = os.path.join("aerospace_agent", "mcp", "schemas", "workflow.py")
            content = self._read_file(rel_path)
            if content is None:
                return self._make_result(
                    "fix_yaml_interpolation", False,
                    error=f"Cannot read file: {rel_path}",
                )

            # 检查是否已修复
            if "_resolve_variables" in content:
                self.log.info("yaml_interpolation_already_fixed")
                return self._make_result(
                    "fix_yaml_interpolation", True,
                    data={
                        "file": "workflow.py",
                        "method_added": "_resolve_variables",
                        "pattern": r"\$\{([^}]+)\}",
                        "details": "already_fixed",
                    },
                )

            # --- 变更 1：添加 import re ---
            old_import = "from __future__ import annotations\n\nfrom dataclasses import dataclass, field"
            new_import = "from __future__ import annotations\n\nimport re\nfrom dataclasses import dataclass, field"

            success, error = self._apply_search_replace(rel_path, old_import, new_import)
            if not success:
                return self._make_result(
                    "fix_yaml_interpolation", False,
                    error=f"Failed to add import re: {error}",
                )

            # 重新读取更新后的内容
            content = self._read_file(rel_path)
            if content is None:
                return self._make_result(
                    "fix_yaml_interpolation", False,
                    error="Cannot re-read file after first edit",
                )

            # --- 变更 2：在 from_yaml_dict 中调用 _resolve_variables ---
            old_from_yaml = (
                '    @classmethod\n'
                '    def from_yaml_dict(cls, d: dict) -> "WorkflowSpec":\n'
                '        """从 YAML 解析后的字典构造（兼容 workflow.yaml 格式）。"""\n'
                '        return cls('
            )
            new_from_yaml = (
                '    @classmethod\n'
                '    def from_yaml_dict(cls, d: dict) -> "WorkflowSpec":\n'
                '        """从 YAML 解析后的字典构造（兼容 workflow.yaml 格式）。\n'
                '\n'
                '        会调用 _resolve_variables 解析 ${inputs.x} / ${steps.y.outputs.z}\n'
                '        等变量引用，将其替换为实际值。\n'
                '        """\n'
                '        d = cls._resolve_variables(d, context=d)\n'
                '        return cls('
            )

            success, error = self._apply_search_replace(rel_path, old_from_yaml, new_from_yaml)
            if not success:
                return self._make_result(
                    "fix_yaml_interpolation", False,
                    error=f"Failed to modify from_yaml_dict: {error}",
                )

            # 重新读取
            content = self._read_file(rel_path)
            if content is None:
                return self._make_result(
                    "fix_yaml_interpolation", False,
                    error="Cannot re-read file after second edit",
                )

            # --- 变更 3：在 from_yaml_dict 方法后添加 _resolve_variables 静态方法 ---
            # 插入点：from_yaml_dict 的 return 语句结束后、ValidationReport 类之前
            old_anchor = (
                "            metadata=d.get(\"metadata\", {}),\n"
                "        )\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class ValidationReport:"
            )
            new_anchor = (
                "            metadata=d.get(\"metadata\", {}),\n"
                "        )\n"
                "\n"
                "    @staticmethod\n"
                "    def _resolve_variables(obj: Any, context: Optional[dict] = None) -> Any:\n"
                "        \"\"\"递归解析变量引用 ``${inputs.x}`` / ``${steps.y.outputs.z}``。\n"
                "\n"
                "        遍历 dict/list 结构，将 ``${path.to.value}`` 替换为 context\n"
                "        中对应的实际值。无法解析的引用保持原样。\n"
                "\n"
                "        Args:\n"
                "            obj: 待解析的对象（dict / list / str / 其他）\n"
                "            context: 变量上下文（如 workflow 字典本身）\n"
                "\n"
                "        Returns:\n"
                "            解析后的对象（深拷贝，不修改原始对象）\n"
                "        \"\"\"\n"
                "        if context is None:\n"
                "            context = {}\n"
                "        _var_pattern = re.compile(r\"\\$\\{([^}]+)\\}\")\n"
                "\n"
                "        def _resolve(value):\n"
                "            if isinstance(value, str):\n"
                "                def _replacer(m):\n"
                "                    path = m.group(1).strip()\n"
                "                    parts = path.split(\".\")\n"
                "                    current = context\n"
                "                    for part in parts:\n"
                "                        if isinstance(current, dict) and part in current:\n"
                "                            current = current[part]\n"
                "                        else:\n"
                "                            return m.group(0)  # 无法解析，保持原样\n"
                "                    return str(current)\n"
                "                return _var_pattern.sub(_replacer, value)\n"
                "            if isinstance(value, dict):\n"
                "                return {k: _resolve(v) for k, v in value.items()}\n"
                "            if isinstance(value, list):\n"
                "                return [_resolve(item) for item in value]\n"
                "            return value\n"
                "\n"
                "        return _resolve(obj)\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class ValidationReport:"
            )

            success, error = self._apply_search_replace(rel_path, old_anchor, new_anchor)
            if not success:
                return self._make_result(
                    "fix_yaml_interpolation", False,
                    error=f"Failed to add _resolve_variables method: {error}",
                )

            self.log.info("yaml_interpolation_fixed", data={"file": rel_path})
            self.metrics.inc("fix.yaml_interpolation_fixed")

            return self._make_result(
                "fix_yaml_interpolation", True,
                data={
                    "file": "workflow.py",
                    "method_added": "_resolve_variables",
                    "pattern": r"\$\{([^}]+)\}",
                    "details": (
                        "Added _resolve_variables static method to WorkflowSpec. "
                        "Recursively resolves ${inputs.x} / ${steps.y.outputs.z} "
                        "variable references in dict/list structures. "
                        "Called from from_yaml_dict before construction."
                    ),
                },
            )

    # ==================================================================
    # 5. J2 摄动占位修复
    # ==================================================================
    def fix_j2_placeholder(self) -> AgentResult:
        """修复 MCP propagate_orbit 的 J2 摄动占位。

        读取 ``aerospace_agent/mcp/tools/propagation_tools.py`` 的 ``_j2_step`` 方法，
        实现真实的 J2 平均摄动：

        - J2 引起的 RAAN 进动: dOmega/dt = -3/2 * J2 * (R_E/a)^2 * n * cos(i)
        - J2 引起的 argp 漂移: dargp/dt = 3/4 * J2 * (R_E/a)^2 * n * (5*cos^2(i) - 1)

        其中 J2=1.08263e-3, R_E=6378137m, n=sqrt(mu/a^3)

        在 _j2_step 中用 RK4 积分这些长期效应。

        Returns:
            AgentResult, data 格式::
                {"file": "propagation_tools.py", "method": "_j2_step", "implementation": "rk4_mean_j2"}
        """
        with self._time_operation("fix_j2_placeholder"):
            rel_path = os.path.join("aerospace_agent", "mcp", "tools", "propagation_tools.py")
            content = self._read_file(rel_path)
            if content is None:
                return self._make_result(
                    "fix_j2_placeholder", False,
                    error=f"Cannot read file: {rel_path}",
                )

            # 检查是否已修复（已修复的版本包含 RK4 标记和 J2 常量）
            if "1.08263e-3" in content and "rk4_mean_j2" in content:
                self.log.info("j2_placeholder_already_fixed")
                return self._make_result(
                    "fix_j2_placeholder", True,
                    data={
                        "file": "propagation_tools.py",
                        "method": "_j2_step",
                        "implementation": "rk4_mean_j2",
                        "details": "already_fixed",
                    },
                )

            # 定向替换 _j2_step 函数体
            old_str = (
                "def _j2_step(r, v, dt, mu):\n"
                '    """J2 摄动传播（简化——仅在二体基础上叠加 J2 平均效应）。"""\n'
                "    # 简化：先用二体传播，J2 效应在当前版本为占位\n"
                "    return _kepler_step(r, v, dt, mu)"
            )

            new_str = (
                "def _j2_step(r, v, dt, mu):\n"
                '    """J2 摄动传播 -- RK4 积分 J2 平均长期效应 (rk4_mean_j2).\n'
                "\n"
                "    J2 引起的 RAAN 进动: dOmega/dt = -3/2 * J2 * (R_E/a)^2 * n * cos(i)\n"
                "    J2 引起的 argp 漂移: dargp/dt = 3/4 * J2 * (R_E/a)^2 * n * (5*cos^2(i) - 1)\n"
                "    其中 J2=1.08263e-3, R_E=6378137m, n=sqrt(mu/a^3)\n"
                "\n"
                "    步骤：\n"
                "      1. 笛卡尔 -> 开普勒元素 (a, e, i, Omega, omega, nu)\n"
                "      2. RK4 积分 Omega 和 omega 的 J2 长期漂移\n"
                "      3. 二体推进真近点角 nu\n"
                "      4. 开普勒元素 -> 笛卡尔 (用更新后的 Omega, omega, nu)\n"
                '    """\n'
                "    J2 = 1.08263e-3\n"
                "    R_E = 6378137.0  # m\n"
                "\n"
                "    if abs(dt) < 1e-12:\n"
                "        return list(r), list(v)\n"
                "\n"
                "    # --- 1. 笛卡尔 -> 开普勒元素 ---\n"
                "    r_mag = math.sqrt(sum(x * x for x in r))\n"
                "    v_mag = math.sqrt(sum(x * x for x in v))\n"
                "\n"
                "    # 角动量向量 h = r x v\n"
                "    h = [r[1] * v[2] - r[2] * v[1],\n"
                "         r[2] * v[0] - r[0] * v[2],\n"
                "         r[0] * v[1] - r[1] * v[0]]\n"
                "    h_mag = math.sqrt(sum(x * x for x in h))\n"
                "\n"
                "    # 半长轴\n"
                "    energy = v_mag * v_mag / 2.0 - mu / r_mag\n"
                "    a = -mu / (2.0 * energy)\n"
                "\n"
                "    # 偏心率向量\n"
                "    rdotv = sum(r[j] * v[j] for j in range(3))\n"
                "    e_vec = [(v_mag * v_mag - mu / r_mag) * r[i] / mu\n"
                "             - rdotv * v[i] / mu for i in range(3)]\n"
                "    e = math.sqrt(sum(x * x for x in e_vec))\n"
                "\n"
                "    # 倾角\n"
                "    cos_i = max(-1.0, min(1.0, h[2] / h_mag)) if h_mag > 0 else 0.0\n"
                "    i_rad = math.acos(cos_i)\n"
                "\n"
                "    # 升交点赤经 (RAAN)\n"
                "    n_vec = [-h[1], h[0], 0.0]  # 节点向量 n = k x h\n"
                "    n_mag = math.sqrt(n_vec[0] ** 2 + n_vec[1] ** 2)\n"
                "    if n_mag > 1e-10:\n"
                "        raan = math.acos(max(-1.0, min(1.0, n_vec[0] / n_mag)))\n"
                "        if n_vec[1] < 0:\n"
                "            raan = 2 * math.pi - raan\n"
                "    else:\n"
                "        raan = 0.0\n"
                "\n"
                "    # 近地点幅角 (argp)\n"
                "    if n_mag > 1e-10 and e > 1e-10:\n"
                "        argp = math.acos(max(-1.0, min(1.0,\n"
                "            sum(n_vec[i] * e_vec[i] for i in range(3)) / (n_mag * e))))\n"
                "        if e_vec[2] < 0:\n"
                "            argp = 2 * math.pi - argp\n"
                "    else:\n"
                "        argp = 0.0\n"
                "\n"
                "    # 真近点角 (true anomaly)\n"
                "    if e > 1e-10:\n"
                "        ta = math.acos(max(-1.0, min(1.0,\n"
                "            sum(e_vec[i] * r[i] for i in range(3)) / (e * r_mag))))\n"
                "        if rdotv < 0:\n"
                "            ta = 2 * math.pi - ta\n"
                "    else:\n"
                "        # 圆轨道：从升交点幅角推算\n"
                "        if n_mag > 1e-10:\n"
                "            u = math.acos(max(-1.0, min(1.0,\n"
                "                sum(n_vec[i] * r[i] for i in range(3)) / (n_mag * r_mag))))\n"
                "            ta = u - argp\n"
                "        else:\n"
                "            ta = 0.0\n"
                "\n"
                "    # --- 2. J2 长期摄动率 ---\n"
                "    n_motion = math.sqrt(mu / a ** 3)  # 平均运动\n"
                "    factor = 1.5 * J2 * (R_E / a) ** 2 * n_motion\n"
                "    d_raan_dt = -factor * math.cos(i_rad)\n"
                "    d_argp_dt = factor * 0.5 * (5 * math.cos(i_rad) ** 2 - 1)\n"
                "\n"
                "    # --- 3. RK4 积分 [raan, argp] 过 dt ---\n"
                "    # （长期率为常数，RK4 退化为精确解，但遵循 RK4 模式）\n"
                "    def _derivs(state):\n"
                "        return [d_raan_dt, d_argp_dt]\n"
                "\n"
                "    s0 = [raan, argp]\n"
                "    k1 = _derivs(s0)\n"
                "    k2 = _derivs([s0[j] + 0.5 * dt * k1[j] for j in range(2)])\n"
                "    k3 = _derivs([s0[j] + 0.5 * dt * k2[j] for j in range(2)])\n"
                "    k4 = _derivs([s0[j] + dt * k3[j] for j in range(2)])\n"
                "    s1 = [s0[j] + dt / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])\n"
                "           for j in range(2)]\n"
                "    raan_new = s1[0]\n"
                "    argp_new = s1[1]\n"
                "\n"
                "    # --- 4. 二体推进真近点角 ---\n"
                "    if e < 1e-10:\n"
                "        ta_new = ta + n_motion * dt\n"
                "    else:\n"
                "        # 真近点角 -> 偏近点角 E\n"
                "        E = 2 * math.atan2(\n"
                "            math.sqrt(1 - e) * math.sin(ta / 2),\n"
                "            math.sqrt(1 + e) * math.cos(ta / 2))\n"
                "        M = E - e * math.sin(E)\n"
                "        M_new = M + n_motion * dt\n"
                "        # Newton-Raphson 求解 Kepler 方程\n"
                "        E_new = M_new\n"
                "        for _ in range(20):\n"
                "            dE = (E_new - e * math.sin(E_new) - M_new) / (1 - e * math.cos(E_new))\n"
                "            E_new -= dE\n"
                "            if abs(dE) < 1e-12:\n"
                "                break\n"
                "        ta_new = 2 * math.atan2(\n"
                "            math.sqrt(1 + e) * math.sin(E_new / 2),\n"
                "            math.sqrt(1 - e) * math.cos(E_new / 2))\n"
                "\n"
                "    # --- 5. 开普勒元素 -> 笛卡尔 ---\n"
                "    p = a * (1 - e ** 2)\n"
                "    r_orb = p / (1 + e * math.cos(ta_new))\n"
                "    # 近焦点坐标系 (PQW) 中的位置和速度\n"
                "    px = r_orb * math.cos(ta_new)\n"
                "    py = r_orb * math.sin(ta_new)\n"
                "    pz = 0.0\n"
                "    if e < 1e-10:\n"
                "        v_circ = math.sqrt(mu / a)\n"
                "        vx = -v_circ * math.sin(ta_new)\n"
                "        vy = v_circ * math.cos(ta_new)\n"
                "    else:\n"
                "        vx = math.sqrt(mu / p) * (-math.sin(ta_new))\n"
                "        vy = math.sqrt(mu / p) * (e + math.cos(ta_new))\n"
                "    vz = 0.0\n"
                "\n"
                "    # 旋转矩阵 R = Rz(raan) * Rx(i) * Rz(argp)\n"
                "    cos_O = math.cos(raan_new)\n"
                "    sin_O = math.sin(raan_new)\n"
                "    cos_w = math.cos(argp_new)\n"
                "    sin_w = math.sin(argp_new)\n"
                "    cos_i = math.cos(i_rad)\n"
                "    sin_i = math.sin(i_rad)\n"
                "\n"
                "    R11 = cos_O * cos_w - sin_O * sin_w * cos_i\n"
                "    R12 = -cos_O * sin_w - sin_O * cos_w * cos_i\n"
                "    R13 = sin_O * sin_i\n"
                "    R21 = sin_O * cos_w + cos_O * sin_w * cos_i\n"
                "    R22 = -sin_O * sin_w + cos_O * cos_w * cos_i\n"
                "    R23 = -cos_O * sin_i\n"
                "    R31 = sin_w * sin_i\n"
                "    R32 = cos_w * sin_i\n"
                "    R33 = cos_i\n"
                "\n"
                "    new_r = [R11 * px + R12 * py + R13 * pz,\n"
                "             R21 * px + R22 * py + R23 * pz,\n"
                "             R31 * px + R32 * py + R33 * pz]\n"
                "    new_v = [R11 * vx + R12 * vy + R13 * vz,\n"
                "             R21 * vx + R22 * vy + R23 * vz,\n"
                "             R31 * vx + R32 * vy + R33 * vz]\n"
                "    return new_r, new_v"
            )

            success, error = self._apply_search_replace(rel_path, old_str, new_str)
            if not success:
                return self._make_result(
                    "fix_j2_placeholder", False,
                    error=error,
                )

            self.log.info("j2_placeholder_fixed", data={"file": rel_path})
            self.metrics.inc("fix.j2_placeholder_fixed")

            return self._make_result(
                "fix_j2_placeholder", True,
                data={
                    "file": "propagation_tools.py",
                    "method": "_j2_step",
                    "implementation": "rk4_mean_j2",
                    "details": (
                        "Replaced placeholder with real J2 mean perturbation: "
                        "cartesian->keplerian, RK4 integration of RAAN/argp secular drifts "
                        "(dOmega/dt = -3/2*J2*(R_E/a)^2*n*cos(i), "
                        "dargp/dt = 3/4*J2*(R_E/a)^2*n*(5cos^2(i)-1)), "
                        "two-body true anomaly advance, keplerian->cartesian."
                    ),
                },
            )

    # ==================================================================
    # 6. 技能系统接入 ReAct 循环
    # ==================================================================
    def fix_skill_integration(self) -> AgentResult:
        """将技能系统接入 ReAct 循环。

        读取 ``aerospace_agent/core/agent.py`` 的 ``run_react_stream``，
        在 system prompt 构建处（build_context 调用后）添加技能描述注入：
        如果 agent.skills 非空，把技能名称和描述追加到 system prompt。

        Returns:
            AgentResult, data 格式::
                {"file": "agent.py", "fix": "skill_injection", "skills_count": N}
        """
        with self._time_operation("fix_skill_integration"):
            rel_path = os.path.join("aerospace_agent", "core", "agent.py")
            content = self._read_file(rel_path)
            if content is None:
                return self._make_result(
                    "fix_skill_integration", False,
                    error=f"Cannot read file: {rel_path}",
                )

            # 检查是否已修复
            if "_skill_lines" in content or "可用技能" in content:
                self.log.info("skill_integration_already_fixed")
                return self._make_result(
                    "fix_skill_integration", True,
                    data={
                        "file": "agent.py",
                        "fix": "skill_injection",
                        "skills_count": 0,
                        "details": "already_fixed",
                    },
                )

            # 在 build_context 调用后、system_prompt 拼接前注入技能描述
            old_str = (
                "        if enable_context:\n"
                "            ctx = self.context_manager.build_context(token_budget=8000)\n"
                "            if ctx:\n"
                '                system_parts.append(f"\\n## 上下文\\n{ctx}")\n'
                '        system_prompt = "\\n".join(system_parts)'
            )

            new_str = (
                "        if enable_context:\n"
                "            ctx = self.context_manager.build_context(token_budget=8000)\n"
                "            if ctx:\n"
                '                system_parts.append(f"\\n## 上下文\\n{ctx}")\n'
                "        # --- 注入技能描述（将技能系统接入 ReAct 循环）---\n"
                "        _skills = getattr(self, \"skills\", None)\n"
                "        if _skills is not None:\n"
                "            _skill_lines = []\n"
                "            _list_fn = getattr(_skills, \"list_skills\", None)\n"
                "            if callable(_list_fn):\n"
                "                for _sk in _list_fn():\n"
                "                    if isinstance(_sk, dict):\n"
                "                        _name = _sk.get(\"name\", str(_sk))\n"
                "                        _desc = _sk.get(\"description\", \"\")\n"
                "                        _skill_lines.append(\n"
                "                            f\"- {_name}: {_desc}\" if _desc else f\"- {_name}\")\n"
                "                    else:\n"
                "                        _skill_lines.append(f\"- {_sk}\")\n"
                "            if _skill_lines:\n"
                '                system_parts.append("\\n## 可用技能\\n" + "\\n".join(_skill_lines))\n'
                '        system_prompt = "\\n".join(system_parts)'
            )

            success, error = self._apply_search_replace(rel_path, old_str, new_str)
            if not success:
                return self._make_result(
                    "fix_skill_integration", False,
                    error=error,
                )

            # 统计可用技能数量
            skills_count = 0
            try:
                from ..skills import SkillRegistry
                _reg = SkillRegistry()
                skills_count = _reg.auto_discover()
            except Exception:
                pass

            self.log.info("skill_integration_fixed", data={
                "file": rel_path, "skills_count": skills_count,
            })
            self.metrics.inc("fix.skill_integration_fixed")
            self.metrics.gauge("fix.skills_count", skills_count)

            return self._make_result(
                "fix_skill_integration", True,
                data={
                    "file": "agent.py",
                    "fix": "skill_injection",
                    "skills_count": skills_count,
                    "details": (
                        "Injected skill descriptions into system prompt of run_react_stream. "
                        "If agent.skills is non-empty, skill name and description are appended "
                        "to system_parts after build_context call."
                    ),
                },
            )

    # ==================================================================
    # 7. 批量执行修复计划
    # ==================================================================
    def apply_fixes(self, fix_plan: List[Dict]) -> AgentResult:
        """批量执行修复计划。

        遍历 fix_plan，按 strategy 调用对应的 fix_* 方法，收集所有结果。

        Args:
            fix_plan: 修复计划列表，每项格式::
                {"id": ..., "strategy": "direct_edit|refactor|new_file|config_change",
                 "priority": ..., "estimated_complexity": "low/medium/high",
                 "file": ..., "issue": ...}

        Returns:
            AgentResult, data 格式::
                {"total_fixes": N, "successful": N, "failed": N, "results": [...]}
        """
        with self._time_operation("apply_fixes"):
            results: List[Dict[str, Any]] = []
            successful = 0
            failed = 0

            for item in fix_plan:
                strategy = item.get("strategy", "")
                issue_id = item.get("id", "")
                issue_text = str(item.get("issue", "")).lower()
                file_path = item.get("file", "")

                try:
                    result: AgentResult

                    if strategy == "direct_edit":
                        # 循环依赖修复
                        if file_path:
                            result = self.fix_circular_import(file_path)
                        else:
                            result = self._make_result(
                                "apply_fixes", True,
                                data={"id": issue_id, "skipped": True,
                                      "reason": "no file specified for direct_edit"},
                            )

                    elif strategy == "config_change":
                        # YAML 变量插值
                        result = self.fix_yaml_interpolation()

                    elif strategy == "refactor":
                        # 根据 issue 内容分发到具体修复方法
                        if any(kw in issue_text for kw in
                               ("context", "上下文", "脱节", "desync")):
                            result = self.fix_context_desync()
                        elif any(kw in issue_text for kw in
                                 ("j2", "摄动", "placeholder", "占位")):
                            result = self.fix_j2_placeholder()
                        elif any(kw in issue_text for kw in
                                 ("skill", "技能")):
                            result = self.fix_skill_integration()
                        else:
                            # 默认尝试上下文修复
                            result = self.fix_context_desync()

                    elif strategy == "new_file":
                        # 技能系统接入
                        result = self.fix_skill_integration()

                    else:
                        result = self._make_result(
                            "apply_fixes", False,
                            error=f"Unknown strategy: {strategy}",
                            data={"id": issue_id},
                        )

                    results.append(result.to_dict())
                    if result.success:
                        successful += 1
                    else:
                        failed += 1

                except Exception as exc:
                    failed += 1
                    results.append({
                        "id": issue_id,
                        "success": False,
                        "error": str(exc),
                    })
                    self.log.error("fix_failed", data={
                        "id": issue_id, "error": str(exc),
                    })

            total = len(fix_plan)
            self.log.info("fixes_applied", data={
                "total": total, "successful": successful, "failed": failed,
            })
            self.metrics.gauge("fix.applied_total", total)
            self.metrics.gauge("fix.applied_successful", successful)
            self.metrics.gauge("fix.applied_failed", failed)

            return self._make_result(
                "apply_fixes", failed == 0,
                data={
                    "total_fixes": total,
                    "successful": successful,
                    "failed": failed,
                    "results": results,
                },
            )

    # ==================================================================
    # 辅助方法
    # ==================================================================
    def _resolve_path(self, file_path: str) -> str:
        """将相对路径解析为绝对路径。

        Args:
            file_path: 文件路径（相对或绝对）

        Returns:
            绝对路径
        """
        if os.path.isabs(file_path):
            return file_path
        return os.path.join(self.project_root, file_path)

    def _read_file(self, file_path: str) -> Optional[str]:
        """读取文件内容。

        Args:
            file_path: 文件路径（相对于 project_root 或绝对路径）

        Returns:
            文件内容字符串；文件不存在时返回 None
        """
        full_path = self._resolve_path(file_path)
        if not os.path.exists(full_path):
            return None
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    def _apply_search_replace(
        self, file_path: str, old_str: str, new_str: str,
    ) -> Tuple[bool, str]:
        """在文件中执行搜索替换（最小变更原则）。

        读取文件，找到 old_str 的首次出现，替换为 new_str，写回文件。
        仅替换第一个匹配项（与 SearchReplace 工具行为一致）。

        Args:
            file_path: 文件路径（相对于 project_root 或绝对路径）
            old_str: 要搜索的文本块
            new_str: 替换后的文本块

        Returns:
            (success, error_message) -- 成功时 error_message 为空字符串
        """
        full_path = self._resolve_path(file_path)

        if not os.path.exists(full_path):
            return False, f"File not found: {full_path}"

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            return False, f"Cannot read file {full_path}: {exc}"

        if old_str not in content:
            return False, f"Pattern not found in {file_path}"

        # 仅替换第一个匹配项
        new_content = content.replace(old_str, new_str, 1)

        # 验证闭环：对 .py 文件用 ast 确认修改后仍是合法 Python
        if full_path.endswith(".py"):
            try:
                ast.parse(new_content, filename=full_path)
            except SyntaxError as exc:
                # 语法校验失败 -- 回滚原始内容
                try:
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception:
                    pass
                return False, f"Syntax error after replace in {file_path}: {exc}"

        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as exc:
            return False, f"Cannot write file {full_path}: {exc}"

        return True, ""

    def _determine_strategy(self, issue: Dict) -> str:
        """根据问题内容确定修复策略。

        Args:
            issue: 问题字典 {"id", "file", "issue", "severity"}

        Returns:
            修复策略: "direct_edit" | "refactor" | "new_file" | "config_change"
        """
        issue_text = (
            str(issue.get("issue", "")) + " " + str(issue.get("file", ""))
        ).lower()
        for keyword, strategy in self._STRATEGY_KEYWORDS.items():
            if keyword in issue_text:
                return strategy
        return "direct_edit"  # 默认策略

    def _estimate_complexity(self, issue: Dict, strategy: str) -> str:
        """估算修复复杂度。

        Args:
            issue: 问题字典
            strategy: 修复策略

        Returns:
            "low" | "medium" | "high"
        """
        severity = str(issue.get("severity", "low")).lower()
        if strategy == "new_file" or severity == "critical":
            return "high"
        if strategy == "refactor" or severity == "high":
            return "medium"
        return "low"
