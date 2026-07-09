"""TestAgent — 测试覆盖分析 · 契约验证 · CI门禁。

第一性原理设计:

  1. 测试是契约的执行证据
     每个公开函数的测试是它与调用方契约的可验证表达。无测试的公开 API
     等同于未经验证的承诺——调用方无法信任其行为。TestAgent 通过
     scan_coverage 将"无测试"从隐性风险变为显性指标。

  2. 覆盖率是风险代理指标
     无测试的模块是未知风险区。但并非所有模块同等重要——按关键路径优先级
     （adapters > loop/engine > rag/router > validation_tools > cli_tui）
     排序，优先保障核心链路。覆盖率数字本身不是目标，关键路径的覆盖才是。

  3. 回归是修复的副作用
     FixAgent 的每次修复必须验证不引入新失败。"之前通过现在失败"的测试
     是回归，severity=critical，阻断发布。verify_fix 通过基线对比实现
     这一检测，形成 修复→验证→回归检测 的闭环。

  4. CI 门禁是质量底线
     没有 CI 的项目无法保证"持续可发布"状态。CI 缺失 = 人工验证 =
     不可重复 = 不可信。check_ci_readiness 检查 CI 配置、测试依赖、
     .gitignore 规则，为编排器提供门禁信号。

  5. 计划优于代码
     先分析测试要点再写测试，避免盲目追求覆盖率数字而忽略边界条件与
     错误路径。generate_test_plan 对每个模块提取公开 API、生成测试
     要点和估算用例数，但不创建测试文件——计划由人工或 FixAgent 执行。

TestAgent 在三 Agent 架构中的定位:
  - 为 FixAgent   提供验证闭环（verify_fix）：修复 → 验证 → 回归检测
  - 为 ArchAgent  提供架构红线（K4 关键路径必须有回归测试）
  - 为编排器      提供 CI 门禁信号（check_ci_readiness）

设计约束:
  - 不创建测试文件，只生成测试计划
  - 所有方法返回 AgentResult，便于编排器统一决策
  - 用 self.log / self.metrics 记录观测数据，用 self._time_operation() 计时
  - import ast, os, subprocess, glob（+ re / sys 用于解析与进程调用）
"""
from __future__ import annotations

import ast
import glob
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from aerospace_agent.local_runtime import run_command

from .base import AgentBase, AgentResult, AgentRole


class TestAgent(AgentBase):
    """测试 Agent — 测试覆盖分析、契约验证与 CI 门禁。

    职责边界:
      - scan_coverage:        扫描无测试模块，计算覆盖率
      - analyze_test_quality: 评估现有测试工程质量
      - generate_test_plan:   为指定模块生成测试计划（不生成代码）
      - run_tests:            运行 pytest 并收集结果
      - verify_fix:           验证修复是否引入回归
      - check_ci_readiness:   检查 CI/CD 就绪度
    """

    __test__ = False  # 防止 pytest 误收集
    role = AgentRole.TEST

    # 关键模块优先级（路径片段 → 优先级，1 = 最高）
    # 排序依据: 越靠近数据/计算核心，回归影响越大
    _KEY_MODULE_PRIORITY: List[Tuple[str, int]] = [
        ("mcp/adapters", 1),           # adapters — 引擎适配，核心抽象层
        ("mcp/loop", 2),               # loop/engine — 工作流引擎，递归控制
        ("rag/router", 3),             # rag/router — 检索路由，决策中枢
        ("mcp/tools/validation", 4),   # validation_tools — 输入验证，安全门
        ("cli_tui", 5),               # cli_tui — 用户交互，外部入口
    ]

    # 扫描时排除的目录名（不纳入覆盖率统计）
    _EXCLUDE_DIRS = {"__pycache__", "build", "tests", ".pytest_cache", ".git"}

    # pytest 配置文件检查清单
    _PYTEST_CONFIG_FILES = ("pytest.ini", "setup.cfg", "pyproject.toml", "tox.ini")

    def __init__(self, project_root: str = "."):
        super().__init__(project_root)
        # 测试基线（test_id → status），用于回归检测
        # 首次 verify_fix 建立基线，后续调用对比
        self._baseline: Dict[str, str] = {}

    # ================================================================
    # 1. 覆盖率扫描
    # ================================================================

    def scan_coverage(self) -> AgentResult:
        """扫描项目源文件，找出无测试的模块。

        匹配规则: 源文件 ``foo/bar.py`` → 测试文件 ``test_bar.py``（任意位置）。
        仅对关键路径模块（adapters / loop / router 等）的缺失生成
        ``severity="high"`` 的 issue。
        """
        with self._time_operation("scan_coverage"):
            source_files = self._collect_source_files()
            test_basenames = self._collect_test_basenames()

            tested: List[str] = []
            untested: List[str] = []
            for sf in source_files:
                stem = os.path.splitext(os.path.basename(sf))[0]
                expected_test = f"test_{stem}.py"
                if expected_test in test_basenames:
                    tested.append(sf)
                else:
                    untested.append(sf)

            total = len(source_files)
            tested_n = len(tested)
            coverage_pct = round(
                (tested_n / total * 100) if total > 0 else 0.0, 2)

            # 仅为无测试的关键模块生成 issue
            issues: List[Dict] = []
            for sf in untested:
                priority = self._get_module_priority(sf)
                if priority is not None:
                    issues.append({
                        "file": sf,
                        "issue": "no_test",
                        "severity": "high",
                        "module_priority": priority,
                        "suggestion": "为关键模块添加测试，保障核心链路可验证",
                    })

            data: Dict[str, Any] = {
                "total_source_files": total,
                "tested_files": tested_n,
                "untested_files": untested,
                "coverage_pct": coverage_pct,
            }

            self.metrics.gauge("agent.test.coverage_pct", coverage_pct,
                               tags={"scope": "package"})
            self.metrics.gauge("agent.test.untested_count", len(untested))
            self.log.info("scan_coverage_complete", data={
                "total": total, "tested": tested_n,
                "coverage_pct": coverage_pct,
                "key_module_issues": len(issues)})

            return self._make_result(
                "scan_coverage", success=True,
                data=data, issues=issues)

    # ================================================================
    # 2. 测试质量分析
    # ================================================================

    def analyze_test_quality(self) -> AgentResult:
        """分析现有测试的工程质量。

        检查: conftest.py / pytest 配置 / @pytest.mark.parametrize /
        mock·patch / @pytest.fixture。
        """
        with self._time_operation("analyze_test_quality"):
            # --- conftest.py ---
            conftest_pattern = os.path.join(
                self.package_root, "**", "conftest.py")
            conftests = [f for f in glob.glob(conftest_pattern, recursive=True)
                         if "__pycache__" not in self._norm(f)]
            has_conftest = len(conftests) > 0

            # --- pytest 配置文件 ---
            has_pytest_config = False
            for cfg in self._PYTEST_CONFIG_FILES:
                if os.path.isfile(os.path.join(self.project_root, cfg)):
                    has_pytest_config = True
                    break

            # --- 扫描测试文件内容 ---
            test_files = self._collect_test_files()
            uses_parametrize = False
            uses_mock = False
            fixture_count = 0

            for tf in test_files:
                try:
                    with open(tf, "r", encoding="utf-8") as fh:
                        content = fh.read()
                except (OSError, UnicodeDecodeError):
                    continue
                if "parametrize" in content:
                    uses_parametrize = True
                if re.search(r"\b(mock|patch|Mock|MagicMock)\b", content):
                    uses_mock = True
                fixture_count += len(
                    re.findall(r"@pytest\.fixture", content))

            # --- issues ---
            issues: List[Dict] = []
            if not has_conftest:
                issues.append({
                    "issue": "no_conftest",
                    "severity": "medium",
                    "suggestion": "添加 conftest.py 集中管理共享 fixture 与测试配置",
                })
            if not has_pytest_config:
                issues.append({
                    "issue": "no_pytest_config",
                    "severity": "medium",
                    "suggestion": "添加 pytest.ini 或在 pyproject.toml 中配置 [tool.pytest.ini_options]",
                })
            if not uses_parametrize:
                issues.append({
                    "issue": "no_parametrize",
                    "severity": "low",
                    "suggestion": "使用 @pytest.mark.parametrize 覆盖多组输入，减少重复用例",
                })
            if not uses_mock:
                issues.append({
                    "issue": "no_mock",
                    "severity": "low",
                    "suggestion": "对外部依赖（引擎/网络/文件系统）使用 unittest.mock 隔离",
                })
            if fixture_count == 0:
                issues.append({
                    "issue": "no_fixtures",
                    "severity": "medium",
                    "suggestion": "定义 @pytest.fixture 复用测试前置数据（如 OrbitState 构造）",
                })

            data: Dict[str, Any] = {
                "has_conftest": has_conftest,
                "has_pytest_config": has_pytest_config,
                "uses_parametrize": uses_parametrize,
                "uses_mock": uses_mock,
                "fixture_count": fixture_count,
                "test_file_count": len(test_files),
            }

            self.metrics.gauge("agent.test.fixture_count", fixture_count)
            self.log.info("analyze_test_quality_complete", data=data)

            return self._make_result(
                "analyze_test_quality", success=True,
                data=data, issues=issues)

    # ================================================================
    # 3. 测试计划生成
    # ================================================================

    def generate_test_plan(self,
                           target_modules: Optional[List[str]] = None
                           ) -> AgentResult:
        """为指定模块生成测试计划（不生成代码，只生成计划）。

        若 ``target_modules`` 为 None，自动选择覆盖率最低的 5 个关键模块。
        关键模块优先级: adapters > loop/engine > rag/router >
                       validation_tools > cli_tui。
        """
        with self._time_operation("generate_test_plan"):
            if target_modules is None:
                target_modules = self._select_low_coverage_modules(limit=5)

            plans: List[Dict] = []
            for module_path in target_modules:
                plan = self._build_module_plan(module_path)
                if plan is not None:
                    plans.append(plan)

            total_cases = sum(p["estimated_cases"] for p in plans)

            data: Dict[str, Any] = {
                "plans": plans,
                "total_modules": len(plans),
                "total_estimated_cases": total_cases,
            }

            self.metrics.gauge("agent.test.planned_modules", len(plans))
            self.log.info("generate_test_plan_complete", data={
                "modules": len(plans), "estimated_cases": total_cases})

            return self._make_result(
                "generate_test_plan", success=True, data=data)

    # ================================================================
    # 4. 运行测试
    # ================================================================

    def run_tests(self, pattern: Optional[str] = None,
                  verbose: bool = False) -> AgentResult:
        """运行 pytest 并收集结果。

        Args:
            pattern: ``-k`` 过滤表达式（None = 不过滤）。
            verbose: True 时使用 ``-v``，否则 ``-q``。
        Returns:
            data 含 passed / failed / errors / duration_s / output（最后 10 行）。
        """
        with self._time_operation("run_tests"):
            cmd = [self._python_exec(), "-m", "pytest",
                   "aerospace_agent/", "-x"]
            cmd.append("-v" if verbose else "-q")
            if pattern:
                cmd.extend(["-k", pattern])
            cmd.append("--tb=short")

            self.log.info("run_tests_start", data={"cmd": " ".join(cmd)})

            try:
                proc = run_command(
                    cmd, cwd=self.project_root, timeout=300)
                if proc.timeout:
                    return self._make_result(
                        "run_tests", success=False,
                        error="pytest execution timed out (>300s)",
                        data={"passed": 0, "failed": 0, "errors": 0,
                              "duration_s": 0.0, "output": proc.stderr})
                output = proc.stdout + proc.stderr
                return_code = proc.returncode
            except subprocess.TimeoutExpired:
                return self._make_result(
                    "run_tests", success=False,
                    error="pytest 执行超时（>300s）",
                    data={"passed": 0, "failed": 0, "errors": 0,
                          "duration_s": 0.0, "output": "TIMEOUT"})
            except FileNotFoundError as exc:
                return self._make_result(
                    "run_tests", success=False,
                    error=f"pytest 执行失败: {exc}",
                    data={"passed": 0, "failed": 0, "errors": 0,
                          "duration_s": 0.0, "output": str(exc)})

            passed, failed, errors, skipped, duration = \
                self._parse_pytest_summary(output)

            last_lines = "\n".join(output.strip().split("\n")[-10:])

            data: Dict[str, Any] = {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "duration_s": duration,
                "return_code": return_code,
                "output": last_lines,
            }

            self.metrics.gauge("agent.test.passed", passed)
            self.metrics.gauge("agent.test.failed", failed)
            self.log.info("run_tests_complete", data={
                "passed": passed, "failed": failed,
                "errors": errors, "duration_s": duration})

            issues: List[Dict] = []
            if failed > 0:
                issues.append({
                    "issue": "test_failures",
                    "severity": "high",
                    "count": failed,
                    "suggestion": "修复失败的测试用例",
                })
            if errors > 0:
                issues.append({
                    "issue": "test_errors",
                    "severity": "high",
                    "count": errors,
                    "suggestion": "排查测试收集/执行错误（导入失败、fixture 错误等）",
                })

            return self._make_result(
                "run_tests",
                success=(return_code == 0),
                data=data, issues=issues)

    # ================================================================
    # 5. 修复验证
    # ================================================================

    def verify_fix(self, changed_files: List[str]) -> AgentResult:
        """验证修复是否通过测试（用于 FixAgent 修复后的验证）。

        接收变更文件列表，根据文件路径推断测试文件并运行。
        回归 = 之前通过现在失败的测试 → ``severity="critical"``。
        """
        with self._time_operation("verify_fix"):
            test_files = self._infer_test_files(changed_files)

            if not test_files:
                data: Dict[str, Any] = {
                    "changed_files": changed_files,
                    "tests_run": 0, "passed": 0, "failed": 0,
                    "regressions": [],
                    "message": "未找到与变更文件对应的测试文件",
                }
                self.log.info("verify_fix_no_tests", data={
                    "changed_files": changed_files})
                return self._make_result(
                    "verify_fix", success=True, data=data)

            cmd = [self._python_exec(), "-m", "pytest", "-v",
                   "--tb=short"] + test_files
            self.log.info("verify_fix_start", data={
                "changed_files": changed_files,
                "test_files": test_files})

            try:
                proc = run_command(
                    cmd, cwd=self.project_root, timeout=300)
                if proc.timeout:
                    return self._make_result(
                        "verify_fix", success=False,
                        error="verification pytest execution timed out (>300s)",
                        data={"changed_files": changed_files,
                              "tests_run": 0, "passed": 0, "failed": 0,
                              "regressions": []})
                output = proc.stdout + proc.stderr
            except subprocess.TimeoutExpired:
                return self._make_result(
                    "verify_fix", success=False,
                    error="验证测试执行超时（>300s）",
                    data={"changed_files": changed_files,
                          "tests_run": 0, "passed": 0, "failed": 0,
                          "regressions": []})
            except FileNotFoundError as exc:
                return self._make_result(
                    "verify_fix", success=False,
                    error=f"pytest 执行失败: {exc}",
                    data={"changed_files": changed_files,
                          "tests_run": 0, "passed": 0, "failed": 0,
                          "regressions": []})

            current = self._parse_verbose_results(output)
            passed, failed, errors, _, _ = \
                self._parse_pytest_summary(output)

            # --- 回归检测 ---
            # 回归 = 基线中 "passed" 但当前非 "passed" 的测试
            regressions: List[Dict] = []
            if self._baseline:
                for test_id, status in current.items():
                    prev = self._baseline.get(test_id)
                    if prev == "passed" and status != "passed":
                        regressions.append({
                            "test": test_id,
                            "previous": prev,
                            "current": status,
                        })

            had_baseline = bool(self._baseline)
            # 更新基线（当前结果成为新基线）
            self._baseline = current

            # --- issues ---
            issues: List[Dict] = []
            for reg in regressions:
                issues.append({
                    "test": reg["test"],
                    "issue": "regression",
                    "severity": "critical",
                    "previous": reg["previous"],
                    "current": reg["current"],
                    "suggestion": "修复引入了回归，需回滚变更或修正副作用",
                })
            if failed > 0 and not regressions:
                issues.append({
                    "issue": "test_failures",
                    "severity": "high",
                    "count": failed,
                    "suggestion": "测试仍未通过，需继续修复",
                })

            data: Dict[str, Any] = {
                "changed_files": changed_files,
                "tests_run": len(current),
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "regressions": regressions,
                "had_baseline": had_baseline,
            }

            self.metrics.gauge("agent.test.regressions", len(regressions))
            self.log.info("verify_fix_complete", data={
                "tests_run": len(current), "passed": passed,
                "failed": failed, "regressions": len(regressions),
                "had_baseline": had_baseline})

            success = (failed == 0 and errors == 0
                       and len(regressions) == 0)
            return self._make_result(
                "verify_fix", success=success,
                data=data, issues=issues)

    # ================================================================
    # 6. CI 就绪度检查
    # ================================================================

    def check_ci_readiness(self) -> AgentResult:
        """检查 CI/CD 就绪度。

        检查: ``.github/workflows/`` / ``requirements.txt`` 声明 pytest /
        ``.gitignore`` 排除 ``.pytest_cache``。返回 issues 含修复建议。
        """
        with self._time_operation("check_ci_readiness"):
            # --- CI 配置 ---
            workflows_dir = os.path.join(
                self.project_root, ".github", "workflows")
            ci_files = []
            if os.path.isdir(workflows_dir):
                ci_files = (glob.glob(os.path.join(workflows_dir, "*.yml"))
                            + glob.glob(os.path.join(workflows_dir, "*.yaml")))
            has_ci = len(ci_files) > 0

            # --- 测试依赖 ---
            has_test_deps = self._check_pytest_in_requirements()

            # --- .gitignore 规则 ---
            has_gitignore_rules = False
            gitignore_path = os.path.join(self.project_root, ".gitignore")
            if os.path.isfile(gitignore_path):
                try:
                    with open(gitignore_path, "r", encoding="utf-8") as fh:
                        if ".pytest_cache" in fh.read():
                            has_gitignore_rules = True
                except OSError:
                    pass

            # --- issues ---
            issues: List[Dict] = []
            if not has_ci:
                issues.append({
                    "issue": "no_ci",
                    "severity": "high",
                    "suggestion": (
                        "创建 .github/workflows/ci.yml，配置 push/PR 触发的 "
                        "pytest 检查 job，确保每次提交都经过测试门禁"),
                })
            if not has_test_deps:
                issues.append({
                    "issue": "no_test_deps",
                    "severity": "medium",
                    "suggestion": (
                        "在 requirements.txt 或 requirements-dev.txt 中声明 "
                        "pytest（及 pytest-cov 等测试依赖）"),
                })
            if not has_gitignore_rules:
                issues.append({
                    "issue": "no_gitignore_pytest_cache",
                    "severity": "low",
                    "suggestion": "在 .gitignore 中添加 .pytest_cache/ 排除测试缓存目录",
                })

            data: Dict[str, Any] = {
                "has_ci": has_ci,
                "has_test_deps": has_test_deps,
                "has_gitignore_rules": has_gitignore_rules,
            }

            self.log.info("check_ci_readiness_complete", data=data)

            return self._make_result(
                "check_ci_readiness", success=True,
                data=data, issues=issues)

    # ================================================================
    # 内部辅助方法
    # ================================================================

    @staticmethod
    def _norm(path: str) -> str:
        """将路径归一化为正斜杠形式（跨平台比较）。"""
        return path.replace("\\", "/")

    @staticmethod
    def _python_exec() -> str:
        """获取 Python 可执行文件路径（确保用同一解释器运行 pytest）。"""
        return sys.executable or "python"

    # --- 文件收集 ---

    def _collect_source_files(self) -> List[str]:
        """收集 package 下所有源文件（排除指定目录与 tests/）。"""
        source_files: List[str] = []
        for root, dirs, files in os.walk(self.package_root):
            # 原地修改 dirs 以实现剪枝
            dirs[:] = [d for d in dirs if d not in self._EXCLUDE_DIRS]
            for f in files:
                if f.endswith(".py"):
                    source_files.append(os.path.join(root, f))
        return sorted(source_files)

    def _collect_test_basenames(self) -> set:
        """收集所有测试文件的基础名集合（用于覆盖率匹配）。"""
        basenames: set = set()
        patterns = [
            os.path.join(self.package_root, "**", "test_*.py"),
            os.path.join(self.project_root, "**", "test_*.py"),
        ]
        for pattern in patterns:
            for tf in glob.glob(pattern, recursive=True):
                norm = self._norm(tf)
                if "__pycache__" in norm or "/build/" in norm:
                    continue
                basenames.add(os.path.basename(tf))
        return basenames

    def _collect_test_files(self) -> List[str]:
        """收集所有测试文件的完整路径（用于内容扫描）。"""
        result: List[str] = []
        patterns = [
            os.path.join(self.package_root, "**", "test_*.py"),
            os.path.join(self.project_root, "**", "test_*.py"),
        ]
        for pattern in patterns:
            for tf in glob.glob(pattern, recursive=True):
                norm = self._norm(tf)
                if "__pycache__" in norm or "/build/" in norm:
                    continue
                result.append(tf)
        return sorted(set(result))

    # --- 模块分析 ---

    def _get_module_priority(self, filepath: str) -> Optional[int]:
        """获取模块的关键路径优先级（None = 非关键模块）。"""
        norm = self._norm(filepath)
        for pattern, priority in self._KEY_MODULE_PRIORITY:
            if pattern in norm:
                return priority
        return None

    def _select_low_coverage_modules(self, limit: int = 5) -> List[str]:
        """选择覆盖率最低的关键模块（按优先级排序，不足时补充其他未测试模块）。"""
        cov_result = self.scan_coverage()
        untested = cov_result.data.get("untested_files", [])

        # 关键模块按优先级排序
        key_modules: List[Tuple[int, str]] = []
        for uf in untested:
            priority = self._get_module_priority(uf)
            if priority is not None:
                key_modules.append((priority, uf))
        key_modules.sort(key=lambda x: x[0])

        selected = [m for _, m in key_modules[:limit]]

        # 不足时补充其他未测试模块
        if len(selected) < limit:
            selected_set = set(selected)
            remaining = [uf for uf in untested if uf not in selected_set]
            selected.extend(remaining[:limit - len(selected)])

        return selected

    def _build_module_plan(self, module_path: str) -> Optional[Dict]:
        """为单个模块构建测试计划。"""
        if not os.path.isfile(module_path):
            return None

        api = self._extract_public_api(module_path)
        functions = api["functions"]
        classes = api["classes"]
        priority = self._get_module_priority(module_path)
        test_priorities = self._generate_test_priorities(module_path)

        # 估算用例数: 每函数 3 个（正常/边界/错误）+ 每类 2 个 + 基础 2 个
        estimated = len(functions) * 3 + len(classes) * 2 + 2

        return {
            "module": module_path,
            "module_priority": priority,
            "functions": functions,
            "classes": classes,
            "test_priorities": test_priorities,
            "estimated_cases": estimated,
        }

    @staticmethod
    def _extract_public_api(filepath: str) -> Dict[str, List[str]]:
        """用 AST 提取模块的顶层公开函数和类（不含 _ 前缀）。"""
        functions: List[str] = []
        classes: List[str] = []
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=filepath)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    if not node.name.startswith("_"):
                        classes.append(node.name)
        except (OSError, SyntaxError):
            pass
        return {"functions": functions, "classes": classes}

    def _generate_test_priorities(self, filepath: str) -> List[str]:
        """根据模块类型生成测试要点。"""
        norm = self._norm(filepath)
        if "mcp/adapters" in norm:
            return [
                "适配器接口契约: 实现基类所有抽象方法",
                "引擎回退路径: 引擎未安装时降级而非崩溃",
                "输入边界: 非法坐标系/异常状态处理",
                "数值正确性: 已知场景的转换/传播结果验证",
            ]
        if "mcp/loop" in norm:
            return [
                "工作流执行: 正常路径端到端验证",
                "递归深度限制: 超限时优雅终止",
                "错误恢复: 子步骤失败的重试/回退",
                "状态一致性: 中断后可恢复",
            ]
        if "rag/router" in norm:
            return [
                "路由决策: 根据查询类型正确分发到子检索器",
                "降级策略: 主路由不可用时回退到默认路径",
                "性能: 路由延迟在可接受范围",
            ]
        if "validation" in norm:
            return [
                "合法输入: 标准格式通过验证",
                "非法输入: 缺失字段/类型错误返回明确错误",
                "边界值: 空值/极值/默认参数处理",
            ]
        if "cli_tui" in norm:
            return [
                "命令解析: 合法命令正确路由到处理函数",
                "错误提示: 非法输入的友好提示",
                "退出处理: 中断/异常时资源清理",
            ]
        return [
            "公开函数: 正常输入的行为验证",
            "错误路径: 异常输入的容错处理",
            "边界条件: 空值/极值/默认参数",
        ]

    # --- 测试推断 ---

    def _infer_test_files(self, changed_files: List[str]) -> List[str]:
        """根据变更文件路径推断测试文件（foo/bar.py → test_bar.py）。"""
        test_files: List[str] = []
        for cf in changed_files:
            stem = os.path.splitext(os.path.basename(cf))[0]
            expected = f"test_{stem}.py"
            pattern = os.path.join(self.package_root, "**", expected)
            matches = glob.glob(pattern, recursive=True)
            matches = [m for m in matches
                       if "__pycache__" not in self._norm(m)
                       and "/build/" not in self._norm(m)]
            if matches:
                test_files.extend(matches)
        return sorted(set(test_files))

    # --- 输出解析 ---

    @staticmethod
    def _parse_pytest_summary(output: str) -> Tuple[int, int, int, int, float]:
        """解析 pytest 摘要行。

        Returns:
            (passed, failed, errors, skipped, duration_s)
        """
        passed = failed = errors = skipped = 0
        duration = 0.0

        m = re.search(r"(\d+)\s+passed", output)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+)\s+failed", output)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+)\s+errors?", output)
        if m:
            errors = int(m.group(1))
        m = re.search(r"(\d+)\s+skipped", output)
        if m:
            skipped = int(m.group(1))
        m = re.search(r"in\s+([\d.]+)s", output)
        if m:
            duration = float(m.group(1))

        return passed, failed, errors, skipped, duration

    @staticmethod
    def _parse_verbose_results(output: str) -> Dict[str, str]:
        """解析 pytest -v 输出，返回 {test_id: status}。

        匹配形如 ``path::node_id PASSED [ 12%]`` 的行。
        """
        results: Dict[str, str] = {}
        for line in output.split("\n"):
            m = re.match(
                r"^(.+\.py)::(.+?)\s+(PASSED|FAILED|ERROR|SKIPPED)\b", line)
            if m:
                test_id = f"{m.group(1)}::{m.group(2)}"
                results[test_id] = m.group(3).lower()
        return results

    # --- 依赖检查 ---

    def _check_pytest_in_requirements(self) -> bool:
        """检查 requirements 文件中是否声明了 pytest。"""
        req_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "dev-requirements.txt",
        ]
        for req_name in req_files:
            req_path = os.path.join(self.project_root, req_name)
            if not os.path.isfile(req_path):
                continue
            try:
                with open(req_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        # 匹配 "pytest" 或 "pytest>=..." 等（忽略注释和空行）
                        stripped = line.strip()
                        if stripped.startswith("#") or not stripped:
                            continue
                        if re.match(r"pytest\b", stripped, re.IGNORECASE):
                            return True
            except OSError:
                continue
        return False
