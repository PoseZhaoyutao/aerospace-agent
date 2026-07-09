"""MultiAgentOrchestrator — 三Agent协作编排器。

第一性原理（K5 专业化分工）：
  1. 复杂任务需要专业化分工——测试、修复、架构治理各有不同目标
  2. Agent 间通过结构化 AgentResult 通信，不共享可变状态
  3. 编排器是唯一的协调者，负责任务分发、结果聚合、循环控制
  4. 协作循环：Arch审计 → Fix修复 → Test验证 → Arch复审（直到收敛）

协作模式：
    ┌──────────────────────────────────────────────┐
    │           MultiAgentOrchestrator             │
    │                                              │
    │  1. ArchAgent.enforce_architecture_redlines  │
    │          ↓ issues                            │
    │  2. FixAgent.analyze_issues + apply_fixes    │
    │          ↓ changed_files                     │
    │  3. TestAgent.verify_fix                     │
    │          ↓ pass/fail                         │
    │  4. if fail → FixAgent re-fix                │
    │     if pass → ArchAgent re-audit             │
    │     if converged → done                      │
    └──────────────────────────────────────────────┘
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.observability import get_logger, get_metrics
from .base import AgentBase, AgentResult, AgentRole
from .arch_agent import ArchAgent
from .fix_agent import FixAgent
from .test_agent import TestAgent


@dataclass
class OrchestrationReport:
    """编排报告——一次完整协作循环的输出。"""
    cycle: int                          # 循环序号
    arch_result: Optional[AgentResult] = None  # 架构审计结果
    fix_result: Optional[AgentResult] = None   # 修复结果
    test_result: Optional[AgentResult] = None  # 测试验证结果
    converged: bool = False             # 是否收敛
    total_issues_found: int = 0         # 总发现问题数
    total_issues_fixed: int = 0         # 总修复问题数
    total_regressions: int = 0          # 回归数
    duration_s: float = 0.0             # 总耗时

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "converged": self.converged,
            "total_issues_found": self.total_issues_found,
            "total_issues_fixed": self.total_issues_fixed,
            "total_regressions": self.total_regressions,
            "duration_s": round(self.duration_s, 3),
            "arch_result": self.arch_result.to_dict() if self.arch_result else None,
            "fix_result": self.fix_result.to_dict() if self.fix_result else None,
            "test_result": self.test_result.to_dict() if self.test_result else None,
        }


class MultiAgentOrchestrator:
    """三Agent协作编排器——ArchAgent + FixAgent + TestAgent。

    用法::

        orchestrator = MultiAgentOrchestrator(project_root="d:/Project/aerospace-agent")
        report = orchestrator.run_cycle(max_iterations=3)
        print(f"收敛: {report.converged}, 修复: {report.total_issues_fixed}")
    """

    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.log = get_logger("agent.orchestrator")
        self.metrics = get_metrics()

        # 初始化三个Agent
        self.arch_agent = ArchAgent(project_root=project_root)
        self.fix_agent = FixAgent(project_root=project_root)
        self.test_agent = TestAgent(project_root=project_root)

        # 编排状态
        self.cycle_count = 0
        self.history: List[OrchestrationReport] = []

    def run_cycle(self, max_iterations: int = 3,
                  auto_fix: bool = True,
                  skip_test: bool = False) -> OrchestrationReport:
        """执行一轮完整的 Arch→Fix→Test 协作循环。

        Args:
            max_iterations: 最大内部修复-验证迭代次数
            auto_fix: 是否自动执行修复（False=只分析不修复）
            skip_test: 是否跳过测试验证（快速模式）
        Returns:
            OrchestrationReport
        """
        self.cycle_count += 1
        cycle_num = self.cycle_count
        start_time = time.perf_counter()
        self.log.info("orchestration_cycle_start",
                      data={"cycle": cycle_num, "max_iterations": max_iterations})

        report = OrchestrationReport(cycle=cycle_num)

        # === 阶段 1: ArchAgent 架构审计 ===
        self.log.info("phase_1_arch_audit_start")
        arch_result = self.arch_agent.enforce_architecture_redlines()
        report.arch_result = arch_result
        report.total_issues_found = len(arch_result.issues)
        self.log.info("phase_1_arch_audit_done",
                      data={"issues": len(arch_result.issues),
                            "critical": sum(1 for i in arch_result.issues
                                           if i.get("severity") == "critical")})
        self.metrics.gauge("orchestration.issues_found", len(arch_result.issues))

        if not arch_result.issues or not auto_fix:
            report.converged = True
            report.duration_s = time.perf_counter() - start_time
            self.log.info("orchestration_cycle_end",
                          data={"cycle": cycle_num, "converged": True,
                                "reason": "no_issues_or_no_auto_fix"})
            self.history.append(report)
            return report

        # === 阶段 2: FixAgent 分析+修复 → TestAgent 验证 (迭代) ===
        all_fixed = 0
        all_regressions = 0
        changed_files: List[str] = []

        for iteration in range(1, max_iterations + 1):
            self.log.info("phase_2_fix_iter_start",
                          data={"iteration": iteration,
                                "issues_to_fix": len(arch_result.issues)})

            # 2a: FixAgent 分析问题
            analyze_result = self.fix_agent.analyze_issues(arch_result.issues)
            fix_plan = analyze_result.data.get("plan", [])

            if not fix_plan:
                self.log.info("phase_2_no_fix_plan")
                break

            # 2b: FixAgent 执行修复
            fix_result = self.fix_agent.apply_fixes(fix_plan)
            report.fix_result = fix_result
            fixed_this_iter = fix_result.data.get("successful", 0)
            all_fixed += fixed_this_iter
            self.log.info("phase_2_fix_iter_done",
                          data={"fixed": fixed_this_iter,
                                "failed": fix_result.data.get("failed", 0)})

            # 收集变更文件
            for r in fix_result.data.get("results", []):
                if r.get("data", {}).get("file"):
                    changed_files.append(r["data"]["file"])

            if skip_test:
                break

            # 2c: TestAgent 验证修复
            test_result = self.test_agent.verify_fix(changed_files)
            report.test_result = test_result
            regressions = len(test_result.data.get("regressions", []))
            all_regressions += regressions
            self.log.info("phase_2_test_done",
                          data={"passed": test_result.data.get("passed", 0),
                                "failed": test_result.data.get("failed", 0),
                                "regressions": regressions})

            # 如果无回归且测试通过，跳出迭代
            if regressions == 0 and test_result.data.get("failed", 0) == 0:
                self.log.info("phase_2_converged",
                              data={"iteration": iteration})
                break

            # 如果有回归，将回归问题加入下一轮 issues
            if regressions > 0:
                for reg in test_result.data.get("regressions", []):
                    arch_result.issues.append({
                        "id": f"regression_{iteration}",
                        "file": reg.get("test_file", ""),
                        "issue": f"回归: {reg.get('reason', '')}",
                        "severity": "critical",
                    })

        report.total_issues_fixed = all_fixed
        report.total_regressions = all_regressions

        # === 阶段 3: ArchAgent 复审 ===
        if auto_fix and all_fixed > 0:
            self.log.info("phase_3_arch_reaudit_start")
            reaudit = self.arch_agent.enforce_architecture_redlines()
            remaining = len(reaudit.issues)
            original = len(arch_result.issues)
            self.log.info("phase_3_arch_reaudit_done",
                          data={"original": original, "remaining": remaining})

            # 收敛判定：剩余问题 ≤ 原始问题的 20% 或无 critical
            critical_remaining = sum(1 for i in reaudit.issues
                                    if i.get("severity") == "critical")
            report.converged = (critical_remaining == 0 and
                                remaining <= max(1, original * 0.2))
        else:
            report.converged = not auto_fix

        report.duration_s = time.perf_counter() - start_time
        self.log.info("orchestration_cycle_end",
                      data={"cycle": cycle_num, "converged": report.converged,
                            "issues_found": report.total_issues_found,
                            "issues_fixed": report.total_issues_fixed,
                            "regressions": report.total_regressions,
                            "duration_s": round(report.duration_s, 3)})

        self.metrics.inc("orchestration.cycles",
                         tags={"converged": str(report.converged)})
        self.metrics.gauge("orchestration.issues_fixed", all_fixed)

        self.history.append(report)
        return report

    def run_full_audit(self) -> Dict[str, Any]:
        """运行完整审计——只检测不修复，生成全景报告。

        调用所有三个 Agent 的所有检测方法，汇总为一份完整报告。
        """
        self.log.info("full_audit_start")
        start = time.perf_counter()

        # ArchAgent 全套检测
        dag = self.arch_agent.check_dag()
        imports = self.arch_agent.audit_imports()
        god_class = self.arch_agent.audit_god_class()
        config = self.arch_agent.unify_config()
        dead_code = self.arch_agent.detect_dead_code()
        versions = self.arch_agent.review_version_consistency()

        # TestAgent 全套检测
        coverage = self.test_agent.scan_coverage()
        quality = self.test_agent.analyze_test_quality()
        ci = self.test_agent.check_ci_readiness()

        # FixAgent 分析（不执行修复）
        all_issues = []
        for result in [dag, imports, god_class, dead_code, versions, coverage, ci]:
            all_issues.extend(result.issues)
        fix_plan = self.fix_agent.analyze_issues(all_issues)

        duration = time.perf_counter() - start
        self.log.info("full_audit_done", data={"duration_s": round(duration, 3)})

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_s": round(duration, 3),
            "arch_agent": {
                "dag": dag.to_dict(),
                "imports": imports.to_dict(),
                "god_class": god_class.to_dict(),
                "config": config.to_dict(),
                "dead_code": dead_code.to_dict(),
                "versions": versions.to_dict(),
            },
            "test_agent": {
                "coverage": coverage.to_dict(),
                "quality": quality.to_dict(),
                "ci_readiness": ci.to_dict(),
            },
            "fix_agent": {
                "fix_plan": fix_plan.to_dict(),
            },
            "summary": {
                "total_issues": len(all_issues),
                "critical": sum(1 for i in all_issues
                               if i.get("severity") == "critical"),
                "high": sum(1 for i in all_issues
                           if i.get("severity") == "high"),
                "medium": sum(1 for i in all_issues
                             if i.get("severity") == "medium"),
                "low": sum(1 for i in all_issues
                          if i.get("severity") == "low"),
            },
        }

    def get_history(self) -> List[Dict]:
        """获取编排历史。"""
        return [r.to_dict() for r in self.history]
