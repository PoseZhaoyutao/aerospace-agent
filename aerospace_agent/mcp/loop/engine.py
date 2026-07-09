"""LoopEngine — LoopRecursive-CEO Phase B 自主交付循环。

八阶段：Plan → Select Engine → Retrieve Demo → Generate Workflow → Run → Validate → Fix → Save

集成方式：
    engine = LoopEngine(llm=create_llm(), tools=TOOL_REGISTRY)
    result = engine.execute("设计地月转移轨道，精度<1km")

每轮 Loop 记录 LoopLedgerEntry，确保可追溯、可回放、可复用。
Loop 是元工作流，在 ReAct 之上编排——Run 阶段将 WorkflowSpec.steps
分发给 MCP tools 执行，Validate/Fix 驱动验证与最小修复。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..schemas.workflow import (
    WorkflowSpec, WorkflowResult, ValidationReport,
    LoopPhase, LoopLedgerEntry, WorkflowStep,
)
from .recursion import FirstPrinciplesAnalyzer


@dataclass
class LoopContext:
    """Loop 上下文——跨阶段共享的状态。"""
    goal: str = ""
    constraints: List[str] = field(default_factory=list)
    blueprint: Optional[dict] = None
    selected_engine: str = "auto"
    candidate_workflows: List[dict] = field(default_factory=list)
    workflow_spec: Optional[WorkflowSpec] = None
    workflow_result: Optional[WorkflowResult] = None
    validation_report: Optional[ValidationReport] = None
    fix_actions: List[str] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3
    ledger: List[LoopLedgerEntry] = field(default_factory=list)

    def add_ledger(self, entry: LoopLedgerEntry) -> None:
        entry.timestamp = datetime.now().isoformat()
        self.ledger.append(entry)


class LoopEngine:
    """Loop 引擎——编排八阶段自主交付循环。"""

    def __init__(self, llm=None, tools: Dict[str, Callable] = None,
                 max_iterations: int = 3):
        self.llm = llm
        self.tools = tools or {}
        self.max_iterations = max_iterations
        self.analyzer = FirstPrinciplesAnalyzer(llm=llm)

    def execute(self, goal: str, constraints: List[str] = None,
                context: Dict[str, Any] = None) -> WorkflowResult:
        """执行完整的 Loop 八阶段。"""
        # --- 观测性埋点 (懒加载) ---
        from ...utils.observability import get_logger, get_metrics
        log = get_logger("loop")
        metrics = get_metrics()
        log.info("loop_start", data={"goal": goal, "max_iterations": self.max_iterations})
        metrics.gauge("loop_max_iterations", self.max_iterations)

        constraints = constraints or []
        ctx = LoopContext(goal=goal, constraints=constraints,
                          max_iterations=self.max_iterations)
        print(f"\n{'='*60}\nLoop 引擎启动 | 目标: {goal}\n约束: {constraints}\n{'='*60}")

        # Phase A: 递归第一性原理分析
        ctx.blueprint = self._phase_plan(ctx, context or {})

        # Phase B: 八阶段循环
        for ctx.iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Loop 第 {ctx.iteration}/{ctx.max_iterations} 轮 ---")
            if ctx.iteration > 1:
                self._phase_fix(ctx)
            self._phase_select_engine(ctx)
            self._phase_retrieve_demo(ctx)
            self._phase_generate_workflow(ctx)
            self._phase_run(ctx)
            self._phase_validate(ctx)
            if ctx.validation_report and ctx.validation_report.passed:
                print(f"[验证通过] 可信度: {ctx.validation_report.confidence}")
                break

        self._phase_save(ctx)
        result = ctx.workflow_result or WorkflowResult()
        result.workflow_id = ctx.workflow_spec.id if ctx.workflow_spec else ""
        result.loop_ledger = list(ctx.ledger)
        result.engine = ctx.selected_engine
        if ctx.validation_report:
            result.validation = ctx.validation_report
        print(f"\n{'='*60}\nLoop 引擎完成 | 状态: {result.status}\n"
              f"迭代轮次: {ctx.iteration} | 账本条目: {len(ctx.ledger)}\n{'='*60}\n")
        log.info("loop_end", data={"status": result.status, "iterations": ctx.iteration, "ledger_entries": len(ctx.ledger)})
        metrics.inc("loop_completions", tags={"status": result.status})
        return result

    def _phase_plan(self, ctx: LoopContext, extra_context: Dict) -> dict:
        """Plan：递归第一性原理分析，产出 v1 蓝图。"""
        from ...utils.observability import get_logger
        get_logger("loop").info("phase_start", data={"phase": "plan"})
        print("[Plan] 递归第一性原理分析 ...")
        blueprint = self.analyzer.analyze(
            goal=ctx.goal, constraints=ctx.constraints, context=extra_context)
        ctx.add_ledger(LoopLedgerEntry(
            phase=LoopPhase.PLAN, goal=ctx.goal,
            tools_used=["FirstPrinciplesAnalyzer"],
            inputs={"goal": ctx.goal, "constraints": ctx.constraints},
            outputs={"blueprint_keys": list(blueprint.get("blueprint", {}).keys())},
            validation_result="blueprint_generated"))
        print(f"  -> 蓝图: {len(blueprint.get('top_k_nodes', []))} 个铰链决策")
        return blueprint

    def _phase_select_engine(self, ctx: LoopContext) -> None:
        """SelectEngine：根据任务类型与引擎能力匹配。"""
        from ...utils.observability import get_logger
        get_logger("loop").info("phase_start", data={"phase": "select_engine"})
        print("[SelectEngine] 引擎能力匹配 ...")
        check_fn = self.tools.get("check_engine_availability")
        engine_info = {}
        if check_fn:
            try:
                engine_info = check_fn() or {}
            except Exception as e:
                engine_info = {"error": str(e)}
        task_type = self._infer_task_type(ctx)
        ctx.selected_engine = self._match_engine(task_type, engine_info, ctx.constraints)
        ctx.add_ledger(LoopLedgerEntry(
            phase=LoopPhase.SELECT_ENGINE, goal=f"为 {task_type} 选择引擎",
            tools_used=["check_engine_availability"],
            inputs={"task_type": task_type},
            outputs={"selected_engine": ctx.selected_engine},
            validation_result=f"engine={ctx.selected_engine}"))
        print(f"  -> 选中引擎: {ctx.selected_engine}")

    def _infer_task_type(self, ctx: LoopContext) -> str:
        gl = ctx.goal.lower()
        if "转移" in ctx.goal or "transfer" in gl: return "maneuver"
        if "传播" in ctx.goal or "propagat" in gl: return "orbit_propagation"
        if "可见" in ctx.goal or "access" in gl or "地面站" in ctx.goal: return "ground_access"
        if "星历" in ctx.goal or "ephemeris" in gl: return "ephemeris"
        if "坐标系" in ctx.goal or "frame" in gl: return "frame_transform"
        if "姿态" in ctx.goal or "attitude" in gl: return "attitude_control"
        if "验证" in ctx.goal or "validate" in gl: return "validation"
        return "orbit_propagation"

    def _match_engine(self, task_type: str, engine_info: Dict,
                      constraints: List[str]) -> str:
        priority = {
            "orbit_propagation": ["orekit", "poliastro", "gmat", "basilisk", "builtin"],
            "maneuver": ["poliastro", "orekit", "gmat"],
            "ground_access": ["orekit", "stk", "builtin"],
            "ephemeris": ["spiceypy", "astropy"],
            "frame_transform": ["astropy", "spiceypy", "orekit"],
            "attitude_control": ["basilisk", "stk"],
            "validation": ["orekit", "gmat", "poliastro"],
        }
        for engine in priority.get(task_type, ["builtin"]):
            info = engine_info.get(engine, {}) if isinstance(engine_info, dict) else {}
            # K5-H6: 默认 False，仅当 info 明确报告 available=True 时才选中
            if isinstance(info, dict) and info.get("available", False):
                return engine
        return "auto"

    def _phase_retrieve_demo(self, ctx: LoopContext) -> None:
        """RetrieveDemo：搜索工作流目录 + 已保存的可复用 Loop 运行。"""
        from ...utils.observability import get_logger
        get_logger("loop").info("phase_start", data={"phase": "retrieve_demo"})
        print("[RetrieveDemo] 搜索工作流目录 + 可复用运行 ...")
        search_fn = self.tools.get("search_workflows")
        candidates = []
        if search_fn:
            try:
                result = search_fn(query=ctx.goal,
                                   task_type=self._infer_task_type(ctx),
                                   preferred_engine=ctx.selected_engine)
                candidates = result.get("candidates", []) if isinstance(result, dict) else []
            except Exception as e:
                print(f"  -> 搜索失败: {e}")
        # 搜索已保存的可复用 Loop 运行 (跨会话复用)
        try:
            reusable = self.search_reusable_runs(ctx.goal, top_k=3)
            if reusable:
                print(f"  -> 找到 {len(reusable)} 个可复用历史运行")
                candidates.extend(reusable)
        except Exception:
            pass
        ctx.candidate_workflows = candidates
        ctx.add_ledger(LoopLedgerEntry(
            phase=LoopPhase.RETRIEVE_DEMO, goal="检索可复用工作流",
            tools_used=["search_workflows"], inputs={"query": ctx.goal},
            outputs={"candidate_count": len(candidates)},
            validation_result=f"found_{len(candidates)}"))
        print(f"  -> 找到 {len(candidates)} 个候选")

    def _phase_generate_workflow(self, ctx: LoopContext) -> None:
        """GenerateWorkflow：生成 WorkflowSpec。"""
        from ...utils.observability import get_logger
        get_logger("loop").info("phase_start", data={"phase": "generate_workflow"})
        print("[GenerateWorkflow] 生成工作流规格 ...")
        gen_fn = self.tools.get("generate_astrodynamics_workflow")
        spec_dict = None
        if gen_fn:
            cid = (ctx.candidate_workflows[0].get("workflow_id", "")
                   if ctx.candidate_workflows else None)
            try:
                spec_dict = gen_fn(user_requirement=ctx.goal,
                                   candidate_workflow_id=cid,
                                   constraints=ctx.constraints)
            except Exception as e:
                print(f"  -> 生成失败: {e}")
        if spec_dict and isinstance(spec_dict, dict):
            ctx.workflow_spec = WorkflowSpec.from_dict(spec_dict)
        else:
            ctx.workflow_spec = self._fallback_workflow(ctx)
        ctx.add_ledger(LoopLedgerEntry(
            phase=LoopPhase.GENERATE_WORKFLOW, goal="生成 WorkflowSpec",
            tools_used=["generate_astrodynamics_workflow"],
            inputs={"requirement": ctx.goal},
            outputs={"workflow_id": ctx.workflow_spec.id,
                     "step_count": len(ctx.workflow_spec.steps)},
            validation_result=f"generated_{len(ctx.workflow_spec.steps)}_steps"))
        print(f"  -> 工作流 ID: {ctx.workflow_spec.id}, 步骤: {len(ctx.workflow_spec.steps)}")

    def _fallback_workflow(self, ctx: LoopContext) -> WorkflowSpec:
        """回退工作流——提供与工具签名匹配的默认输入。"""
        import math as _m
        tt = self._infer_task_type(ctx)
        # 默认 LEO 轨道状态（ISS-like，400km 圆轨道）
        _alt = 400_000.0
        _r = 6_378_137.0 + _alt
        _v = _m.sqrt(3.986004418e14 / _r)
        _orbit = {
            "epoch": {"value": "2026-01-01T00:00:00", "scale": "UTC", "format": "ISO"},
            "frame": {"name": "GCRF", "center": "Earth", "realization": "IERS2010"},
            "representation": "cartesian",
            "position_m": [_r, 0.0, 0.0],
            "velocity_mps": [0.0, _v, 0.0],
        }
        _force = {
            "central_body": "Earth", "gravity": "point_mass",
            "degree": 0, "order": 0,
            "drag": {"enabled": False}, "srp": {"enabled": False},
            "third_body": [], "relativity": False,
        }
        # 按任务类型构建与工具签名匹配的步骤
        if tt == "orbit_propagation":
            step = WorkflowStep(
                name="propagate", tool="propagate_orbit",
                inputs={
                    "initial_state_dict": _orbit,
                    "force_model_dict": _force,
                    "duration_s": 86400.0,
                    "output_step_s": 300.0,
                    "engine": ctx.selected_engine or "auto",
                },
                outputs=["state_history"],
                description="回退：二体轨道传播 1 天",
            )
        elif tt == "ground_access":
            step = WorkflowStep(
                name="compute_access", tool="compute_ground_access",
                inputs={
                    "orbit_state_dict": _orbit,
                    "ground_station_dict": {
                        "name": "Beijing_Station",
                        "latitude_deg": 40.0789, "longitude_deg": 116.5867,
                        "altitude_m": 50.0, "min_elevation_deg": 5.0,
                    },
                    "start_epoch_dict": {"value": "2026-01-01T00:00:00",
                                         "scale": "UTC", "format": "ISO"},
                    "stop_epoch_dict": {"value": "2026-01-01T12:00:00",
                                        "scale": "UTC", "format": "ISO"},
                    "min_elevation_deg": 5.0,
                },
                outputs=["access_windows"],
                description="回退：地面站可见性计算 12 小时",
            )
        elif tt == "frame_transform":
            step = WorkflowStep(
                name="transform", tool="transform_frame",
                inputs={"state_dict": _orbit, "target_frame": "ITRF"},
                outputs=["state"],
                description="回退：GCRF → ITRF 坐标系转换",
            )
        elif tt == "ephemeris":
            step = WorkflowStep(
                name="query", tool="query_ephemeris_state",
                inputs={
                    "target": "Moon", "observer": "Earth",
                    "epoch_dict": {"value": "2026-01-01T00:00:00",
                                   "scale": "UTC", "format": "ISO"},
                    "frame": "GCRF", "aberration_correction": "NONE",
                },
                outputs=["position_m", "velocity_mps"],
                description="回退：月球星历查询",
            )
        else:
            # 通用回退：轨道传播
            step = WorkflowStep(
                name="propagate", tool="propagate_orbit",
                inputs={
                    "initial_state_dict": _orbit,
                    "force_model_dict": _force,
                    "duration_s": 86400.0,
                    "engine": "auto",
                },
                outputs=["state_history"],
                description="回退：通用二体轨道传播",
            )
        return WorkflowSpec(
            id=f"fallback_{tt}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            goal=ctx.goal, task_type=tt, engine=ctx.selected_engine,
            steps=[step],
            outputs={"result": "dict"},
            validation={"min_checks": ["output_exists"]},
            failure_handling={"retry": 1, "fallback_engine": "builtin"})

    def _phase_run(self, ctx: LoopContext) -> None:
        """Run：执行工作流每个步骤。"""
        from ...utils.observability import get_logger
        get_logger("loop").info("phase_start", data={"phase": "run"})
        print("[Run] 执行工作流步骤 ...")
        if not ctx.workflow_spec:
            ctx.workflow_result = WorkflowResult(status="failed", errors=["无工作流规格"])
            return
        result = WorkflowResult(workflow_id=ctx.workflow_spec.id,
                                status="running", engine=ctx.selected_engine)
        errors = []
        for i, step in enumerate(ctx.workflow_spec.steps, 1):
            print(f"  步骤 {i}/{len(ctx.workflow_spec.steps)}: {step.name} -> {step.tool}")
            tool_fn = self.tools.get(step.tool)
            if tool_fn is None:
                errors.append(f"步骤 {step.name}: 未知工具 '{step.tool}'")
                continue
            # K5-C3: 运行时注入 ${steps.前步名.outputs.输出键} 变量引用
            resolved_inputs = self._inject_step_variables(step.inputs, result.outputs)
            try:
                sr = tool_fn(**resolved_inputs) if isinstance(resolved_inputs, dict) else tool_fn(resolved_inputs)
                # K5-M6: 检查工具返回错误状态，错误不计入有效 outputs
                if isinstance(sr, dict) and sr.get("status") in ("error", "unavailable", "failed"):
                    errors.append(f"步骤 {step.name}: 工具返回错误 - {sr.get('error', sr.get('message', ''))}")
                else:
                    result.outputs[step.name] = sr
                    if isinstance(sr, dict) and "state_history" in sr:
                        result.state_history = sr["state_history"]
            except Exception as e:
                errors.append(f"步骤 {step.name}: {e}")
        result.errors = errors
        result.status = "failed" if (errors and not result.outputs) else (
            "partial" if errors else "success")
        ctx.workflow_result = result
        ctx.add_ledger(LoopLedgerEntry(
            phase=LoopPhase.RUN, goal="执行工作流步骤",
            tools_used=[s.tool for s in ctx.workflow_spec.steps],
            inputs={"step_count": len(ctx.workflow_spec.steps)},
            outputs={"status": result.status, "output_keys": list(result.outputs.keys())},
            errors=errors, validation_result=result.status))
        print(f"  -> 执行状态: {result.status}")

    @staticmethod
    def _inject_step_variables(inputs: dict, outputs: dict) -> dict:
        """K5-C3: 运行时注入 ${steps.前步名.outputs.输出键} 变量引用。

        将 step.inputs 中形如 ${steps.xxx.outputs.yyy} 的字符串
        替换为 outputs["xxx"]["yyy"] 的实际值。
        """
        if not isinstance(inputs, dict):
            return inputs
        import re
        pattern = re.compile(r"\$\{steps\.([^}]+)\.outputs\.([^}]+)\}")
        resolved = {}
        for key, val in inputs.items():
            if isinstance(val, str):
                matches = pattern.findall(val)
                if matches:
                    for step_name, output_key in matches:
                        step_output = outputs.get(step_name, {})
                        if isinstance(step_output, dict):
                            actual = step_output.get(output_key)
                            if actual is not None:
                                # 整个字符串就是变量引用 → 返回原始类型
                                if val.strip() == f"${{steps.{step_name}.outputs.{output_key}}}":
                                    val = actual
                                    break
                                else:
                                    val = val.replace(
                                        f"${{steps.{step_name}.outputs.{output_key}}}",
                                        str(actual))
            resolved[key] = val
        return resolved

    def _phase_validate(self, ctx: LoopContext) -> None:
        """Validate：验证执行结果——双源交叉验证 + 阈值判定。"""
        from ...utils.observability import get_logger, get_metrics
        log = get_logger("loop")
        metrics = get_metrics()
        log.info("phase_start", data={"phase": "validate"})
        print("[Validate] 验证结果 ...")
        if not ctx.workflow_result:
            ctx.validation_report = ValidationReport(passed=False, confidence="low",
                                                      notes="无执行结果")
            if ctx.validation_report:
                metrics.gauge("validation_confidence",
                              {"high": 3, "medium": 2, "low": 1}.get(ctx.validation_report.confidence, 0))
                metrics.inc("validations", tags={"passed": str(ctx.validation_report.passed)})
            ctx.add_ledger(LoopLedgerEntry(phase=LoopPhase.VALIDATE, goal="验证",
                                           validation_result="FAIL_no_result"))
            return
        result = ctx.workflow_result
        checks = []
        checks.append({"name": "output_exists", "passed": bool(result.outputs),
                       "detail": f"输出键: {list(result.outputs.keys())}"})
        checks.append({"name": "no_errors", "passed": not result.errors,
                       "detail": f"错误数: {len(result.errors)}"})
        if result.state_history:
            checks.append({"name": "units_labeled", "passed": bool(result.units),
                           "detail": f"units={result.units}, frame={result.frame}"})
            checks.append({"name": "state_history_complete", "passed": len(result.state_history) > 0,
                           "detail": f"状态点数: {len(result.state_history)}"})

        # 交叉验证
        pos_err: Optional[float] = None
        vel_err: Optional[float] = None
        cross_fn = self.tools.get("cross_validate_results")
        if cross_fn and ctx.workflow_spec:
            try:
                # 提取阈值 —— 兼容两种 validation 结构
                spec_dict = ctx.workflow_spec.to_dict() if hasattr(ctx.workflow_spec, "to_dict") else {}
                validation_cfg = spec_dict.get("validation", {})
                # K5-H7: 先查 thresholds.position_error_m（生成工作流格式），
                #        再查 position_error.threshold_m（YAML 格式）
                thresholds_cfg = validation_cfg.get("thresholds", {})
                pos_threshold = (thresholds_cfg.get("position_error_m")
                                 or validation_cfg.get("position_error", {}).get("threshold_m", 1000.0))
                vel_threshold = (thresholds_cfg.get("velocity_error_mps")
                                 or validation_cfg.get("velocity_error", {}).get("threshold_mps", 0.1))
                thresholds = {
                    "position_error_m": pos_threshold,
                    "velocity_error_mps": vel_threshold,
                }
                cv = cross_fn(task_spec=spec_dict, thresholds=thresholds)
                if isinstance(cv, dict):
                    pos_err = cv.get("position_error_m")
                    vel_err = cv.get("velocity_error_mps")
                    cv_passed = cv.get("passed", False)
                    cv_conf = cv.get("confidence", "low")
                    pos_threshold = thresholds.get("position_error_m", 1000.0)
                    vel_threshold = thresholds.get("velocity_error_mps", 0.1)
                    checks.append({"name": "cross_validate",
                                   "passed": cv_passed,
                                   "detail": (f"pos_err={pos_err} m (阈值 {pos_threshold} m), "
                                              f"vel_err={vel_err} m/s (阈值 {vel_threshold} m/s), "
                                              f"confidence={cv_conf}"),
                                   "value": pos_err, "threshold": pos_threshold})
                    # 额外检查项：RMS 误差
                    rms_pos = cv.get("rms_position_error_m")
                    if rms_pos is not None:
                        checks.append({"name": "rms_position_error",
                                       "passed": rms_pos < pos_threshold,
                                       "detail": f"RMS 位置误差: {rms_pos:.3f} m",
                                       "value": rms_pos, "threshold": pos_threshold})
            except Exception as exc:
                checks.append({"name": "cross_validate", "passed": False,
                               "detail": f"交叉验证异常: {exc}"})

        passed = all(c["passed"] for c in checks)
        conf = "high" if passed and len(checks) >= 3 else ("medium" if passed else "low")
        ctx.validation_report = ValidationReport(
            passed=passed, checks=checks, confidence=conf,
            position_error_m=pos_err,
            velocity_error_mps=vel_err,
            notes=f"通过 {sum(1 for c in checks if c['passed'])}/{len(checks)} 项")
        if ctx.validation_report:
            metrics.gauge("validation_confidence",
                          {"high": 3, "medium": 2, "low": 1}.get(ctx.validation_report.confidence, 0))
            metrics.inc("validations", tags={"passed": str(ctx.validation_report.passed)})
        ctx.add_ledger(LoopLedgerEntry(
            phase=LoopPhase.VALIDATE, goal="验证结果正确性",
            tools_used=["cross_validate_results"] if cross_fn else [],
            inputs={"check_count": len(checks)},
            outputs={"passed": passed, "confidence": conf,
                     "position_error_m": pos_err, "velocity_error_mps": vel_err},
            validation_result=f"{'PASS' if passed else 'FAIL'}_{conf}"))
        print(f"  -> {'通过' if passed else '未通过'} (可信度: {conf})"
              + (f" | 位置误差: {pos_err:.3f} m" if pos_err is not None else "")
              + (f" | 速度误差: {vel_err:.6f} m/s" if vel_err is not None else ""))

    def _phase_fix(self, ctx: LoopContext) -> None:
        """Fix：最小修复——分析失败检查项并调整。"""
        from ...utils.observability import get_logger
        get_logger("loop").info("phase_start", data={"phase": "fix"})
        print("[Fix] 最小修复 ...")
        if not ctx.validation_report or ctx.validation_report.passed:
            return
        failed_checks = [c for c in ctx.validation_report.checks if not c["passed"]]
        fixes = []
        for fc in failed_checks:
            name = fc.get("name", "")
            if name == "no_errors":
                fixes.append(f"重试失败步骤，切换到 fallback_engine")
                ctx.selected_engine = ctx.workflow_spec.failure_handling.get(
                    "fallback_engine", "builtin") if ctx.workflow_spec else "builtin"
            elif name == "output_exists":
                fixes.append(f"使用回退工作流重新生成")
                ctx.workflow_spec = self._fallback_workflow(ctx)
            elif name == "cross_validate":
                fixes.append(f"放宽验证阈值或增加引擎对比")
            else:
                fixes.append(f"检查 {name}: {fc.get('detail','')}")
        ctx.fix_actions.extend(fixes)
        ctx.add_ledger(LoopLedgerEntry(
            phase=LoopPhase.FIX, goal="修复失败检查项",
            tools_used=[], inputs={"failed_checks": [c["name"] for c in failed_checks]},
            outputs={"fix_actions": fixes},
            validation_result=f"applied_{len(fixes)}_fixes"))
        print(f"  -> 应用 {len(fixes)} 个修复")

    def _phase_save(self, ctx: LoopContext) -> None:
        """Save：沉淀可复用工作流 + 持久化 LoopLedger 到磁盘。"""
        from ...utils.observability import get_logger
        get_logger("loop").info("phase_start", data={"phase": "save"})
        print("[Save] 沉淀可复用工作流 ...")
        saved = False
        save_path = ""
        if (ctx.workflow_result and ctx.workflow_result.status == "success"
                and ctx.workflow_spec):
            saved = True
        # 持久化 LoopLedger (无论成功与否,都保存以供回放/审计)
        try:
            save_path = self.save_ledger(ctx)
        except Exception as e:
            print(f"  -> 账本持久化失败: {e}")
        ctx.add_ledger(LoopLedgerEntry(
            phase=LoopPhase.SAVE, goal="沉淀可复用工作流",
            tools_used=[], inputs={"status": ctx.workflow_result.status if ctx.workflow_result else "none"},
            outputs={"saved_as_reusable": saved, "ledger_path": save_path},
            validation_result=f"saved={saved}",
            saved_as_reusable=saved))
        print(f"  -> {'已保存为可复用' if saved else '未保存（执行未成功）'}")
        if save_path:
            print(f"  -> 账本已写入: {save_path}")

    # ------------------------------------------------------------------
    # LoopLedger 持久化 (写盘支持回放/复用/审计)
    # ------------------------------------------------------------------
    DEFAULT_RUNS_DIR = "data/loop_runs"

    def save_ledger(self, ctx: LoopContext, runs_dir: str = None) -> str:
        """将完整 LoopLedger + 蓝图 + 工作流规格 + 结果写入 JSON 文件。

        Args:
            ctx: LoopContext (含全程 ledger)
            runs_dir: 输出目录 (默认 data/loop_runs)

        Returns:
            保存的文件路径
        """
        runs_dir = runs_dir or self.DEFAULT_RUNS_DIR
        Path(runs_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 用 goal 前 20 字符做文件名,确保合法
        goal_slug = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in ctx.goal[:20]
        ).strip("_") or "task"
        filename = f"{timestamp}_{goal_slug}.json"
        filepath = os.path.join(runs_dir, filename)

        # 序列化 ledger
        ledger_data = []
        for entry in ctx.ledger:
            ledger_data.append({
                "phase": entry.phase.value if hasattr(entry.phase, "value") else str(entry.phase),
                "goal": entry.goal,
                "tools_used": entry.tools_used,
                "inputs": entry.inputs,
                "outputs": entry.outputs,
                "errors": entry.errors,
                "fix_action": entry.fix_action,
                "validation_result": entry.validation_result,
                "saved_as_reusable": entry.saved_as_reusable,
                "timestamp": entry.timestamp,
            })

        # 序列化 blueprint
        blueprint_data = None
        if ctx.blueprint:
            bp = ctx.blueprint.get("blueprint", ctx.blueprint)
            blueprint_data = {
                "architecture": bp.get("architecture"),
                "data_model": bp.get("data_model"),
                "workflow_shape": bp.get("workflow_shape"),
                "key_principles": bp.get("key_principles", []),
                "risk_mitigations": bp.get("risk_mitigations", []),
            }

        # 序列化 workflow_spec
        spec_data = None
        if ctx.workflow_spec:
            spec_data = ctx.workflow_spec.to_dict()

        # 序列化 result
        result_data = None
        if ctx.workflow_result:
            result_data = ctx.workflow_result.to_dict()

        run_record = {
            "run_id": timestamp,
            "timestamp": timestamp,
            "goal": ctx.goal,
            "constraints": ctx.constraints,
            "selected_engine": ctx.selected_engine,
            "iterations": ctx.iteration,
            "max_iterations": ctx.max_iterations,
            "blueprint": blueprint_data,
            "workflow_spec": spec_data,
            "result": result_data,
            "ledger": ledger_data,
            "ledger_count": len(ledger_data),
            "saved_as_reusable": (ctx.workflow_result is not None
                                  and ctx.workflow_result.status == "success"),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(run_record, f, ensure_ascii=False, indent=2, default=str)
        return filepath

    @classmethod
    def load_ledger(cls, filepath: str) -> dict:
        """从 JSON 文件加载已保存的 Loop 运行记录。

        Args:
            filepath: save_ledger 保存的文件路径

        Returns:
            包含 goal/constraints/blueprint/workflow_spec/result/ledger 的字典
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def list_saved_runs(cls, runs_dir: str = None) -> List[dict]:
        """列出已保存的 Loop 运行记录 (按时间倒序)。

        Args:
            runs_dir: 运行记录目录 (默认 data/loop_runs)

        Returns:
            运行记录摘要列表,每项含 run_id/timestamp/goal/saved_as_reusable/ledger_count
        """
        runs_dir = runs_dir or cls.DEFAULT_RUNS_DIR
        if not os.path.isdir(runs_dir):
            return []
        runs = []
        for fname in sorted(os.listdir(runs_dir), reverse=True):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(runs_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    record = json.load(f)
                runs.append({
                    "run_id": record.get("run_id", fname),
                    "timestamp": record.get("timestamp", ""),
                    "goal": record.get("goal", "")[:80],
                    "saved_as_reusable": record.get("saved_as_reusable", False),
                    "ledger_count": record.get("ledger_count", 0),
                    "filepath": fpath,
                })
            except (json.JSONDecodeError, IOError):
                continue
        return runs

    @classmethod
    def search_reusable_runs(cls, query: str, runs_dir: str = None,
                             top_k: int = 5) -> List[dict]:
        """搜索可复用的 Loop 运行记录 (供 RetrieveDemo 阶段使用)。

        Args:
            query: 搜索查询 (如任务目标)
            runs_dir: 运行记录目录
            top_k: 返回前 N 条

        Returns:
            匹配的可复用运行记录列表
        """
        all_runs = cls.list_saved_runs(runs_dir)
        reusable = [r for r in all_runs if r.get("saved_as_reusable")]
        # 简单关键词匹配
        query_lower = query.lower()
        scored = []
        for run in reusable:
            goal_lower = run.get("goal", "").lower()
            # 计算关键词重合度
            overlap = sum(1 for w in query_lower.split()
                         if len(w) > 1 and w in goal_lower)
            scored.append((overlap, run))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]
