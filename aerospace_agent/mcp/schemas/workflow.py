"""Workflow schemas — 工作流规格、结果、验证报告、Loop 账本。

第一性原理（K3）：Loop 是编排 ReAct 步骤的元工作流。
每个 WorkflowSpec 必须包含 goal/inputs/models/engine/steps/outputs/validation/failure_handling，
使 Loop 引擎能自动执行 Plan→Select→Retrieve→Generate→Run→Validate→Fix→Save。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LoopPhase(str, Enum):
    """Loop 八阶段。"""
    PLAN = "Plan"
    SELECT_ENGINE = "SelectEngine"
    RETRIEVE_DEMO = "RetrieveDemo"
    GENERATE_WORKFLOW = "GenerateWorkflow"
    RUN = "Run"
    VALIDATE = "Validate"
    FIX = "Fix"
    SAVE = "Save"


@dataclass
class WorkflowStep:
    """工作流单步。

    Attributes:
        name: 步骤名
        tool: 调用的 MCP tool 名
        inputs: 输入参数（Canonical Model 字典）
        outputs: 输出键名列表
        description: 步骤说明
    """
    name: str = ""
    tool: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "tool": self.tool,
            "inputs": self.inputs, "outputs": self.outputs,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowStep":
        return cls(
            name=d.get("name", ""), tool=d.get("tool", ""),
            inputs=d.get("inputs", {}), outputs=d.get("outputs", []),
            description=d.get("description", ""),
        )


@dataclass
class WorkflowSpec:
    """工作流规格（对应 workflow.yaml 的结构）。

    必须包含：goal, inputs, models, engine, steps, outputs, validation, failure_handling。
    """
    id: str = ""
    goal: str = ""
    task_type: str = ""           # orbit_propagation / ground_access / ...
    inputs: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)   # force_model / spacecraft 等
    engine: str = "auto"          # preferred engine
    steps: List[WorkflowStep] = field(default_factory=list)
    outputs: Dict[str, str] = field(default_factory=dict)  # 输出名 → 类型
    validation: Dict[str, Any] = field(default_factory=dict)
    failure_handling: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "goal": self.goal, "task_type": self.task_type,
            "inputs": self.inputs, "models": self.models,
            "engine": self.engine,
            "steps": [s.to_dict() for s in self.steps],
            "outputs": self.outputs,
            "validation": self.validation,
            "failure_handling": self.failure_handling,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowSpec":
        return cls(
            id=d.get("id", ""), goal=d.get("goal", ""),
            task_type=d.get("task_type", ""),
            inputs=d.get("inputs", {}), models=d.get("models", {}),
            engine=d.get("engine", "auto"),
            steps=[WorkflowStep.from_dict(s) for s in d.get("steps", [])],
            outputs=d.get("outputs", {}),
            validation=d.get("validation", {}),
            failure_handling=d.get("failure_handling", {}),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_yaml_dict(cls, d: dict) -> "WorkflowSpec":
        """从 YAML 解析后的字典构造（兼容 workflow.yaml 格式）。

        自动解析 ${inputs.x} / ${models.y} / ${steps.z.outputs.w} 变量引用。
        """
        resolved = cls._resolve_variables(d, context=d)
        return cls(
            id=resolved.get("id", resolved.get("name", "")),
            goal=resolved.get("goal", ""),
            task_type=resolved.get("task_type", ""),
            inputs=resolved.get("inputs", {}),
            models=resolved.get("models", {}),
            engine=resolved.get("engine", "auto"),
            steps=[WorkflowStep.from_dict(s) for s in resolved.get("steps", [])],
            outputs=resolved.get("outputs", {}),
            validation=resolved.get("validation", {}),
            failure_handling=resolved.get("failure_handling", {}),
            metadata=resolved.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # YAML 变量插值引擎
    # ------------------------------------------------------------------
    _VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

    @classmethod
    def _resolve_variables(cls, obj: Any, context: dict) -> Any:
        """递归解析 ${inputs.x} / ${models.y} / ${steps.z.outputs.w} 变量引用。

        Args:
            obj: 待解析的对象 (dict/list/str/其他)
            context: 顶层 YAML 字典，用于查找变量路径
        Returns:
            解析后的对象（变量引用被替换为实际值）
        """
        if isinstance(obj, dict):
            return {k: cls._resolve_variables(v, context) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._resolve_variables(item, context) for item in obj]
        if isinstance(obj, str):
            return cls._resolve_string(obj, context)
        return obj

    @classmethod
    def _resolve_string(cls, s: str, context: dict) -> Any:
        """解析字符串中的 ${...} 变量引用。

        如果整个字符串就是一个变量引用，返回原始类型值（dict/list/number）。
        如果字符串包含变量引用和其他文本，做字符串替换。
        """
        matches = cls._VAR_PATTERN.findall(s)
        if not matches:
            return s

        # 整个字符串就是单个变量引用 → 返回原始类型
        if len(matches) == 1 and s.strip() == f"${{{matches[0]}}}":
            return cls._lookup(matches[0].strip(), context)

        # 字符串中嵌入变量 → 字符串替换
        result = s
        for var_path in matches:
            val = cls._lookup(var_path.strip(), context)
            if val is not None:
                result = result.replace(f"${{{var_path}}}", str(val))
        return result

    @classmethod
    def _lookup(cls, path: str, context: dict) -> Any:
        """按点分路径查找变量值。

        支持: inputs.x, models.force_model, steps.传播轨道.outputs.final_state
        对于 steps.xxx.outputs.yyy，由于步骤在执行前无法解析，
        保留原始 ${...} 字符串（运行时由 LoopEngine 注入）。
        """
        parts = path.split(".")
        if not parts:
            return None

        # steps.xxx.outputs.yyy → 运行时解析，保留原样
        if parts[0] == "steps" and len(parts) >= 3 and parts[-2] == "outputs":
            return f"${{{path}}}"  # 保留，运行时注入

        # 其他路径从 context 中查找
        current = context
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current


@dataclass
class ValidationReport:
    """验证报告。

    Attributes:
        passed: 是否通过
        checks: 检查项列表 [{name, passed, detail, value, threshold}]
        position_error_m: 位置误差 m（交叉验证用）
        velocity_error_mps: 速度误差 m/s
        event_time_error_s: 事件时间误差 s
        confidence: 可信度评级（high/medium/low）
        notes: 备注
    """
    passed: bool = False
    checks: List[Dict[str, Any]] = field(default_factory=list)
    position_error_m: Optional[float] = None
    velocity_error_mps: Optional[float] = None
    event_time_error_s: Optional[float] = None
    confidence: str = "medium"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "position_error_m": self.position_error_m,
            "velocity_error_mps": self.velocity_error_mps,
            "event_time_error_s": self.event_time_error_s,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class WorkflowResult:
    """工作流执行结果。

    Attributes:
        workflow_id: 工作流 ID
        status: 状态（success/failed/partial）
        outputs: 输出数据
        state_history: 轨道状态历史（传播类工作流）
        validation: 验证报告
        engine: 实际使用的引擎
        engine_version: 引擎版本
        units: 输出单位说明
        frame: 输出坐标系
        time_scale: 时间尺度
        errors: 错误信息列表
        loop_ledger: Loop 账本（逐轮记录）
        elapsed_s: 执行耗时 s
    """
    workflow_id: str = ""
    status: str = "pending"       # success / failed / partial
    outputs: Dict[str, Any] = field(default_factory=dict)
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    validation: Optional[ValidationReport] = None
    engine: str = ""
    engine_version: str = ""
    units: str = "SI (m, m/s, s)"
    frame: str = "GCRF"
    time_scale: str = "UTC"
    errors: List[str] = field(default_factory=list)
    loop_ledger: List["LoopLedgerEntry"] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "outputs": self.outputs,
            "state_history": self.state_history,
            "validation": self.validation.to_dict() if self.validation else None,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "units": self.units,
            "frame": self.frame,
            "time_scale": self.time_scale,
            "errors": self.errors,
            "loop_ledger": [e.to_dict() for e in self.loop_ledger],
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class LoopLedgerEntry:
    """Loop 账本单条记录——每轮 Loop 必须记录。

    Attributes:
        phase: Loop 阶段
        goal: 当前目标
        tools_used: 使用的工具列表
        inputs: 输入参数
        outputs: 输出结果
        errors: 错误信息
        fix_action: 修复动作
        validation_result: 验证结果
        saved_as_reusable: 是否保存为可复用工作流
        timestamp: 时间戳
    """
    phase: LoopPhase = LoopPhase.PLAN
    goal: str = ""
    tools_used: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    fix_action: str = ""
    validation_result: Optional[str] = None
    saved_as_reusable: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if isinstance(self.phase, str):
            self.phase = LoopPhase(self.phase)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "goal": self.goal,
            "tools_used": self.tools_used,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "errors": self.errors,
            "fix_action": self.fix_action,
            "validation_result": self.validation_result,
            "saved_as_reusable": self.saved_as_reusable,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoopLedgerEntry":
        return cls(
            phase=LoopPhase(d.get("phase", "Plan")),
            goal=d.get("goal", ""),
            tools_used=d.get("tools_used", []),
            inputs=d.get("inputs", {}),
            outputs=d.get("outputs", {}),
            errors=d.get("errors", []),
            fix_action=d.get("fix_action", ""),
            validation_result=d.get("validation_result"),
            saved_as_reusable=d.get("saved_as_reusable", False),
            timestamp=d.get("timestamp", ""),
        )
