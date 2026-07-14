"""Schema I/O 协议 — Pydantic 模型定义。

第一性原理 (K2): Pydantic 模型为单一真相源。
所有节点输入输出严格类型校验，违反即终止。
支持 JSON Schema 导出供外部系统消费。

覆盖的航天领域 Schema:
    - 轨道状态 (OrbitState)
    - 轨道设计请求/响应 (OrbitDesignRequest/Response)
    - 发射窗口请求/响应 (LaunchWindowRequest/Response)
    - 月球转移请求/响应 (LunarTransferRequest/Response)
    - 工具调用请求/响应 (ToolCallRequest/Response)
    - RAG 查询请求/响应 (RagQueryRequest/Response)
    - Agent 输入/输出 (AgentInput/AgentOutput)
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Literal, Optional, Union
import re
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# 枚举类型
# ============================================================

class FrameType(str, Enum):
    """参考系类型。"""
    ECI = "eci"
    ECEF = "ecef"
    GCRF = "gcrf"
    TEME = "teme"
    LVLH = "lvlh"
    J2000 = "j2000"


class TimeScale(str, Enum):
    """时间尺度。"""
    UTC = "utc"
    TAI = "tai"
    TT = "tt"
    TDB = "tdb"
    GPS = "gps"


class ForceModel(str, Enum):
    """力模型类型。"""
    TWO_BODY = "two_body"
    J2 = "j2"
    J4 = "j4"
    FULL_GRAVITY = "full_gravity"
    DRAG = "drag"
    SRP = "srp"
    THIRD_BODY = "third_body"


class PropagatorType(str, Enum):
    """传播器类型。"""
    KEPLER = "kepler"
    RK4 = "rk4"
    RK45 = "rk45"
    DOP853 = "dop853"
    ENCKE = "encke"


class RunStatus(str, Enum):
    """Stable terminal/runtime statuses exposed by the agent protocol."""

    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    CYCLE_DETECTED = "cycle_detected"
    LIMIT_REACHED = "limit_reached"


class ActionType(str, Enum):
    """Actions a planner may request from the graph."""

    RETRIEVE = "retrieve"
    CALL_TOOL = "call_tool"
    RESPOND = "respond"
    STOP = "stop"


class IntentType(str, Enum):
    """航天意图分类。"""
    ORBIT_DESIGN = "orbit_design"
    ORBIT_PROPAGATION = "orbit_propagation"
    LAUNCH_WINDOW = "launch_window"
    LUNAR_TRANSFER = "lunar_transfer"
    MANEUVER_PLANNING = "maneuver_planning"
    KNOWLEDGE_QUERY = "knowledge_query"
    TOOL_DISCOVERY = "tool_discovery"
    GENERAL = "general"


# ============================================================
# 轨道状态 Schema
# ============================================================

class OrbitState(BaseModel):
    """轨道状态 — 笛卡尔坐标表示。

    单位: 位置 [m], 速度 [m/s], 历元 [MJD]
    """
    x: float = Field(..., description="位置 X [m]")
    y: float = Field(..., description="位置 Y [m]")
    z: float = Field(..., description="位置 Z [m]")
    vx: float = Field(..., description="速度 X [m/s]")
    vy: float = Field(..., description="速度 Y [m/s]")
    vz: float = Field(..., description="速度 Z [m/s]")
    epoch_mjd: float = Field(..., description="历元 [MJD]")
    frame: FrameType = Field(default=FrameType.GCRF, description="参考系")
    time_scale: TimeScale = Field(default=TimeScale.UTC, description="时间尺度")

    @field_validator("epoch_mjd")
    @classmethod
    def epoch_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"epoch_mjd 必须为正数，当前值: {v}")
        return v


class KeplerianOrbitState(BaseModel):
    """轨道状态 — 开普勒根数表示。

    单位: 半长轴 [m], 偏心率 [无量纲], 倾角 [rad],
          RAAN [rad], 近地点幅角 [rad], 真近点角 [rad]
    """
    semi_major_axis: float = Field(..., gt=0, description="半长轴 [m]")
    eccentricity: float = Field(..., ge=0, lt=1, description="偏心率 [0, 1)")
    inclination: float = Field(..., ge=0, le=3.1416, description="倾角 [rad]")
    raan: float = Field(..., ge=0, le=6.2832, description="升交点赤经 [rad]")
    arg_periapsis: float = Field(..., ge=0, le=6.2832, description="近地点幅角 [rad]")
    true_anomaly: float = Field(..., ge=0, le=6.2832, description="真近点角 [rad]")
    epoch_mjd: float = Field(..., gt=0, description="历元 [MJD]")
    frame: FrameType = Field(default=FrameType.GCRF)
    time_scale: TimeScale = Field(default=TimeScale.UTC)


# ============================================================
# 轨道设计请求/响应
# ============================================================

class OrbitDesignRequest(BaseModel):
    """轨道设计请求。"""
    orbit_type: str = Field(..., description="轨道类型: LEO/MEO/GEO/HEO/SSO")
    altitude_km: Optional[float] = Field(default=None, description="目标高度 [km]")
    inclination_deg: Optional[float] = Field(default=None, description="目标倾角 [度]")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="额外约束")
    force_models: List[ForceModel] = Field(default=[ForceModel.TWO_BODY])
    propagator: PropagatorType = Field(default=PropagatorType.RK45)


class OrbitDesignResponse(BaseModel):
    """轨道设计响应。"""
    status: str = Field(..., description="执行状态: success / error")
    initial_state: Optional[OrbitState] = Field(default=None)
    keplerian_elements: Optional[KeplerianOrbitState] = Field(default=None)
    propagation_result: Optional[Dict[str, Any]] = Field(default=None)
    validation_checks: Dict[str, bool] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ============================================================
# 发射窗口请求/响应
# ============================================================

class LaunchWindowRequest(BaseModel):
    """发射窗口分析请求。"""
    target_orbit: OrbitState = Field(..., description="目标轨道状态")
    launch_site_lat_deg: float = Field(..., ge=-90, le=90, description="发射场纬度 [度]")
    launch_site_lon_deg: float = Field(..., ge=-180, le=180, description="发射场经度 [度]")
    start_time_mjd: float = Field(..., gt=0, description="搜索起始时间 [MJD]")
    end_time_mjd: float = Field(..., gt=0, description="搜索结束时间 [MJD]")
    time_step_hours: float = Field(default=1.0, gt=0, description="搜索步长 [小时]")


class LaunchWindowResponse(BaseModel):
    """发射窗口分析响应。"""
    status: str = Field(..., description="执行状态")
    windows: List[Dict[str, Any]] = Field(default_factory=list)
    total_windows: int = Field(default=0)
    best_window: Optional[Dict[str, Any]] = Field(default=None)
    errors: List[str] = Field(default_factory=list)


# ============================================================
# 月球转移请求/响应
# ============================================================

class LunarTransferRequest(BaseModel):
    """月球转移轨道请求。"""
    departure_orbit: OrbitState = Field(..., description="出发轨道状态")
    target_perilune_km: float = Field(..., gt=0, description="目标近月点高度 [km]")
    departure_time_mjd: float = Field(..., gt=0, description="出发时间 [MJD]")
    tof_range_days: List[float] = Field(default=[3.0, 5.0], description="飞行时间范围 [天]")


class LunarTransferResponse(BaseModel):
    """月球转移轨道响应。"""
    status: str = Field(..., description="执行状态")
    delta_v_total_ms: Optional[float] = Field(default=None, description="总速度增量 [m/s]")
    transfer_time_days: Optional[float] = Field(default=None, description="转移时间 [天]")
    trajectory_points: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ============================================================
# 工具调用请求/响应
# ============================================================

class ToolCallRequest(BaseModel):
    """工具调用请求。"""
    tool_name: str = Field(..., min_length=1, description="工具名称")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="工具参数")
    is_read_only: bool = Field(default=True, description="是否只读操作")
    timeout_seconds: float = Field(default=30.0, gt=0, description="超时 [秒]")


class ToolCallResponse(BaseModel):
    """工具调用响应。"""
    tool_name: str = Field(..., min_length=1, description="工具名称")
    status: str = Field(..., description="执行状态: success / error / timeout")
    result: Any = Field(default=None, description="工具返回结果")
    error: Optional[str] = Field(default=None, description="错误信息")
    duration_ms: float = Field(default=0.0, ge=0, description="执行耗时 [毫秒]")

    @model_validator(mode="after")
    def error_required_for_non_success(self) -> "ToolCallResponse":
        if self.status != "success" and not self.error:
            raise ValueError("error is required when tool call status is not success")
        return self


# ============================================================
# RAG 查询请求/响应
# ============================================================

class RagQueryRequest(BaseModel):
    """RAG 查询请求。"""
    query: str = Field(..., min_length=1, description="查询文本")
    top_k: int = Field(default=5, ge=1, le=50, description="返回文档数")
    sources: List[str] = Field(
        default=["memory", "code", "orbit_dynamics"],
        description="检索源: memory / code / orbit_dynamics / papers"
    )
    require_verification: bool = Field(default=True, description="是否验证检索结果")


class RagQueryResponse(BaseModel):
    """RAG 查询响应。"""
    query: str = Field(..., description="原始查询")
    documents: List[Dict[str, str]] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)
    verified: bool = Field(default=False)
    verification_notes: List[str] = Field(default_factory=list)
    retrieval_time_ms: float = Field(default=0.0)


# ============================================================
# Agent 输入/输出（顶层协议）
# ============================================================

def _safe_relative_path(value: Union[str, Path], *, field_name: str) -> str:
    """Validate relative paths independent of the host operating system."""
    text = str(value).strip()
    if not text or text == ".":
        raise ValueError(f"{field_name} must be non-empty and relative")
    # Check both grammars: tests and API clients may submit POSIX paths on
    # Windows (or Windows paths on Linux).
    if (
        PurePosixPath(text).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or PureWindowsPath(text).drive
        or re.match(r"^[A-Za-z]:", text)
        or text.startswith(("\\", "/"))
    ):
        raise ValueError(f"{field_name} must be relative")
    if any(part == ".." for part in re.split(r"[\\\\/]", text)):
        raise ValueError(f"{field_name} must be relative and cannot contain '..'")
    return text.replace("\\", "/")


class AgentInput(BaseModel):
    """Versioned, validated input boundary for one agent run."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    user_message: str = Field(..., min_length=1)
    thread_id: str = Field(default="default", min_length=1)
    run_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    mode: str = Field(default="react", min_length=1)
    max_steps: int = Field(default=15, ge=1, le=100)
    recursion_limit: int = Field(default=40, ge=1, le=1000)
    context: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("user_message", "thread_id", "run_id", "mode")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must be non-empty")
        return value

    @model_validator(mode="after")
    def recursion_budget_is_outer_bound(self) -> "AgentInput":
        if self.recursion_limit <= self.max_steps:
            raise ValueError("recursion_limit must be greater than max_steps")
        return self


class EvidenceItem(BaseModel):
    """A bounded, traceable citation returned by retrieval."""

    source_id: str = Field(..., min_length=1, max_length=256)
    page_path: str = Field(..., min_length=1, max_length=1024)
    chunk_id: str = Field(..., min_length=1, max_length=256)
    score: float = Field(..., ge=0.0, le=1.0)
    excerpt: str = Field(..., min_length=1, max_length=4000)
    page_id: Optional[str] = Field(default=None, max_length=256)
    title: Optional[str] = Field(default=None, max_length=512)
    source_uri: Optional[str] = Field(default=None, max_length=2048)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id", "page_path", "chunk_id")
    @classmethod
    def non_empty_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("identifier must be non-empty")
        return value

    @field_validator("page_id", "title", "source_uri")
    @classmethod
    def optional_metadata_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("source metadata identifiers must be non-empty")
        return value

    @field_validator("page_path")
    @classmethod
    def relative_wiki_path(cls, value: str) -> str:
        return _safe_relative_path(value, field_name="page_path")

    @field_validator("metadata")
    @classmethod
    def bounded_source_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if len(value) > 50:
            raise ValueError("source metadata is too large")
        for key, item in value.items():
            if not str(key).strip() or len(str(key)) > 128:
                raise ValueError("source metadata keys must be non-empty and bounded")
            if key in {"source_id", "page_id", "page_path", "chunk_id", "content_sha256"}:
                if item is None or not str(item).strip():
                    raise ValueError(f"source metadata {key} must be non-empty")
            if isinstance(item, str) and len(item) > 4096:
                raise ValueError("source metadata values must be bounded")
            if key == "page_path":
                _safe_relative_path(str(item), field_name="source metadata page_path")
        return value


class Decision(BaseModel):
    """Planner decision exchanged between graph nodes."""

    action: ActionType
    rationale: str = Field(..., min_length=1)
    next_action: Optional[str] = Field(default=None)
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_request: Optional[ToolCallRequest] = Field(default=None)

    @field_validator("rationale")
    @classmethod
    def rationale_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("rationale must be non-empty")
        return value

    @model_validator(mode="after")
    def tool_request_for_call_tool(self) -> "Decision":
        if self.action == ActionType.CALL_TOOL and self.tool_request is None:
            raise ValueError("tool_request is required for call_tool")
        return self


class EvolutionFileChange(BaseModel):
    """One constrained, relative workspace file operation."""

    operation: Literal["create", "update", "delete"]
    path: Path
    content: Optional[str] = None

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: Path) -> Path:
        return Path(_safe_relative_path(value, field_name="path"))

    @model_validator(mode="after")
    def content_required_for_writes(self) -> "EvolutionFileChange":
        if self.operation in {"create", "update"} and self.content is None:
            raise ValueError("content is required for create/update")
        if self.operation == "delete" and self.content is not None:
            raise ValueError("content must be omitted for delete")
        return self


class EvolutionProposal(BaseModel):
    """Reviewable proposal for isolated workspace evolution."""

    thread_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    checkpoint_id: Optional[str] = Field(default=None)
    rationale: str = Field(..., min_length=1)
    changes: List[EvolutionFileChange] = Field(default_factory=list)
    source: Dict[str, Any] = Field(default_factory=dict)
    unfinished_items: List[str] = Field(default_factory=list)
    required_validations: List[str] = Field(default_factory=list)


class ManifestEntry(BaseModel):
    """One target snapshot in an evolution transaction manifest."""

    index: int = 0
    path: str
    operation: str
    prior_exists: bool = False
    prior_bytes: Any = None
    prior_mode: Optional[int] = None
    mode: Optional[int] = None
    before_sha256: Optional[str] = None
    after_sha256: Optional[str] = None
    after_bytes: Any = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class EvolutionRecord(BaseModel):
    """Durable evolution transaction record and validation report references."""

    evolution_id: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    status: Literal[
        "proposed", "staged", "backed_up", "validating", "committed",
        "rollback_requested", "validation_failed", "commit_failed",
        "rolled_back", "conflict",
    ] = "proposed"
    proposal: Optional[EvolutionProposal] = Field(default=None)
    # Durable transaction metadata exposed to callers and acceptance tests.
    state_history: List[str] = Field(default_factory=list)
    manifest: List[ManifestEntry] = Field(default_factory=list)
    before_manifest: List[Dict[str, Any]] = Field(default_factory=list)
    after_manifest: List[Dict[str, Any]] = Field(default_factory=list)
    validation_results: List[Dict[str, Any]] = Field(default_factory=list)
    validation_details: Dict[str, Any] = Field(default_factory=dict)
    # Derived knowledge artifacts are rebuilt as part of Wiki transactions.
    # Keep their state/error report on the record instead of hiding failures
    # in the journal.
    rebuild: Dict[str, Any] = Field(default_factory=dict)
    proposal_path: Optional[Path] = Field(default=None)
    manifest_path: Optional[Path] = Field(default=None)
    report_path: Optional[Path] = Field(default=None)
    checkpoint_id: Optional[str] = Field(default=None)
    source: Dict[str, Any] = Field(default_factory=dict)

    @property
    def rebuild_status(self) -> Dict[str, Any]:
        """Compatibility alias for integrations naming the rebuild report."""
        return self.rebuild



class ValidationResult(BaseModel):
    """Structured result returned by an evolution validator."""

    name: str = "validator"
    ok: bool = False
    message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.ok

    def as_dict(self) -> Dict[str, Any]:
        return self.model_dump()

class AgentOutput(BaseModel):
    """Versioned output protocol for every completed or interrupted run."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    status: RunStatus
    answer: str = Field(default="")
    intent: IntentType = Field(default=IntentType.GENERAL)
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citations: List[EvidenceItem] = Field(default_factory=list)
    tool_results: List[ToolCallResponse] = Field(default_factory=list)
    steps: int = Field(default=0, ge=0)
    cycle_triggers: int = Field(default=0, ge=0)
    checkpoint_id: Optional[str] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)
    # Keep structured error categories while accepting legacy strings.
    errors: List[Any] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Schema 验证工具函数
# ============================================================

def validate_input(data: Dict[str, Any]) -> AgentInput:
    """验证并解析 Agent 输入。

    Args:
        data: 原始输入字典

    Returns:
        验证后的 AgentInput 实例

    Raises:
        ValidationError: 输入不符合 Schema
    """
    return AgentInput(**data)


def validate_output(data: Dict[str, Any]) -> AgentOutput:
    """验证 Agent 输出是否符合 Schema。

    Args:
        data: Agent 输出字典

    Returns:
        验证后的 AgentOutput 实例

    Raises:
        ValidationError: 输出不符合 Schema
    """
    return AgentOutput(**data)


def export_json_schemas() -> Dict[str, Dict[str, Any]]:
    """导出所有 Schema 的 JSON Schema 定义。

    供外部系统消费（如 MCP 工具注册、API 文档生成）。

    Returns:
        {schema_name: json_schema_dict}
    """
    schemas = [
        OrbitState, KeplerianOrbitState,
        OrbitDesignRequest, OrbitDesignResponse,
        LaunchWindowRequest, LaunchWindowResponse,
        LunarTransferRequest, LunarTransferResponse,
        ToolCallRequest, ToolCallResponse,
        RagQueryRequest, RagQueryResponse,
        Decision, EvidenceItem, EvolutionFileChange,
        EvolutionProposal, EvolutionRecord, ValidationResult,
        AgentInput, AgentOutput,
    ]
    return {s.__name__: s.model_json_schema() for s in schemas}
