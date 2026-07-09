"""aerospace_agent.mcp — 统一航天动力学 MCP Server（集成于 Agent 内部）。

架构层次：
    schemas/       Canonical Astrodynamics Model（统一中间层）
    adapters/      7 引擎适配器（orekit/gmat/spiceypy/astropy/poliastro/basilisk/stk）
    tools/         12 个 MCP 工具（白名单封装，LLM 不可直接调底层库）
    loop/          Loop 引擎（Plan→Select→Retrieve→Generate→Run→Validate→Fix→Save）
    resources/     工作流目录、Demo 索引、Kernel 注册表
    safety/        许可检查、沙箱、路径策略
    prompts/       MCP 提示模板
    server.py      MCP Server 入口

该模块已从独立的 astro_dynamics_mcp 包集成到 aerospace_agent 内部，
作为 Agent 的 MCP 工具箱，通过 mcp_tools/ 桥接层供 ReAct Agent 调用。
"""
from .schemas import (
    Epoch, Frame, Body, OrbitState, AttitudeState,
    ForceModel, PropagatorConfig, GroundStation, SpacecraftConfig,
    WorkflowSpec, WorkflowResult, ValidationReport,
    LoopPhase, LoopLedgerEntry,
)
from .adapters.base import BaseAdapter, AdapterError

__version__ = "0.1.0"

__all__ = [
    "Epoch", "Frame", "Body", "OrbitState", "AttitudeState",
    "ForceModel", "PropagatorConfig", "GroundStation", "SpacecraftConfig",
    "WorkflowSpec", "WorkflowResult", "ValidationReport",
    "LoopPhase", "LoopLedgerEntry",
    "BaseAdapter", "AdapterError",
    "__version__",
]
