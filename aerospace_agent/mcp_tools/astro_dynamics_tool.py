"""AstroDynamicsMCPTool — 将 astro_dynamics_mcp 的 12 个工具桥接到 Agent。

本模块是集成层：把独立的 astro_dynamics_mcp 包的 12 个工具包装为
现有 BaseTool 子类，使 AerospaceAgent 能通过 register_mcp_tool() 自动加载。

设计原则：
  1. 适配器模式——不修改 astro_dynamics_mcp 源码，只包装调用
  2. 每个方法对应一个 MCP 工具，方法名 = 工具名
  3. 返回标准 {success, source, result, message} 格式
  4. astro_dynamics_mcp 不可用时静默回退（source='unavailable'）
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseTool


def _import_mcp():
    """懒加载 aerospace_agent.mcp 工具包。"""
    try:
        from ..mcp.tools import TOOL_REGISTRY
        return type("MCPModule", (), {"TOOL_REGISTRY": TOOL_REGISTRY})()
    except Exception:
        return None


class AstroDynamicsMCPTool(BaseTool):
    """航天动力学 MCP 工具桥接器。

    将 astro_dynamics_mcp 的 12 个工具统一包装为 BaseTool 接口，
    让 AerospaceAgent 能像调用 orekit_tool / gmat_tool 一样调用。

    方法（对应 12 个 MCP 工具）：
        check_engine_availability
        index_reference_demos
        search_workflows
        generate_astrodynamics_workflow
        convert_time
        transform_frame
        query_ephemeris_state
        convert_orbit_representation
        propagate_orbit
        compute_ground_access
        run_gmat_script
        cross_validate_results
    """

    name = "astro_dynamics"
    description = (
        "航天动力学 MCP 工具箱（12 工具）：引擎检测、工作流搜索/生成、"
        "时间转换、坐标系转换、星历查询、轨道传播、地面可见性、"
        "GMAT 脚本执行、交叉验证"
    )
    library_name = "astro_dynamics_mcp"

    methods_schema: Dict[str, Dict[str, Any]] = {
        "check_engine_availability": {
            "params": {"engines": "list (可选)"},
            "returns": "dict — 7 引擎可用性/版本/能力/许可证",
        },
        "index_reference_demos": {
            "params": {"sources": "list", "scan_paths": "list"},
            "returns": "dict — Demo 元数据索引（只读）",
        },
        "search_workflows": {
            "params": {"query": "str", "task_type": "str", "preferred_engine": "str"},
            "returns": "dict — 候选工作流列表",
        },
        "generate_astrodynamics_workflow": {
            "params": {"user_requirement": "str", "candidate_workflow_id": "str"},
            "returns": "dict — WorkflowSpec",
        },
        "convert_time": {
            "params": {"value": "str/float", "from_scale": "str",
                       "from_format": "str", "to_scale": "str", "to_format": "str"},
            "returns": "dict — 转换后时间 + 引擎 + 尺度",
        },
        "transform_frame": {
            "params": {"state_dict": "dict", "target_frame": "str"},
            "returns": "dict — 转换后 OrbitState + 引擎",
        },
        "query_ephemeris_state": {
            "params": {"target": "str", "observer": "str", "epoch_dict": "dict",
                       "frame": "str", "kernels": "list"},
            "returns": "dict — position/velocity + kernel list",
        },
        "convert_orbit_representation": {
            "params": {"state_dict": "dict", "target_representation": "str", "mu": "float"},
            "returns": "dict — 转换后 OrbitState",
        },
        "propagate_orbit": {
            "params": {"initial_state_dict": "dict", "force_model_dict": "dict",
                       "duration_s": "float", "output_step_s": "float", "engine": "str"},
            "returns": "dict — state_history + metadata",
        },
        "compute_ground_access": {
            "params": {"orbit_state_dict": "dict", "ground_station_dict": "dict",
                       "start_epoch_dict": "dict", "stop_epoch_dict": "dict",
                       "min_elevation_deg": "float"},
            "returns": "dict — access_windows 列表",
        },
        "run_gmat_script": {
            "params": {"script_text": "str", "script_path": "str", "workspace": "str"},
            "returns": "dict — stdout/stderr/output_files",
        },
        "cross_validate_results": {
            "params": {"task_spec": "dict", "engines": "list", "existing_results": "list"},
            "returns": "dict — position_error/velocity_error/confidence",
        },
    }

    def __init__(self):
        self._mcp_tools = None
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        """懒加载 astro_dynamics_mcp 工具注册表。"""
        if self._loaded:
            return self._mcp_tools is not None
        self._loaded = True
        mod = _import_mcp()
        if mod is not None:
            self._mcp_tools = getattr(mod, "TOOL_REGISTRY", None)
        return self._mcp_tools is not None

    @property
    def is_available(self) -> bool:
        """astro_dynamics_mcp 是否可导入。"""
        return self._ensure_loaded()

    @property
    def source(self) -> str:
        return "real" if self.is_available else "unavailable"

    def call(self, method: str, **kwargs) -> dict:
        """统一调用入口——转发到对应的 MCP 工具函数。"""
        if not self._ensure_loaded():
            return self._unavailable(method, "astro_dynamics_mcp",
                                     "pip install -e astro_dynamics_mcp")
        tool_fn = self._mcp_tools.get(method)
        if tool_fn is None:
            return self._fail(
                error=f"未知方法 '{method}'，可用: {self.list_methods()}",
                source="unavailable",
                message=f"astro_dynamics_mcp 无 '{method}' 工具",
            )
        try:
            result = tool_fn(**kwargs)
            # 判断结果是否为错误
            if isinstance(result, dict) and result.get("status") == "error":
                return self._fail(
                    error=result.get("reason", "未知错误"),
                    source="real",
                    message=f"{method} 执行失败",
                )
            return self._ok(
                result=result,
                source="real",
                message=f"{method} 执行成功",
            )
        except Exception as e:
            return self._fail(
                error=str(e),
                source="real",
                message=f"{method} 执行异常",
            )


def load_astro_dynamics_tools() -> List[BaseTool]:
    """加载所有 astro_dynamics_mcp 桥接工具。

    返回 [AstroDynamicsMCPTool]。
    不可用时返回空列表（静默回退）。

    .. note::
        LoopEngineTool 已移除——八阶段 Loop 编排改为由 Agent 直接调用
        BaseWorkflow + 精简 ReAct 实现，减少元工作流开销。
    """
    tools: List[BaseTool] = []
    try:
        tools.append(AstroDynamicsMCPTool())
    except Exception:
        pass
    return tools
