"""tools — 12 个 MCP 工具的统一注册与导出。

第一性原理（K2 白名单封装）：
  1. LLM 只能调用 TOOL_REGISTRY 中注册的工具——不可直接调底层库
  2. 每个工具返回 JSON 可序列化字典
  3. 所有失败返回结构化 {status:"error", reason:...}——绝不静默失败
  4. get_tool_definitions() 生成 MCP 协议所需的 JSON Schema
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

# 环境工具
from .environment_tools import check_engine_availability, index_reference_demos
# 工作流工具
from .workflow_tools import (
    search_workflows, generate_astrodynamics_workflow, list_workflow_templates,
)
# 时间工具
from .time_tools import convert_time
# 坐标系工具
from .frame_tools import transform_frame
# 星历工具
from .ephemeris_tools import query_ephemeris_state
# 传播工具
from .propagation_tools import convert_orbit_representation, propagate_orbit
# 可见性工具
from .access_tools import compute_ground_access
# GMAT 工具
from .gmat_tools import run_gmat_script
# 验证工具
from .validation_tools import cross_validate_results
from .space_tools import (
    SPACE_TOOL_SPECS,
    get_space_tool_definitions,
    get_space_tool_specs,
    register_space_tools,
)

#: 工具注册表 — 名称 → 可调用对象
TOOL_REGISTRY: Dict[str, Callable[..., Dict]] = {
    "check_engine_availability": check_engine_availability,
    "index_reference_demos": index_reference_demos,
    "search_workflows": search_workflows,
    "generate_astrodynamics_workflow": generate_astrodynamics_workflow,
    "list_workflow_templates": list_workflow_templates,
    "convert_time": convert_time,
    "transform_frame": transform_frame,
    "query_ephemeris_state": query_ephemeris_state,
    "convert_orbit_representation": convert_orbit_representation,
    "propagate_orbit": propagate_orbit,
    "compute_ground_access": compute_ground_access,
    "run_gmat_script": run_gmat_script,
    "cross_validate_results": cross_validate_results,
}

register_space_tools(TOOL_REGISTRY)

#: 12 个核心 MCP 工具名（不含 list_workflow_templates 辅助工具）
CORE_TOOLS: List[str] = [
    "check_engine_availability",
    "index_reference_demos",
    "search_workflows",
    "generate_astrodynamics_workflow",
    "convert_time",
    "transform_frame",
    "query_ephemeris_state",
    "convert_orbit_representation",
    "propagate_orbit",
    "compute_ground_access",
    "run_gmat_script",
    "cross_validate_results",
]


def _register_research_tools() -> None:
    """将 105 个科研工具注册到 TOOL_REGISTRY（懒加载，仅调用一次）。"""
    if getattr(_register_research_tools, "_done", False):
        return
    _register_research_tools._done = True
    try:
        from ...research_tools import get_registry
        reg = get_registry()
        for name in reg.list_all():
            # 创建闭包绑定工具名
            def _make_caller(tool_name):
                def _caller(**kwargs):
                    return get_registry().call(tool_name, **kwargs)
                _caller.__name__ = f"research_{tool_name}"
                return _caller
            TOOL_REGISTRY[name] = _make_caller(name)
    except Exception:
        pass


def get_tool_definitions() -> List[Dict[str, Any]]:
    """返回 MCP 协议格式的工具定义（JSON Schema）。

    包含 12 个核心航天 MCP 工具 + 105 个科研原子工具。
    """
    _register_research_tools()

    # 核心航天工具定义（硬编码）
    core_defs = [
        {
            "name": "check_engine_availability",
            "description": "检查全部或指定引擎的可用性、版本、能力、数据路径和许可证状态。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "engines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "引擎名列表；省略则检查全部 7 个",
                    },
                },
            },
        },
        {
            "name": "index_reference_demos",
            "description": "扫描配置路径索引各引擎的示例/测试/教程（只读）。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "scan_paths": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        {
            "name": "search_workflows",
            "description": "搜索工作流目录，返回匹配的候选工作流。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "task_type": {"type": "string"},
                    "preferred_engine": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "generate_astrodynamics_workflow",
            "description": "根据用户需求生成完整 WorkflowSpec（含 goal/steps/outputs/validation）。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_requirement": {"type": "string"},
                    "candidate_workflow_id": {"type": "string"},
                    "constraints": {"type": "object"},
                },
                "required": ["user_requirement"],
            },
        },
        {
            "name": "convert_time",
            "description": "跨时间尺度（UTC/TAI/TT/TDB/ET）和格式（ISO/JD/MJD/UNIX）转换。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "value": {"type": ["string", "number"]},
                    "from_scale": {"type": "string"},
                    "from_format": {"type": "string"},
                    "to_scale": {"type": "string"},
                    "to_format": {"type": "string"},
                },
                "required": ["value"],
            },
        },
        {
            "name": "transform_frame",
            "description": "将轨道状态转换到目标坐标系（GCRF/ICRF/EME2000/J2000/ITRF/TEME/BodyFixed）。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state_dict": {"type": "object"},
                    "target_frame": {"type": "string"},
                    "target_center": {"type": "string"},
                },
                "required": ["state_dict", "target_frame"],
            },
        },
        {
            "name": "query_ephemeris_state",
            "description": "基于 SPICE kernel 查询目标天体相对观察者的位置和速度。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "observer": {"type": "string"},
                    "epoch_dict": {"type": "object"},
                    "frame": {"type": "string"},
                    "aberration_correction": {"type": "string"},
                    "kernels": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["target", "observer", "epoch_dict"],
            },
        },
        {
            "name": "convert_orbit_representation",
            "description": "轨道状态表示转换（笛卡尔↔开普勒），显式标注 mu 和 units。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state_dict": {"type": "object"},
                    "target_representation": {"type": "string"},
                    "mu": {"type": "number"},
                },
                "required": ["state_dict", "target_representation"],
            },
        },
        {
            "name": "propagate_orbit",
            "description": "轨道传播（二体+J2占位），支持 auto/poliastro/orekit/gmat 引擎。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "initial_state_dict": {"type": "object"},
                    "force_model_dict": {"type": "object"},
                    "duration_s": {"type": "number"},
                    "output_step_s": {"type": "number"},
                    "engine": {"type": "string"},
                },
                "required": ["initial_state_dict", "force_model_dict",
                             "duration_s"],
            },
        },
        {
            "name": "compute_ground_access",
            "description": "计算卫星对地面站的可见时间窗口。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "orbit_state_dict": {"type": "object"},
                    "ground_station_dict": {"type": "object"},
                    "start_epoch_dict": {"type": "object"},
                    "stop_epoch_dict": {"type": "object"},
                    "min_elevation_deg": {"type": "number"},
                },
                "required": ["orbit_state_dict", "ground_station_dict",
                             "start_epoch_dict", "stop_epoch_dict"],
            },
        },
        {
            "name": "run_gmat_script",
            "description": "在沙箱工作区内运行 GMAT 脚本。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "script_text": {"type": "string"},
                    "script_path": {"type": "string"},
                    "workspace": {"type": "string"},
                },
            },
        },
        {
            "name": "cross_validate_results",
            "description": "多引擎交叉验证——比较同一任务在不同引擎上的结果差异。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_spec": {"type": "object"},
                    "engines": {"type": "array", "items": {"type": "string"}},
                    "existing_results": {"type": "object"},
                },
                "required": ["task_spec"],
            },
        },
    ]

    core_defs.extend(get_space_tool_definitions())

    # 追加 105 个科研工具的 JSON Schema
    try:
        from ...research_tools import get_registry
        reg = get_registry()
        research_defs = reg.get_json_schemas()
        # 只添加尚未在 core_defs 中定义的工具
        core_names = {d["name"] for d in core_defs}
        for rd in research_defs:
            if rd["name"] not in core_names:
                core_defs.append(rd)
    except Exception:
        pass

    return core_defs


__all__ = [
    # 工具函数
    "check_engine_availability", "index_reference_demos",
    "search_workflows", "generate_astrodynamics_workflow",
    "list_workflow_templates",
    "convert_time", "transform_frame", "query_ephemeris_state",
    "convert_orbit_representation", "propagate_orbit",
    "compute_ground_access", "run_gmat_script",
    "cross_validate_results",
    # 注册表
    "TOOL_REGISTRY", "CORE_TOOLS", "get_tool_definitions",
    "SPACE_TOOL_SPECS", "get_space_tool_specs",
    "get_space_tool_definitions", "register_space_tools",
]
