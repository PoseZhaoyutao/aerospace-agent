"""工作流工具 — 搜索、生成、列出航天动力学工作流。

第一性原理（K3 工作流编排）：
  1. 工作流是 ReAct 步骤的元编排——每步绑定一个 MCP 工具
  2. 生成的工作流必须包含 goal/inputs/models/engine/steps/outputs/validation/failure_handling
  3. 从已有 catalog 和 demo 索引中检索可复用模板，避免重复造轮子
  4. 工作流规格必须可序列化为 WorkflowSpec，供 Loop 引擎执行
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..schemas import WorkflowSpec, WorkflowStep
from ..adapters import get_all_adapters


def search_workflows(query: str, task_type: Optional[str] = None,
                     preferred_engine: Optional[str] = None) -> Dict:
    """搜索工作流目录，返回匹配的候选工作流。

    Args:
        query: 搜索关键词（如 "LEO 轨道传播"、"地面站可见性"）
        task_type: 任务类型过滤（orbit_propagation/ground_access/...）
        preferred_engine: 偏好引擎过滤
    Returns:
        {total, candidates: [{workflow_id, source, use_case, risks, reuse_notes}]}
    """
    catalog = _load_catalog()
    query_lower = query.lower().strip()
    candidates: List[Dict] = []

    for wf in catalog:
        # 任务类型过滤
        if task_type and wf.get("task_type", "") != task_type:
            continue
        # 引擎偏好过滤
        if preferred_engine and wf.get("engine", "auto") != preferred_engine \
                and wf.get("engine", "auto") != "auto":
            continue
        # 关键词匹配
        searchable = " ".join([
            wf.get("id", ""), wf.get("goal", ""),
            wf.get("task_type", ""), wf.get("description", ""),
        ]).lower()
        if query_lower and query_lower not in searchable:
            continue
        candidates.append({
            "workflow_id": wf.get("id", ""),
            "source": wf.get("source", "catalog"),
            "use_case": wf.get("goal", ""),
            "task_type": wf.get("task_type", ""),
            "engine": wf.get("engine", "auto"),
            "risks": wf.get("risks", []),
            "reuse_notes": wf.get("reuse_notes", ""),
            "steps_count": len(wf.get("steps", [])),
        })

    return {
        "total": len(candidates),
        "query": query,
        "candidates": candidates,
    }


def generate_astrodynamics_workflow(user_requirement: str,
                                    candidate_workflow_id: Optional[str] = None,
                                    constraints=None) -> Dict:
    """根据用户需求生成 WorkflowSpec。

    Args:
        user_requirement: 用户的自然语言需求描述
        candidate_workflow_id: 候选工作流 ID（基于已有模板生成）
        constraints: 约束条件，支持两种形式：
            - Dict: 如 {engine:"poliastro", max_duration_s:86400}
            - List[str]: 如 ["精度<1km", "二体模型"]（Loop 引擎传入）
    Returns:
        WorkflowSpec 字典（含 goal/inputs/models/engine/steps/outputs/
        validation/failure_handling）
    """
    # 统一 constraints 为 Dict——Loop 引擎传入 List[str] 时自动包装
    if constraints is None:
        constraints = {}
    elif isinstance(constraints, list):
        constraints = {"constraint_list": constraints, "raw": constraints}

    # 若指定了候选模板，基于模板生成
    base_spec = None
    if candidate_workflow_id:
        base_spec = _find_catalog_workflow(candidate_workflow_id)

    task_type = _infer_task_type(user_requirement)
    engine = constraints.get("engine", _infer_engine(task_type, user_requirement))

    # 构建 WorkflowSpec
    spec = WorkflowSpec(
        id=_generate_id(user_requirement, candidate_workflow_id),
        goal=user_requirement,
        task_type=task_type,
        engine=engine,
    )

    if base_spec:
        spec.inputs = base_spec.get("inputs", {})
        spec.models = base_spec.get("models", {})
        spec.steps = [WorkflowStep.from_dict(s)
                      for s in base_spec.get("steps", [])]

    # 根据任务类型填充默认步骤
    if not spec.steps:
        spec.steps = _default_steps(task_type, engine)

    # 输出定义
    spec.outputs = _default_outputs(task_type)

    # 验证策略
    spec.validation = _default_validation(task_type)

    # 失败处理
    spec.failure_handling = {
        "on_engine_unavailable": "fallback_to_builtin",
        "on_validation_failure": "retry_with_adjusted_params",
        "max_retries": 3,
    }

    # 元数据
    spec.metadata = {
        "source": "generated" if not base_spec else "catalog_template",
        "base_workflow_id": candidate_workflow_id or "",
        "constraints": constraints,
        "available_engines": _available_engines(),
    }

    return spec.to_dict()


def list_workflow_templates(task_type: Optional[str] = None) -> Dict:
    """列出工作流目录中的可用模板。

    Args:
        task_type: 可选的任务类型过滤
    Returns:
        {total, templates: [{id, goal, task_type, engine, steps_count}]}
    """
    catalog = _load_catalog()
    templates = []
    for wf in catalog:
        if task_type and wf.get("task_type", "") != task_type:
            continue
        templates.append({
            "id": wf.get("id", ""),
            "goal": wf.get("goal", ""),
            "task_type": wf.get("task_type", ""),
            "engine": wf.get("engine", "auto"),
            "steps_count": len(wf.get("steps", [])),
        })
    return {"total": len(templates), "templates": templates}


# ----------------------------------------------------------------------
# 内部辅助
# ----------------------------------------------------------------------

def _load_catalog() -> List[Dict]:
    """加载工作流目录（resources/workflow_catalog.py 的 WorkflowCatalog 类）。

    优先从配置的 workflows 目录加载 YAML；无目录时回退到内置目录。
    """
    # 尝试使用 WorkflowCatalog 加载 YAML 目录
    try:
        from ..resources.workflow_catalog import WorkflowCatalog
        import os
        wf_dir = os.environ.get("ASTRO_DYNAMICS_WORKFLOWS", "")
        catalog = WorkflowCatalog()
        if wf_dir and os.path.isdir(wf_dir):
            catalog.load_from_dir(wf_dir)
            if len(catalog) > 0:
                return catalog.list_all()
    except Exception:
        pass
    return _builtin_catalog()


def _find_catalog_workflow(wf_id: str) -> Optional[Dict]:
    """从目录中按 ID 查找工作流。"""
    # 优先用 WorkflowCatalog.get()
    try:
        from ..resources.workflow_catalog import WorkflowCatalog
        import os
        wf_dir = os.environ.get("ASTRO_DYNAMICS_WORKFLOWS", "")
        catalog = WorkflowCatalog()
        if wf_dir and os.path.isdir(wf_dir):
            catalog.load_from_dir(wf_dir)
            wf = catalog.get(wf_id)
            if wf:
                return wf
    except Exception:
        pass
    # 回退到内置目录
    for wf in _builtin_catalog():
        if wf.get("id") == wf_id:
            return wf
    return None


def _builtin_catalog() -> List[Dict]:
    """内置最小工作流目录（catalog 不可用时回退）。"""
    return [
        {
            "id": "wf-leo-propagation",
            "goal": "近地轨道二体传播",
            "task_type": "orbit_propagation",
            "engine": "auto",
            "steps": [
                {"name": "propagate", "tool": "propagate_orbit",
                 "inputs": {}, "outputs": ["state_history"]},
            ],
            "inputs": {"initial_state": "OrbitState", "duration_s": "float"},
            "models": {"force_model": "two_body"},
        },
        {
            "id": "wf-ground-access",
            "goal": "卫星对地面站可见性计算",
            "task_type": "ground_access",
            "engine": "auto",
            "steps": [
                {"name": "access", "tool": "compute_ground_access",
                 "inputs": {}, "outputs": ["access_windows"]},
            ],
            "inputs": {"orbit_state": "OrbitState", "station": "GroundStation"},
        },
        {
            "id": "wf-frame-transform",
            "goal": "GCRF 到 ITRF 坐标系转换",
            "task_type": "frame_transform",
            "engine": "astropy",
            "steps": [
                {"name": "transform", "tool": "transform_frame",
                 "inputs": {"target_frame": "ITRF"}, "outputs": ["state"]},
            ],
        },
    ]


def _infer_task_type(req: str) -> str:
    lower = req.lower()
    if any(k in lower for k in ("传播", "propag", "轨道外推")):
        return "orbit_propagation"
    if any(k in lower for k in ("可见", "access", "地面站", "visibility")):
        return "ground_access"
    if any(k in lower for k in ("坐标", "frame", "转换", "transform")):
        return "frame_transform"
    if any(k in lower for k in ("星历", "ephemeris", "spice")):
        return "ephemeris_query"
    if any(k in lower for k in ("时间", "time", "epoch")):
        return "time_conversion"
    return "orbit_propagation"


def _infer_engine(task_type: str, req: str) -> str:
    lower = req.lower()
    if "poliastro" in lower:
        return "poliastro"
    if "orekit" in lower:
        return "orekit"
    if "gmat" in lower:
        return "gmat"
    if task_type == "frame_transform":
        return "astropy"
    if task_type == "ephemeris_query":
        return "spiceypy"
    return "auto"


def _default_steps(task_type: str, engine: str) -> List[WorkflowStep]:
    # 默认 LEO 轨道状态（ISS-like，400km 圆轨道）
    import math as _m
    _alt = 400_000.0  # 400 km
    _r = 6_378_137.0 + _alt
    _v = _m.sqrt(3.986004418e14 / _r)
    _default_orbit = {
        "epoch": {"value": "2026-01-01T00:00:00", "scale": "UTC", "format": "ISO"},
        "frame": {"name": "GCRF", "center": "Earth", "realization": "IERS2010"},
        "representation": "cartesian",
        "position_m": [_r, 0.0, 0.0],
        "velocity_mps": [0.0, _v, 0.0],
    }
    _default_force = {
        "central_body": "Earth",
        "gravity": "point_mass",
        "degree": 0, "order": 0,
        "drag": {"enabled": False}, "srp": {"enabled": False},
        "third_body": [], "relativity": False,
    }
    _default_station = {
        "name": "Beijing_Station",
        "latitude_deg": 40.0789, "longitude_deg": 116.5867,
        "altitude_m": 50.0, "min_elevation_deg": 5.0,
    }
    _epoch_start = {"value": "2026-01-01T00:00:00", "scale": "UTC", "format": "ISO"}
    _epoch_stop = {"value": "2026-01-01T12:00:00", "scale": "UTC", "format": "ISO"}

    steps_map = {
        "orbit_propagation": [
            WorkflowStep(
                name="propagate", tool="propagate_orbit",
                inputs={
                    "initial_state_dict": _default_orbit,
                    "force_model_dict": _default_force,
                    "duration_s": 86400.0,
                    "output_step_s": 300.0,
                    "engine": engine,
                },
                outputs=["state_history"],
                description="二体轨道传播 1 天，300s 采样",
            ),
        ],
        "ground_access": [
            WorkflowStep(
                name="compute_access", tool="compute_ground_access",
                inputs={
                    "orbit_state_dict": _default_orbit,
                    "ground_station_dict": _default_station,
                    "start_epoch_dict": _epoch_start,
                    "stop_epoch_dict": _epoch_stop,
                    "min_elevation_deg": 5.0,
                },
                outputs=["access_windows"],
                description="计算 12 小时内北京站可见性窗口",
            ),
        ],
        "frame_transform": [
            WorkflowStep(
                name="transform", tool="transform_frame",
                inputs={
                    "state_dict": _default_orbit,
                    "target_frame": "ITRF",
                },
                outputs=["state"],
                description="GCRF → ITRF 坐标系转换",
            ),
        ],
        "ephemeris_query": [
            WorkflowStep(
                name="query", tool="query_ephemeris_state",
                inputs={
                    "target": "Moon",
                    "observer": "Earth",
                    "epoch_dict": _epoch_start,
                    "frame": "GCRF",
                    "aberration_correction": "NONE",
                },
                outputs=["position_m", "velocity_mps"],
                description="查询月球相对地球的星历状态",
            ),
        ],
        "time_conversion": [
            WorkflowStep(
                name="convert", tool="convert_time",
                inputs={
                    "value": "2026-01-01T00:00:00",
                    "from_scale": "UTC", "from_format": "ISO",
                    "to_scale": "TDB", "to_format": "JD",
                },
                outputs=["output"],
                description="UTC ISO → TDB JD 时间转换",
            ),
        ],
    }
    return steps_map.get(task_type, [WorkflowStep(
        name="propagate", tool="propagate_orbit",
        inputs={
            "initial_state_dict": _default_orbit,
            "force_model_dict": _default_force,
            "duration_s": 86400.0,
            "engine": "auto",
        },
        outputs=["state_history"])])


def _default_outputs(task_type: str) -> Dict[str, str]:
    return {
        "orbit_propagation": {"state_history": "List[OrbitState]"},
        "ground_access": {"access_windows": "List[AccessWindow]"},
        "frame_transform": {"state": "OrbitState"},
        "ephemeris_query": {"position_m": "List[float]",
                            "velocity_mps": "List[float]"},
        "time_conversion": {"output": "Epoch"},
    }.get(task_type, {"result": "dict"})


def _default_validation(task_type: str) -> Dict:
    return {
        "method": "cross_engine" if task_type == "orbit_propagation" else "sanity_check",
        "thresholds": {
            "position_error_m": 1000.0,
            "velocity_error_mps": 1.0,
        },
        "engines_for_cross_check": ["poliastro", "orekit", "builtin"],
    }


def _available_engines() -> List[str]:
    adapters = get_all_adapters()
    return [e for e, a in adapters.items() if a.is_available()]


def _generate_id(req: str, base_id: Optional[str]) -> str:
    import hashlib
    h = hashlib.md5(req.encode()).hexdigest()[:8]
    prefix = base_id or "wf-gen"
    return f"{prefix}-{h}"


__all__ = ["search_workflows", "generate_astrodynamics_workflow",
           "list_workflow_templates"]
