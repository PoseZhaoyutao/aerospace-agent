"""SPICE MCP 工具 — 基于 spiceypy 的星历、时间、坐标系、观测几何、掩星与 TLE 转换。

第一性原理（K2 白名单封装）：
  1. LLM 只能调用本模块注册的 MCP 工具——不可直接调 spiceypy
  2. 每个工具返回 JSON 可序列化字典
  3. 所有失败返回结构化 {status:"error", reason:...}——绝不静默失败
  4. 每个工具由 @error_handler 包装——统一异常转结构化错误
  5. 依赖 SpiceyPyAdapter 适配器层，不直接操作 spiceypy
"""
from __future__ import annotations

import functools
import traceback
from typing import Any, Callable, Dict, List, Optional

from ..adapters import get_adapter
from ..schemas import Epoch

# ------------------------------------------------------------------
# 本地 error_handler（避免与 server.py 循环导入）
# ------------------------------------------------------------------
def error_handler(func: Callable[..., Dict]) -> Callable[..., Dict]:
    """工具调用统一错误处理装饰器。

    捕获所有异常，转为结构化 {status:"error", reason:..., tool:...}。
    绝不静默失败。
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Dict:
        try:
            result = func(*args, **kwargs)
            if result is None:
                return {
                    "status": "error",
                    "reason": "工具返回 None——内部逻辑异常",
                    "tool": func.__name__,
                }
            return result
        except TypeError as exc:
            return {
                "status": "error",
                "reason": f"参数类型错误: {exc}",
                "tool": func.__name__,
            }
        except ValueError as exc:
            return {
                "status": "error",
                "reason": f"数值错误: {exc}",
                "tool": func.__name__,
            }
        except Exception as exc:
            return {
                "status": "error",
                "reason": f"未预期错误: {exc}",
                "tool": func.__name__,
                "traceback": traceback.format_exc().split("\n")[-5:],
            }
    return wrapper


# ------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------
def _resolve_epoch(epoch_dict: Dict) -> Epoch:
    """从 epoch_dict 解析 Epoch 对象。"""
    return Epoch.from_dict(epoch_dict)


def _check_adapter():
    """返回 spiceypy 适配器，不可用时返回 None。"""
    adapter = get_adapter("spiceypy")
    if not adapter.is_available():
        return None
    return adapter


# ------------------------------------------------------------------
# 8 个 SPICE MCP 工具
# ------------------------------------------------------------------

@error_handler
def spice_query_ephemeris(
    target: str,
    observer: str,
    epoch_dict: Dict,
    frame: str = "J2000",
    aberration_correction: str = "NONE",
    kernels: Optional[List[str]] = None,
) -> Dict:
    """星历查询：查询目标天体相对观察者的位置和速度。

    基于 SPICE spkezr，输出 SI 单位（m, m/s）。

    Args:
        target: 目标天体名或 NAIF ID（如 "Moon"、"399"、"MARS"）
        observer: 观察者天体名或 NAIF ID（如 "Earth"、"10"、"SUN"）
        epoch_dict: 时间字典 {value, scale, format}，如 {"value":"2026-01-01T00:00:00","scale":"UTC","format":"ISO"}
        frame: 参考系（J2000/ICRF/GCRF/ITRF93），默认 J2000
        aberration_correction: 光行差修正（NONE/LT/LT+S），默认 NONE
        kernels: 可选 kernel 路径列表；省略则从 SPICE_KERNELS 环境变量自动发现

    Returns:
        {position_m, velocity_mps, frame, epoch, et, light_time_s, kernel_list, engine}
    """
    adapter = _check_adapter()
    if adapter is None:
        return {
            "status": "error",
            "reason": "spiceypy 未安装或不可用。请 pip install spiceypy 并配置 SPICE_KERNELS。",
            "engine": "spiceypy",
        }

    try:
        epoch = _resolve_epoch(epoch_dict)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"epoch 解析失败: {exc}",
            "engine": "spiceypy",
        }

    # 加载 kernel（若提供）
    if kernels:
        for k in kernels:
            try:
                import spiceypy as spice
                spice.furnsh(k)
            except Exception as exc:
                return {
                    "status": "error",
                    "reason": f"kernel 加载失败 {k}: {exc}",
                    "engine": "spiceypy",
                }

    result = adapter.query_ephemeris(target, observer, epoch, frame,
                                     aberration_correction=aberration_correction)
    return result


@error_handler
def spice_transform_frame(
    state_dict: Dict,
    target_frame: str,
) -> Dict:
    """坐标系转换：将轨道状态转换到目标参考系。

    基于 SPICE sxform（6x6 状态旋转矩阵），包含速度交叉项。

    Args:
        state_dict: OrbitState.to_dict() 格式，含 position_m, velocity_mps, epoch, frame
        target_frame: 目标参考系名（J2000/ICRF/GCRF/ITRF93/IAU_EARTH 等）

    Returns:
        {position_m, velocity_mps, source_frame, target_frame}
    """
    adapter = _check_adapter()
    if adapter is None:
        return {
            "status": "error",
            "reason": "spiceypy 未安装或不可用。",
            "engine": "spiceypy",
        }

    # 构造一个简单的 state 对象给适配器
    class _State:
        pass

    state = _State()
    state.position_m = state_dict.get("position_m", [0.0, 0.0, 0.0])
    state.velocity_mps = state_dict.get("velocity_mps", [0.0, 0.0, 0.0])

    # frame 简化为 name 属性
    class _Frame:
        pass

    state.frame = _Frame()
    source_frame = state_dict.get("frame", "J2000")
    state.frame.name = source_frame

    try:
        epoch = _resolve_epoch(state_dict.get("epoch", {"value": "2000-01-01T12:00:00", "scale": "UTC", "format": "ISO"}))
        state.epoch = epoch
    except Exception:
        state.epoch = Epoch(value="2000-01-01T12:00:00", scale="UTC", format="ISO")

    result = adapter.transform_frame(state, target_frame)
    return result


@error_handler
def spice_convert_time(
    epoch_dict: Dict,
    target_scale: str,
) -> Dict:
    """时间转换：在 UTC/TAI/TDB/ET 时间尺度间转换。

    基于 SPICE str2et/et2utc/unitim。

    Args:
        epoch_dict: 时间字典 {value, scale, format}，如 {"value":"2026-01-01T00:00:00","scale":"UTC","format":"ISO"}
        target_scale: 目标时间尺度（UTC/TAI/TDB/ET）

    Returns:
        {epoch: {value, scale, format}}
    """
    adapter = _check_adapter()
    if adapter is None:
        return {
            "status": "error",
            "reason": "spiceypy 未安装或不可用。",
            "engine": "spiceypy",
        }

    try:
        epoch = _resolve_epoch(epoch_dict)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"epoch 解析失败: {exc}",
            "engine": "spiceypy",
        }

    result = adapter.convert_time(epoch, target_scale)
    return result


@error_handler
def spice_load_kernels(
    kernel_paths: List[str],
) -> Dict:
    """加载 SPICE kernel 文件。

    加载后 kernel 驻留在 SPICE 内核池中，后续星历/坐标系/观测几何查询均可使用。

    Args:
        kernel_paths: kernel 文件路径列表，如 ["/data/naif0012.tls", "/data/de440.bsp"]

    Returns:
        {loaded, failed, total_loaded}
    """
    adapter = _check_adapter()
    if adapter is None:
        return {
            "status": "error",
            "reason": "spiceypy 未安装或不可用。",
            "engine": "spiceypy",
        }

    result = adapter.load_kernels(kernel_paths)
    return result


@error_handler
def spice_list_loaded_kernels() -> Dict:
    """列出当前已加载的全部 SPICE kernel 文件。

    返回 SPICE 内核池中所有 kernel 的文件名、类型与来源。

    Returns:
        {count, kernels: [{file, type, source}, ...]}
    """
    adapter = _check_adapter()
    if adapter is None:
        return {
            "status": "error",
            "reason": "spiceypy 未安装或不可用。",
            "engine": "spiceypy",
        }

    result = adapter.list_loaded_kernels()
    return result


@error_handler
def spice_compute_observation_geometry(
    target: str,
    observer: str,
    epoch_dict: Dict,
    frame: str = "J2000",
) -> Dict:
    """计算观测几何：光照角、距离、亚观测点/亚日点。

    基于 SPICE ilumin/subpnt/subslr，输出角度（deg）、距离（m）。
    需要目标天体的 PCK 椭球模型已加载。

    Args:
        target: 目标天体名（如 "MOON"、"MARS"）
        observer: 观察者天体名（如 "EARTH"、"SUN"）
        epoch_dict: 时间字典 {value, scale, format}
        frame: 参考系，默认 J2000

    Returns:
        {phase_angle_deg, solar_incidence_angle_deg, emission_angle_deg,
         visible, target_observer_distance_m, target_sun_distance_m,
         sub_observer_lat_deg, sub_observer_lon_deg,
         sub_solar_lat_deg, sub_solar_lon_deg}
    """
    adapter = _check_adapter()
    if adapter is None:
        return {
            "status": "error",
            "reason": "spiceypy 未安装或不可用。",
            "engine": "spiceypy",
        }

    try:
        epoch = _resolve_epoch(epoch_dict)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"epoch 解析失败: {exc}",
            "engine": "spiceypy",
        }

    result = adapter.compute_observation_geometry(target, observer, epoch, frame)
    return result


@error_handler
def spice_compute_occultation(
    target: str,
    occulting_body: str,
    observer: str,
    epoch_dict: Dict,
    frame: str = "J2000",
) -> Dict:
    """计算掩星事件：判断目标天体是否被掩星体遮挡。

    基于角距离比较目标与掩星体视圆面是否重叠。
    需要目标与掩星体的 PCK 椭球模型已加载。

    Args:
        target: 被掩天体名（如 "MOON"、"MARS"、"IO"）
        occulting_body: 掩星体名（如 "EARTH"、"JUPITER"）
        observer: 观察者天体名（如 "SUN"、"EARTH"）
        epoch_dict: 时间字典 {value, scale, format}
        frame: 参考系，默认 J2000

    Returns:
        {occultation_code (0=无/1=部分/2=全), occultation_type,
         angular_separation_deg, target_angular_radius_deg,
         occulting_body_angular_radius_deg}
    """
    adapter = _check_adapter()
    if adapter is None:
        return {
            "status": "error",
            "reason": "spiceypy 未安装或不可用。",
            "engine": "spiceypy",
        }

    try:
        epoch = _resolve_epoch(epoch_dict)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"epoch 解析失败: {exc}",
            "engine": "spiceypy",
        }

    result = adapter.compute_occultation(target, occulting_body, observer, epoch, frame)
    return result


@error_handler
def spice_two_line_elements_to_state(
    line1: str,
    line2: str,
    frame: str = "J2000",
) -> Dict:
    """TLE（两行轨道根数）→ 轨道状态转换。

    解析 TLE 格式 Line 1/Line 2，提取开普勒轨道根数，解 Kepler 方程
    得到笛卡尔位置/速度（SI 单位）。使用二体模型，不含 SGP4 长期项。

    注意：此为简化模型，精度约 km 级。高精度需求请使用 sgp4 库。

    Args:
        line1: TLE 第一行，如 "1 25544U 98067A   20001.12345678  ..."
        line2: TLE 第二行，如 "2 25544  51.6435 123.4567 0001234  ..."
        frame: 目标参考系，默认 J2000

    Returns:
        {position_m, velocity_mps, frame, orbital_elements: {...}, note}
    """
    adapter = _check_adapter()
    if adapter is None:
        return {
            "status": "error",
            "reason": "spiceypy 未安装或不可用。",
            "engine": "spiceypy",
        }

    # TLE 工具不需要 epoch 输入（epoch 在 TLE 自身中），传一个占位 epoch
    epoch = Epoch(value="2000-01-01T12:00:00", scale="UTC", format="ISO")

    result = adapter.two_line_elements_to_state(line1, line2, epoch, frame)
    return result


__all__ = [
    "spice_query_ephemeris",
    "spice_transform_frame",
    "spice_convert_time",
    "spice_load_kernels",
    "spice_list_loaded_kernels",
    "spice_compute_observation_geometry",
    "spice_compute_occultation",
    "spice_two_line_elements_to_state",
]