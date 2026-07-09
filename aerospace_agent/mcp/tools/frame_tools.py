"""坐标系转换工具 — 跨参考系的轨道状态变换。

第一性原理（K3 坐标系一致性）：
  1. 坐标值没有坐标系标签就是无意义的数字
  2. 惯性系↔固连系转换需要地球定向参数（ERA/UT1）
  3. 引擎优先级：astropy（GCRF/ITRF）> spiceypy（需 kernel）> orekit（预留）
  4. 输出必须显式标注 frame 和 engine_used
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from ..schemas import OrbitState, Frame, FrameName, FrameCenter, Epoch
from ..adapters import get_adapter

#: 支持的坐标系
_SUPPORTED_FRAMES = {
    "GCRF", "ICRF", "EME2000", "J2000", "ITRF", "TEME", "BodyFixed",
}

#: 惯性系等价组（GCRF/EME2000/J2000 在地心精度容限内可视为同一系）
# K5-C5: 移除 ICRF（太阳系质心系），与 GCRF（地心系）原点差约 1AU
_INERTIAL_EQUIV = {"GCRF", "EME2000", "J2000"}


def transform_frame(state_dict: Dict, target_frame: str,
                    target_center: Optional[str] = None) -> Dict:
    """将轨道状态转换到目标坐标系。

    Args:
        state_dict: OrbitState.to_dict() 格式的状态字典
        target_frame: 目标坐标系（GCRF/ICRF/EME2000/J2000/ITRF/TEME/BodyFixed）
        target_center: 目标中心天体（可选，默认保持原中心）
    Returns:
        {state, engine_used, frame_info} 字典
    """
    target_frame = target_frame.upper().strip()

    if target_frame not in _SUPPORTED_FRAMES:
        return _error(f"不支持的目标坐标系: '{target_frame}'，"
                      f"支持: {sorted(_SUPPORTED_FRAMES)}")

    try:
        state = OrbitState.from_dict(state_dict)
    except Exception as exc:
        return _error(f"输入状态解析失败: {exc}")

    src_frame = state.frame.name.value
    src_center = state.frame.center.value

    # 同系等价转换——直接返回
    if _is_inertial_equiv(src_frame) and _is_inertial_equiv(target_frame):
        return _identity_result(state, target_frame, target_center,
                                src_frame, "astropy")

    # 尝试 astropy
    result = _try_astropy_transform(state, target_frame, target_center)
    if result:
        return result

    # 尝试 spiceypy
    result = _try_spiceypy_transform(state, target_frame, target_center)
    if result:
        return result

    # 回退：简化的解析转换（仅 GCRF↔ITRF 一阶近似）
    return _fallback_analytic(state, target_frame, target_center, src_frame)


def _is_inertial_equiv(frame: str) -> bool:
    return frame in _INERTIAL_EQUIV


def _try_astropy_transform(state, target_frame, target_center) -> Optional[Dict]:
    """使用 astropy 进行坐标系转换。"""
    try:
        from astropy import coordinates as coord  # type: ignore
        from astropy.time import Time  # type: ignore
        import astropy.units as u  # type: ignore

        if state.position_m is None or state.velocity_mps is None:
            return None

        epoch_str = state.epoch.to_iso_utc()
        t = Time(epoch_str, scale="utc")

        src = state.frame.name.value

        # GCRF ↔ ITRF 转换
        if _is_inertial_equiv(src) and target_frame == "ITRF":
            # K5-C2: 正确使用 CartesianRepresentation + CartesianDifferential
            pos = coord.CartesianRepresentation(
                state.position_m[0] * u.m,
                state.position_m[1] * u.m,
                state.position_m[2] * u.m)
            vel = coord.CartesianRepresentation(
                state.velocity_mps[0] * u.m / u.s,
                state.velocity_mps[1] * u.m / u.s,
                state.velocity_mps[2] * u.m / u.s)
            pos_with_vel = pos.with_differentials(
                {"s": coord.CartesianDifferential(
                    state.velocity_mps[0] * u.m / u.s,
                    state.velocity_mps[1] * u.m / u.s,
                    state.velocity_mps[2] * u.m / u.s)})
            gcrs = coord.GCRS(pos_with_vel, obstime=t)
            itrs = gcrs.transform_to(coord.ITRS(obstime=t))
            new_pos = [float(itrs.x.value), float(itrs.y.value),
                       float(itrs.z.value)]
            itrs_vel = itrs.differentials["s"]
            new_vel = [float(itrs_vel.d_x.value), float(itrs_vel.d_y.value),
                       float(itrs_vel.d_z.value)]
            return _build_result(new_pos, new_vel, state, target_frame,
                                 target_center, "astropy",
                                 "astropy coordinates GCRS→ITRS 转换")

        if src == "ITRF" and _is_inertial_equiv(target_frame):
            # K5-C2: 同样使用 with_differentials 关联速度
            pos = coord.CartesianRepresentation(
                state.position_m[0] * u.m,
                state.position_m[1] * u.m,
                state.position_m[2] * u.m)
            pos_with_vel = pos.with_differentials(
                {"s": coord.CartesianDifferential(
                    state.velocity_mps[0] * u.m / u.s,
                    state.velocity_mps[1] * u.m / u.s,
                    state.velocity_mps[2] * u.m / u.s)})
            itrs = coord.ITRS(pos_with_vel, obstime=t)
            gcrs = itrs.transform_to(coord.GCRS(obstime=t))
            new_pos = [float(gcrs.x.value), float(gcrs.y.value),
                       float(gcrs.z.value)]
            gcrs_vel = gcrs.differentials["s"]
            new_vel = [float(gcrs_vel.d_x.value), float(gcrs_vel.d_y.value),
                       float(gcrs_vel.d_z.value)]
            return _build_result(new_pos, new_vel, state, target_frame,
                                 target_center, "astropy",
                                 "astropy coordinates ITRS→GCRS 转换")
    except Exception:
        pass
    return None


def _try_spiceypy_transform(state, target_frame, target_center) -> Optional[Dict]:
    """使用 spiceypy 进行坐标系转换（需 kernel）。"""
    adapter = get_adapter("spiceypy")
    if not adapter.is_available():
        return None
    try:
        import spiceypy as spice  # type: ignore
        # 需要 kernel 加载才能做旋转矩阵查询
        # 此处预留：实际实现需 furnsh 后调用 sxform
        return None
    except Exception:
        return None


def _fallback_analytic(state, target_frame, target_center, src_frame) -> Dict:
    """简化的解析转换——仅 GCRF↔ITRF 用地球自转一阶近似。"""
    if state.position_m is None:
        return _error("缺少 position_m，无法进行坐标系转换")

    src = src_frame
    if _is_inertial_equiv(src) and target_frame == "ITRF":
        theta = _earth_rotation_angle(state.epoch)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        px, py, pz = state.position_m
        vx, vy, vz = state.velocity_mps or [0, 0, 0]
        omega = 7.2921159e-5  # rad/s
        new_pos = [cos_t * px + sin_t * py, -sin_t * px + cos_t * py, pz]
        new_vel = [cos_t * vx + sin_t * vy + omega * new_pos[1],
                   -sin_t * vx + cos_t * vy - omega * new_pos[0], vz]
        return _build_result(new_pos, new_vel, state, target_frame,
                             target_center, "analytic_fallback",
                             "简化解析转换（仅地球自转一阶近似，忽略岁差章动极移）")

    if src == "ITRF" and _is_inertial_equiv(target_frame):
        theta = _earth_rotation_angle(state.epoch)
        cos_t, sin_t = math.cos(-theta), math.sin(-theta)
        px, py, pz = state.position_m
        vx, vy, vz = state.velocity_mps or [0, 0, 0]
        omega = 7.2921159e-5
        new_pos = [cos_t * px + sin_t * py, -sin_t * px + cos_t * py, pz]
        new_vel = [cos_t * vx + sin_t * vy - omega * new_pos[1],
                   -sin_t * vx + cos_t * vy + omega * new_pos[0], vz]
        return _build_result(new_pos, new_vel, state, target_frame,
                             target_center, "analytic_fallback",
                             "简化解析转换（仅地球自转一阶近似，忽略岁差章动极移）")

    return _error(f"无法完成 {src}→{target_frame} 转换："
                  "astropy/spiceypy 均不可用且无解析回退")


def _earth_rotation_angle(epoch: Epoch) -> float:
    """K5-M1: 地球自转角（ERA）—— IERS 2010 规范。

    ERA = 2π·(0.7790572732640 + 1.00273781191135448·Tu)
    其中 Tu = J2000 以来的 UT1 天数（此处用 UTC 近似）。
    """
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(str(epoch.value).replace("Z", "+00:00"))
    except Exception:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    days = (dt - j2000).total_seconds() / 86400.0
    return (2 * math.pi * (0.7790572732640 + 1.00273781191135448 * days)) % (2 * math.pi)


def _build_result(pos, vel, state, target_frame, target_center,
                  engine, notes) -> Dict:
    new_frame = Frame(
        name=FrameName(target_frame),
        center=FrameCenter(target_center) if target_center else state.frame.center,
    )
    new_state = OrbitState(
        epoch=state.epoch, frame=new_frame,
        position_m=pos, velocity_mps=vel,
    )
    return {
        "state": new_state.to_dict(),
        "engine_used": engine,
        "frame_info": {
            "source_frame": state.frame.name.value,
            "target_frame": target_frame,
            "target_center": target_center or state.frame.center.value,
            "units": "SI (m, m/s)",
        },
        "notes": notes,
    }


def _identity_result(state, target_frame, target_center,
                     src_frame, engine) -> Dict:
    new_frame = Frame(
        name=FrameName(target_frame),
        center=FrameCenter(target_center) if target_center else state.frame.center,
    )
    new_state = OrbitState(
        epoch=state.epoch, frame=new_frame,
        position_m=state.position_m, velocity_mps=state.velocity_mps,
    )
    return {
        "state": new_state.to_dict(),
        "engine_used": engine,
        "frame_info": {
            "source_frame": src_frame,
            "target_frame": target_frame,
            "target_center": target_center or state.frame.center.value,
            "units": "SI (m, m/s)",
        },
        "notes": f"{src_frame} 与 {target_frame} 为等价惯性系，无需旋转",
    }


def _error(reason) -> Dict:
    return {"status": "error", "reason": reason, "engine": None}


__all__ = ["transform_frame"]
