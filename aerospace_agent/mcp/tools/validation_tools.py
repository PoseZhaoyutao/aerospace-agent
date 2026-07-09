"""交叉验证工具 — 多引擎结果比对与精度评估。

第一性原理（K3 结果可信度）：
  1. 同一任务在多个引擎上的结果差异量化为位置/速度/事件时间误差
  2. 误差在阈值内 → confidence: high；超阈值 → confidence: low
  3. 支持两种模式：实时多引擎运行 / 比较已有结果
  4. 所有误差必须带单位，便于审计
  5. 时间序列比对：逐时刻对齐 + max/mean/RMS 统计
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import OrbitState
from ..adapters import get_all_adapters

#: 默认验证阈值
DEFAULT_THRESHOLDS = {
    "position_error_m": 1000.0,
    "velocity_error_mps": 0.1,
}


def cross_validate_results(task_spec: Dict,
                           engines: Optional[List[str]] = None,
                           existing_results: Optional[Dict[str, Dict]] = None,
                           thresholds: Optional[Dict[str, float]] = None,
                           ) -> Dict:
    """交叉验证——多引擎结果比对。

    三种模式：
      1. existing_results 为 None：在指定引擎上运行 task_spec 并比较
      2. existing_results 提供 {engine: result_dict}：直接比较多引擎已有结果
      3. existing_results 提供 {reference: ..., candidate: ...}：双源阈值验证

    Args:
        task_spec: 任务规格（含 initial_state, force_model, duration_s 等）
        engines: 参与验证的引擎列表（None 时自动选可用引擎）
        existing_results: 已有结果 {engine: result_dict} 或 {reference, candidate}
        thresholds: 阈值 {position_error_m, velocity_error_mps}
    Returns:
        {position_error_m, velocity_error_mps, event_time_error_s,
         difference_sources, confidence, per_engine_results, passed}
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS

    # 模式 3：双源阈值验证（reference vs candidate）
    if existing_results and "reference" in existing_results:
        return _compare_reference_candidate(existing_results, thresholds)

    # 模式 2：比较已有结果
    if existing_results:
        return _compare_existing(existing_results, thresholds)

    # 模式 1：运行多引擎
    if engines is None:
        engines = _select_engines()

    per_engine: Dict[str, Dict] = {}
    # 并行运行多引擎验证
    import concurrent.futures
    max_workers = min(4, len(engines)) if engines else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_run_single_engine, eng, task_spec): eng
            for eng in engines
        }
        for future in concurrent.futures.as_completed(future_map, timeout=120):
            eng = future_map[future]
            try:
                per_engine[eng] = future.result(timeout=120)
            except Exception as exc:
                per_engine[eng] = {
                    "engine": eng, "status": "error",
                    "reason": str(exc), "final_state": None,
                }

    return _compare_existing(per_engine, thresholds)


def _select_engines() -> List[str]:
    """自动选择可用的传播引擎（至少 2 个才能交叉验证）。"""
    adapters = get_all_adapters()
    capable = [e for e, a in adapters.items()
               if a.is_available() and "propagate_orbit" in a.capabilities()]
    # 确保至少包含内置引擎作为基准
    if "builtin" not in capable:
        capable.append("builtin")
    return capable[:4]  # 限制最多 4 个


def _run_single_engine(engine: str, task_spec: Dict) -> Dict:
    """在单个引擎上运行任务。"""
    try:
        initial_state = task_spec.get("initial_state", {})
        force_model = task_spec.get("force_model", {})
        duration_s = task_spec.get("duration_s", 86400.0)

        if engine == "builtin":
            from .propagation_tools import propagate_orbit
            result = propagate_orbit(
                initial_state, force_model, duration_s,
                engine="builtin",
            )
        else:
            from .propagation_tools import propagate_orbit
            result = propagate_orbit(
                initial_state, force_model, duration_s,
                engine=engine,
            )

        # 提取终态和完整历史
        history = result.get("state_history", [])
        final_state = history[-1] if history else None

        return {
            "engine": engine,
            "status": result.get("status", "success"),
            "final_state": final_state,
            "state_history": history,
            "state_count": len(history),
            "engine_version": result.get("metadata", {}).get(
                "engine_version", "unknown"),
        }
    except Exception as exc:
        return {
            "engine": engine,
            "status": "error",
            "reason": str(exc),
            "final_state": None,
        }


def _compare_existing(results: Dict[str, Dict],
                      thresholds: Optional[Dict[str, float]] = None) -> Dict:
    """比较多引擎已有结果——优先使用时间序列比对。"""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    # 筛选出有效结果
    valid = {e: r for e, r in results.items()
             if r.get("status") not in ("error", "unavailable")
             and r.get("final_state") is not None}

    if len(valid) < 2:
        return _insufficient_results(results, valid)

    engines = list(valid.keys())

    # 优先：时间序列比对（当至少一对引擎有 state_history 时）
    has_history = any(
        len(r.get("state_history", [])) > 1 for r in valid.values())
    if has_history:
        return _compare_histories(valid, engines, thresholds)

    # 回退：终态比对
    return _compare_final_states(valid, engines, thresholds)


def _compare_histories(valid: Dict[str, Dict], engines: List[str],
                       thresholds: Dict[str, float]) -> Dict:
    """时间序列比对——逐时刻对齐 + max/mean/RMS 统计。"""
    all_pos_errors: List[float] = []
    all_vel_errors: List[float] = []
    differences: List[Dict] = []

    for i in range(len(engines)):
        for j in range(i + 1, len(engines)):
            e1, e2 = engines[i], engines[j]
            hist1 = valid[e1].get("state_history", [])
            hist2 = valid[e2].get("state_history", [])

            # 如果任一方无历史，回退到终态比对
            if len(hist1) <= 1 or len(hist2) <= 1:
                s1 = valid[e1].get("final_state", {})
                s2 = valid[e2].get("final_state", {})
                pos_err = _vector_diff_norm(
                    s1.get("position_m"), s2.get("position_m"))
                vel_err = _vector_diff_norm(
                    s1.get("velocity_mps"), s2.get("velocity_mps"))
            else:
                stats = compare_state_histories(hist1, hist2)
                pos_err = stats["max_position_error_m"]
                vel_err = stats["max_velocity_error_mps"]
                all_pos_errors.extend(stats["per_step_position_errors_m"])
                all_vel_errors.extend(stats["per_step_velocity_errors_mps"])

            differences.append({
                "engine_a": e1, "engine_b": e2,
                "position_error_m": round(pos_err, 6),
                "velocity_error_mps": round(vel_err, 9),
            })

    max_pos_err = max(d["position_error_m"] for d in differences)
    max_vel_err = max(d["velocity_error_mps"] for d in differences)

    # RMS 统计（如果有逐时刻数据）
    rms_pos = (math.sqrt(sum(e ** 2 for e in all_pos_errors) / len(all_pos_errors))
               if all_pos_errors else max_pos_err)
    rms_vel = (math.sqrt(sum(e ** 2 for e in all_vel_errors) / len(all_vel_errors))
               if all_vel_errors else max_vel_err)

    confidence = _assess_confidence(max_pos_err, max_vel_err)
    pos_threshold = thresholds.get("position_error_m", 1000.0)
    vel_threshold = thresholds.get("velocity_error_mps", 0.1)
    passed = max_pos_err <= pos_threshold and max_vel_err <= vel_threshold

    return {
        "position_error_m": round(max_pos_err, 6),
        "velocity_error_mps": round(max_vel_err, 9),
        "rms_position_error_m": round(rms_pos, 6),
        "rms_velocity_error_mps": round(rms_vel, 9),
        "event_time_error_s": None,
        "difference_sources": differences,
        "confidence": confidence,
        "passed": passed,
        "thresholds": thresholds,
        "per_engine_results": {
            e: {
                "status": r.get("status"),
                "engine_version": r.get("engine_version", "unknown"),
                "state_count": r.get("state_count", 0),
            }
            for e, r in valid.items()
        },
        "engines_compared": engines,
        "units": "SI (m, m/s, s)",
        "summary": (
            f"{len(engines)} 个引擎时间序列交叉验证完成，"
            f"最大位置误差 {max_pos_err:.3f} m (RMS {rms_pos:.3f} m)，"
            f"最大速度误差 {max_vel_err:.6f} m/s (RMS {rms_vel:.6f} m/s)，"
            f"可信度: {confidence}，{'通过' if passed else '未通过'}"
        ),
    }


def _compare_final_states(valid: Dict[str, Dict], engines: List[str],
                          thresholds: Dict[str, float]) -> Dict:
    """终态比对（无时间序列时的回退）。"""
    position_errors: List[float] = []
    velocity_errors: List[float] = []
    differences: List[Dict] = []

    for i in range(len(engines)):
        for j in range(i + 1, len(engines)):
            e1, e2 = engines[i], engines[j]
            s1 = valid[e1]["final_state"]
            s2 = valid[e2]["final_state"]
            pos_err = _vector_diff_norm(
                s1.get("position_m"), s2.get("position_m"))
            vel_err = _vector_diff_norm(
                s1.get("velocity_mps"), s2.get("velocity_mps"))
            position_errors.append(pos_err)
            velocity_errors.append(vel_err)
            differences.append({
                "engine_a": e1, "engine_b": e2,
                "position_error_m": round(pos_err, 6),
                "velocity_error_mps": round(vel_err, 9),
            })

    max_pos_err = max(position_errors) if position_errors else 0.0
    max_vel_err = max(velocity_errors) if velocity_errors else 0.0
    confidence = _assess_confidence(max_pos_err, max_vel_err)
    pos_threshold = thresholds.get("position_error_m", 1000.0)
    vel_threshold = thresholds.get("velocity_error_mps", 0.1)
    passed = max_pos_err <= pos_threshold and max_vel_err <= vel_threshold

    return {
        "position_error_m": round(max_pos_err, 6),
        "velocity_error_mps": round(max_vel_err, 9),
        "rms_position_error_m": round(max_pos_err, 6),
        "rms_velocity_error_mps": round(max_vel_err, 9),
        "event_time_error_s": None,
        "difference_sources": differences,
        "confidence": confidence,
        "passed": passed,
        "thresholds": thresholds,
        "per_engine_results": {
            e: {
                "status": r.get("status"),
                "engine_version": r.get("engine_version", "unknown"),
                "state_count": r.get("state_count", 0),
            }
            for e, r in valid.items()
        },
        "engines_compared": engines,
        "units": "SI (m, m/s, s)",
        "summary": (
            f"{len(engines)} 个引擎终态交叉验证完成，"
            f"最大位置误差 {max_pos_err:.3f} m，"
            f"最大速度误差 {max_vel_err:.6f} m/s，"
            f"可信度: {confidence}，{'通过' if passed else '未通过'}"
        ),
    }


def _compare_reference_candidate(data: Dict[str, Any],
                                 thresholds: Dict[str, float]) -> Dict:
    """双源阈值验证——reference vs candidate。

    data 格式:
      {"reference": state_history_or_result,
       "candidate": state_history_or_result}
    其中 state_history_or_result 可以是:
      - list[dict] (state history)
      - dict with "state_history" key
      - dict with "final_state" key
    """
    ref = data.get("reference")
    cand = data.get("candidate")
    if ref is None or cand is None:
        return {
            "position_error_m": None,
            "velocity_error_mps": None,
            "confidence": "low",
            "passed": False,
            "summary": "缺少 reference 或 candidate 数据",
        }

    # 统一提取 state_history 或 final_state
    ref_hist = _extract_history(ref)
    cand_hist = _extract_history(cand)

    if len(ref_hist) > 1 and len(cand_hist) > 1:
        stats = compare_state_histories(ref_hist, cand_hist)
        pos_err = stats["max_position_error_m"]
        vel_err = stats["max_velocity_error_mps"]
        rms_pos = stats["rms_position_error_m"]
        rms_vel = stats["rms_velocity_error_mps"]
        aligned_count = stats["aligned_steps"]
    else:
        # 终态比对
        ref_final = ref_hist[-1] if ref_hist else {}
        cand_final = cand_hist[-1] if cand_hist else {}
        pos_err = _vector_diff_norm(
            ref_final.get("position_m"), cand_final.get("position_m"))
        vel_err = _vector_diff_norm(
            ref_final.get("velocity_mps"), cand_final.get("velocity_mps"))
        rms_pos = pos_err
        rms_vel = vel_err
        aligned_count = 1

    confidence = _assess_confidence(pos_err, vel_err)
    pos_threshold = thresholds.get("position_error_m", 1000.0)
    vel_threshold = thresholds.get("velocity_error_mps", 0.1)
    passed = pos_err <= pos_threshold and vel_err <= vel_threshold

    return {
        "position_error_m": round(pos_err, 6),
        "velocity_error_mps": round(vel_err, 9),
        "rms_position_error_m": round(rms_pos, 6),
        "rms_velocity_error_mps": round(rms_vel, 9),
        "event_time_error_s": None,
        "difference_sources": [{
            "reference": "reference",
            "candidate": "candidate",
            "position_error_m": round(pos_err, 6),
            "velocity_error_mps": round(vel_err, 9),
            "aligned_steps": aligned_count,
        }],
        "confidence": confidence,
        "passed": passed,
        "thresholds": thresholds,
        "units": "SI (m, m/s, s)",
        "summary": (
            f"双源验证完成 (对齐 {aligned_count} 步)，"
            f"最大位置误差 {pos_err:.3f} m (阈值 {pos_threshold} m)，"
            f"最大速度误差 {vel_err:.6f} m/s (阈值 {vel_threshold} m/s)，"
            f"可信度: {confidence}，{'通过' if passed else '未通过'}"
        ),
    }


def _extract_history(data: Any) -> List[Dict]:
    """从各种格式提取 state_history。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        hist = data.get("state_history", [])
        if hist:
            return hist
        final = data.get("final_state")
        if final:
            return [final]
        # 如果 data 本身就是单个状态
        if "position_m" in data:
            return [data]
    return []


def compare_state_histories(reference: List[Dict],
                            candidate: List[Dict],
                            time_tolerance_s: float = 1.0,
                            ) -> Dict[str, Any]:
    """时间序列比对——逐时刻对齐 + max/mean/RMS 统计。

    Args:
        reference: 参考状态历史 [{position_m, velocity_mps, elapsed_s/time_s}, ...]
        candidate: 待验证状态历史
        time_tolerance_s: 时间对齐容差 (s)
    Returns:
        {max_position_error_m, max_velocity_error_mps,
         mean_position_error_m, mean_velocity_error_mps,
         rms_position_error_m, rms_velocity_error_mps,
         aligned_steps, per_step_position_errors_m, per_step_velocity_errors_mps}
    """
    # 构建时间→状态映射
    ref_map = _build_time_map(reference)
    cand_map = _build_time_map(candidate)

    # 对齐时间戳
    aligned_times = _align_timestamps(
        list(ref_map.keys()), list(cand_map.keys()), time_tolerance_s)

    pos_errors: List[float] = []
    vel_errors: List[float] = []

    for t_ref, t_cand in aligned_times:
        s_ref = ref_map[t_ref]
        s_cand = cand_map[t_cand]
        pe = _vector_diff_norm(
            s_ref.get("position_m"), s_cand.get("position_m"))
        ve = _vector_diff_norm(
            s_ref.get("velocity_mps"), s_cand.get("velocity_mps"))
        pos_errors.append(pe)
        vel_errors.append(ve)

    if not pos_errors:
        return {
            "max_position_error_m": float("inf"),
            "max_velocity_error_mps": float("inf"),
            "mean_position_error_m": float("inf"),
            "mean_velocity_error_mps": float("inf"),
            "rms_position_error_m": float("inf"),
            "rms_velocity_error_mps": float("inf"),
            "aligned_steps": 0,
            "per_step_position_errors_m": [],
            "per_step_velocity_errors_mps": [],
        }

    max_pos = max(pos_errors)
    max_vel = max(vel_errors)
    mean_pos = sum(pos_errors) / len(pos_errors)
    mean_vel = sum(vel_errors) / len(vel_errors)
    rms_pos = math.sqrt(sum(e ** 2 for e in pos_errors) / len(pos_errors))
    rms_vel = math.sqrt(sum(e ** 2 for e in vel_errors) / len(vel_errors))

    return {
        "max_position_error_m": round(max_pos, 6),
        "max_velocity_error_mps": round(max_vel, 9),
        "mean_position_error_m": round(mean_pos, 6),
        "mean_velocity_error_mps": round(mean_vel, 9),
        "rms_position_error_m": round(rms_pos, 6),
        "rms_velocity_error_mps": round(rms_vel, 9),
        "aligned_steps": len(aligned_times),
        "per_step_position_errors_m": [round(e, 6) for e in pos_errors],
        "per_step_velocity_errors_mps": [round(e, 9) for e in vel_errors],
    }


def _build_time_map(history: List[Dict]) -> Dict[float, Dict]:
    """从状态历史构建 时间→状态 映射。"""
    result = {}
    for state in history:
        t = state.get("elapsed_s", state.get("time_s", 0.0))
        result[float(t)] = state
    return result


def _align_timestamps(ref_times: List[float], cand_times: List[float],
                      tolerance_s: float) -> List[Tuple[float, float]]:
    """对齐两组时间戳——最近邻匹配。"""
    aligned = []
    cand_sorted = sorted(cand_times)
    for t_ref in sorted(ref_times):
        # 找候选中最接近的时间
        best_t = min(cand_sorted, key=lambda tc: abs(tc - t_ref))
        if abs(best_t - t_ref) <= tolerance_s:
            aligned.append((t_ref, best_t))
    return aligned


def _vector_diff_norm(v1: Optional[List], v2: Optional[List]) -> float:
    """计算两个向量的差的模。"""
    if v1 is None or v2 is None or len(v1) != 3 or len(v2) != 3:
        return float("inf")
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(3)))


def _assess_confidence(pos_err: float, vel_err: float) -> str:
    """基于误差评估可信度。"""
    if pos_err < 1.0 and vel_err < 0.001:
        return "high"
    if pos_err < 100.0 and vel_err < 0.1:
        return "medium"
    return "low"


def _insufficient_results(results: Dict, valid: Dict) -> Dict:
    """有效结果不足时的结构化返回。"""
    return {
        "position_error_m": None,
        "velocity_error_mps": None,
        "rms_position_error_m": None,
        "rms_velocity_error_mps": None,
        "event_time_error_s": None,
        "difference_sources": [],
        "confidence": "low",
        "passed": False,
        "thresholds": DEFAULT_THRESHOLDS,
        "per_engine_results": {
            e: {"status": r.get("status", "unknown"),
                "reason": r.get("reason", "")}
            for e, r in results.items()
        },
        "engines_compared": list(valid.keys()),
        "units": "SI (m, m/s, s)",
        "summary": (
            f"仅 {len(valid)} 个引擎产生有效结果，"
            "无法进行交叉验证（需至少 2 个）。"
            "建议安装更多引擎或检查引擎配置。"
        ),
    }


__all__ = ["cross_validate_results", "compare_state_histories"]
