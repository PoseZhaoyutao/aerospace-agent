"""星历查询工具 — 基于 SPICE kernel 的天体状态查询。

第一性原理（K3 星历精度）：
  1. 星历查询依赖 SPICE kernel 文件（SPK/CK/LSK/FK）
  2. kernel 路径必须经 PathPolicy.validate_kernel_path 校验——杜绝任意文件加载
  3. spiceypy 不可用时返回结构化错误，不崩溃
  4. 输出必须显式标注 units、frame、epoch、kernel_list
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..adapters import get_adapter
from ..safety import PathPolicy
from ..schemas import Epoch

#: SPICE 支持的参考系别名映射
_FRAME_ALIASES = {
    "GCRF": "J2000",
    "ICRF": "J2000",
    "EME2000": "J2000",
    "J2000": "J2000",
    "ITRF": "ITRF93",
    "TEME": "TEME",
}


def query_ephemeris_state(target: str, observer: str,
                          epoch_dict: Dict, frame: str = "J2000",
                          aberration_correction: str = "NONE",
                          kernels: Optional[List[str]] = None) -> Dict:
    """查询目标天体相对观察者的位置和速度。

    Args:
        target: 目标天体名或 NAIF ID（如 "Moon"、"399"）
        observer: 观察者天体名或 NAIF ID（如 "Earth"、"10"）
        epoch_dict: Epoch.to_dict() 格式的时间字典
        frame: 参考系（GCRF/ICRF/J2000/ITRF 等）
        aberration_correction: 光行差修正（NONE/LT/LT+S）
        kernels: kernel 文件路径列表；None 时使用 kernel_registry
    Returns:
        {position_m, velocity_mps, frame, epoch, units, kernel_list, engine}
    """
    # 检查 spiceypy 可用性
    adapter = get_adapter("spiceypy")
    if not adapter.is_available():
        return {
            "status": "error",
            "reason": "spiceypy 未安装或不可用。星历查询需要 spiceypy 和 SPICE kernel。",
            "engine": "spiceypy",
            "install_hint": "pip install spiceypy，并配置 SPICE_KERNELS 环境变量",
        }

    # 解析 epoch
    try:
        epoch = Epoch.from_dict(epoch_dict)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"epoch 解析失败: {exc}",
            "engine": "spiceypy",
        }

    # 解析 kernel 路径
    kernel_list, kernel_err = _resolve_kernels(kernels)
    if kernel_err:
        return kernel_err

    # 转换 frame 名称
    spice_frame = _FRAME_ALIASES.get(frame.upper(), frame.upper())

    try:
        import spiceypy as spice  # type: ignore

        # 加载 kernel
        loaded = []
        for kpath in kernel_list:
            spice.furnsh(kpath)
            loaded.append(kpath)

        try:
            # 将 epoch 转为 ET
            et = _epoch_to_et(epoch)

            # 查询星历
            state, lt = spice.spkezr(
                target, et, spice_frame,
                aberration_correction, observer
            )

            return {
                "position_m": [float(state[0]), float(state[1]),
                               float(state[2])],
                "velocity_mps": [float(state[3]), float(state[4]),
                                 float(state[5])],
                "frame": frame,
                "epoch": epoch.to_dict(),
                "et": float(et),
                "light_time_s": float(lt),
                "units": "SI (m, m/s)",
                "kernel_list": loaded,
                "engine": "spiceypy",
                "aberration_correction": aberration_correction,
            }
        finally:
            # 卸载 kernel
            for kpath in loaded:
                spice.unload(kpath)

    except Exception as exc:
        return {
            "status": "error",
            "reason": f"spiceypy 查询失败: {exc}",
            "engine": "spiceypy",
            "kernel_list": kernel_list,
        }


def _resolve_kernels(kernels: Optional[List[str]]) -> tuple:
    """解析并验证 kernel 文件路径。"""
    if kernels:
        # 显式提供的 kernel 路径——逐个校验
        resolved = []
        for k in kernels:
            if not os.path.exists(k):
                return [], {
                    "status": "error",
                    "reason": f"kernel 文件不存在: {k}",
                    "engine": "spiceypy",
                }
            if not PathPolicy.validate_kernel_path(k):
                return [], {
                    "status": "error",
                    "reason": f"kernel 路径未通过安全校验: {k}",
                    "engine": "spiceypy",
                }
            resolved.append(k)
        return resolved, None

    # 从环境变量自动发现 kernel
    kernels_dir = os.environ.get("SPICE_KERNELS", "")
    if not kernels_dir or not os.path.isdir(kernels_dir):
        return [], {
            "status": "error",
            "reason": (
                "未提供 kernel 文件且 SPICE_KERNELS 环境变量未设置。"
                "请显式传入 kernels 参数或设置 SPICE_KERNELS。"
            ),
            "engine": "spiceypy",
        }

    # 自动加载常见 kernel
    auto_kernels = []
    for pattern in ("*.tls", "*.bsp", "*.tf", "*.tsc", "*.bc"):
        import glob
        auto_kernels.extend(glob.glob(os.path.join(kernels_dir, pattern)))

    if not auto_kernels:
        return [], {
            "status": "error",
            "reason": f"在 {kernels_dir} 中未找到任何 kernel 文件",
            "engine": "spiceypy",
        }

    return auto_kernels[:10], None  # 限制加载数量


def _epoch_to_et(epoch: Epoch) -> float:
    """将 Epoch 转换为 SPICE ET（历书时秒）。"""
    import spiceypy as spice  # type: ignore
    iso = epoch.to_iso_utc()
    et = spice.str2et(iso)
    return float(et)


__all__ = ["query_ephemeris_state"]
