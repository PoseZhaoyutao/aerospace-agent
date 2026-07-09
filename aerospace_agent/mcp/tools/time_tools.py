"""时间转换工具 — 跨时间尺度与格式的统一转换。

第一性原理（K3 时间正确性）：
  1. 时间不是字符串，是 (value, scale, format) 三元组
  2. 闰秒和相对论修正在 UTC↔TDB 之间可达 km 级误差
  3. 引擎优先级：astropy（高精度）> spiceypy ET（需 kernel）> datetime（仅 UTC）
  4. 输出必须显式标注 time_scale 和 format
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

from ..adapters import get_adapter

#: 支持的时间尺度
_VALID_SCALES = {"UTC", "TAI", "TT", "TDB", "ET"}

#: 支持的格式
_VALID_FORMATS = {"ISO", "JD", "MJD", "UNIX"}


def convert_time(value, from_scale: str = "UTC", from_format: str = "ISO",
                 to_scale: str = "UTC", to_format: str = "ISO") -> Dict:
    """跨时间尺度与格式转换。

    Args:
        value: 输入时间值（ISO 字符串或浮点数）
        from_scale: 源时间尺度（UTC/TAI/TT/TDB/ET）
        from_format: 源格式（ISO/JD/MJD/UNIX）
        to_scale: 目标时间尺度
        to_format: 目标格式
    Returns:
        {input, output, engine_used, notes} 字典
    """
    from_scale = from_scale.upper().strip()
    to_scale = to_scale.upper().strip()
    from_format = from_format.upper().strip()
    to_format = to_format.upper().strip()

    # 参数校验
    for label, val, valid in [
        ("from_scale", from_scale, _VALID_SCALES),
        ("to_scale", to_scale, _VALID_SCALES),
        ("from_format", from_format, _VALID_FORMATS),
        ("to_format", to_format, _VALID_FORMATS),
    ]:
        if val not in valid:
            return _error(from_scale, from_format,
                          f"无效 {label}: '{val}'，支持: {sorted(valid)}")

    # 尝试 astropy
    result = _try_astropy(value, from_scale, from_format,
                          to_scale, to_format)
    if result:
        return result

    # 尝试 spiceypy ET
    result = _try_spiceypy(value, from_scale, from_format,
                           to_scale, to_format)
    if result:
        return result

    # 回退到 datetime（仅支持 UTC）
    return _fallback_datetime(value, from_scale, from_format,
                              to_scale, to_format)


def _try_astropy(value, fs, ff, ts, tf) -> Optional[Dict]:
    """优先使用 astropy.time.Time 进行高精度转换。"""
    try:
        from astropy.time import Time  # type: ignore
        t = _build_astropy_time(value, fs, ff)
        if t is None:
            return None
        t_out = getattr(t, ts.lower()) if ts != "ET" else t.tdb
        out_val = _format_astropy(t_out, tf)
        return {
            "input": {"value": value, "scale": fs, "format": ff},
            "output": {"value": out_val, "scale": ts, "format": tf},
            "engine_used": "astropy",
            "notes": "使用 astropy.time.Time 高精度转换（含闰秒与相对论修正）",
        }
    except Exception:
        return None


def _build_astropy_time(value, scale, fmt):
    from astropy.time import Time  # type: ignore
    astropy_fmt = {"ISO": "isot", "JD": "jd", "MJD": "mjd", "UNIX": "unix"}
    f = astropy_fmt.get(fmt)
    if f is None:
        return None
    astropy_scale = "utc" if scale == "UTC" else scale.lower()
    if scale == "ET":
        astropy_scale = "tdb"
    return Time(value, scale=astropy_scale, format=f)


def _format_astropy(t, fmt) -> object:
    if fmt == "ISO":
        return t.isot
    if fmt == "JD":
        return float(t.jd)
    if fmt == "MJD":
        return float(t.mjd)
    if fmt == "UNIX":
        return float(t.unix)
    return str(t)


def _try_spiceypy(value, fs, ff, ts, tf) -> Optional[Dict]:
    """当 astropy 不可用时，尝试 spiceypy ET 转换（需 kernel）。"""
    adapter = get_adapter("spiceypy")
    if not adapter.is_available():
        return None
    try:
        import spiceypy as spice  # type: ignore
        # ET 仅在 spice 中有定义；此路径仅处理涉及 ET 的转换
        if "ET" not in (fs, ts):
            return None
        # 需要 kernel 才能做历元转换，此处简化
        return None
    except Exception:
        return None


def _fallback_datetime(value, fs, ff, ts, tf) -> Dict:
    """最终回退：使用 datetime 仅处理 UTC 场景。"""
    if fs != "UTC" or ts != "UTC":
        return _error(fs, ff,
                      "astropy/spiceypy 均不可用，datetime 仅支持 UTC↔UTC 转换。"
                      "请安装 astropy（pip install astropy）以获得完整时间尺度支持。")
    try:
        dt = _parse_utc(value, ff)
        out_val = _format_utc(dt, tf)
        return {
            "input": {"value": value, "scale": "UTC", "format": ff},
            "output": {"value": out_val, "scale": "UTC", "format": tf},
            "engine_used": "datetime_builtin",
            "notes": "使用 Python 内置 datetime 近似转换（忽略闰秒，仅 UTC）",
        }
    except Exception as exc:
        return _error(fs, ff, f"datetime 解析失败: {exc}")


def _parse_utc(value, fmt) -> datetime:
    if fmt == "ISO":
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if fmt == "UNIX":
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if fmt == "JD":
        return datetime(2000, 1, 1, tzinfo=timezone.utc) + \
            timedelta(days=float(value) - 2451544.5)
    if fmt == "MJD":
        return datetime(1858, 11, 17, tzinfo=timezone.utc) + \
            timedelta(days=float(value))
    raise ValueError(f"不支持的格式: {fmt}")


def _format_utc(dt: datetime, fmt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    if fmt == "ISO":
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    if fmt == "UNIX":
        return dt.timestamp()
    if fmt == "JD":
        return dt.toordinal() + 1721424.5 + \
            (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    if fmt == "MJD":
        return dt.toordinal() - 678576 + \
            (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    raise ValueError(f"不支持的格式: {fmt}")


def _error(fs, ff, reason) -> Dict:
    return {
        "input": {"value": None, "scale": fs, "format": ff},
        "output": None,
        "engine_used": None,
        "notes": "",
        "status": "error",
        "reason": reason,
    }


__all__ = ["convert_time"]
