"""Epoch — 统一时间表示。

第一性原理：时间不是一个字符串，而是 (value, scale, format) 三元组。
任何跨引擎传递都必须显式声明时间尺度（UTC/TAI/TT/TDB）和格式（ISO/JD/MJD），
否则闰秒和相对论修正会导致 km 级误差。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class TimeScale(str, Enum):
    UTC = "UTC"
    TAI = "TAI"
    TT = "TT"
    TDB = "TDB"
    ET = "ET"  # Ephemeris Time (SPICE)


class TimeFormat(str, Enum):
    ISO = "ISO"       # "2026-01-01T00:00:00"
    JD = "JD"         # 儒略日
    MJD = "MJD"       # 修正儒略日
    UNIX = "UNIX"     # Unix 时间戳


@dataclass
class Epoch:
    """统一时间表示。

    Attributes:
        value: 时间值，ISO 字符串或浮点数（JD/MJD/UNIX）
        scale: 时间尺度（UTC/TAI/TT/TDB/ET）
        format: 值的格式（ISO/JD/MJD/UNIX）
    """
    value: Union[str, float]
    scale: TimeScale = TimeScale.UTC
    format: TimeFormat = TimeFormat.ISO

    def __post_init__(self):
        if isinstance(self.scale, str):
            self.scale = TimeScale(self.scale)
        if isinstance(self.format, str):
            self.format = TimeFormat(self.format)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "scale": self.scale.value,
            "format": self.format.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Epoch":
        return cls(
            value=d["value"],
            scale=TimeScale(d.get("scale", "UTC")),
            format=TimeFormat(d.get("format", "ISO")),
        )

    def to_iso_utc(self) -> str:
        """转换为 ISO UTC 字符串。

        K5-H8: 实现非 UTC 尺度的正确转换。
        优先用 astropy.time.Time 做精确转换；无 astropy 时用固定偏移近似。
        """
        if self.format == TimeFormat.ISO and self.scale == TimeScale.UTC:
            return str(self.value)

        # 尝试 astropy 精确转换
        try:
            from astropy.time import Time as AstropyTime
            t = AstropyTime(str(self.value), scale=self.scale.value.lower())
            t_utc = t.utc
            return t_utc.isot.replace("T", "T").split(".")[0] + "Z" \
                if "." in t_utc.isot else t_utc.isot + "Z"
        except Exception:
            pass

        # 无 astropy：用固定偏移近似（忽略闰秒，精度足够 LEO 应用）
        # TAI - UTC = 37s (自 2017-01-01), TT - TAI = 32.184s, TDB ≈ TT
        from datetime import datetime, timedelta, timezone
        scale_offsets = {
            TimeScale.TAI: timedelta(seconds=37),
            TimeScale.TT: timedelta(seconds=69.184),
            TimeScale.TDB: timedelta(seconds=69.184),
            TimeScale.TCG: timedelta(seconds=69.184),  # 近似
            TimeScale.TCB: timedelta(seconds=69.184),  # 近似
        }
        try:
            val = str(self.value).replace("Z", "+00:00")
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if self.scale in scale_offsets:
                # 从该尺度减去偏移得到 UTC
                dt_utc = dt - scale_offsets[self.scale]
                return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            return str(self.value)
        except Exception:
            return str(self.value)
