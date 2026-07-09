"""Frame — 统一坐标系表示。

第一性原理：坐标值没有坐标系标签就是无意义的数字。
每个状态都必须绑定 (frame_name, center, realization)。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FrameName(str, Enum):
    GCRF = "GCRF"           # 地心天球参考系（ICRS 对齐）
    ICRF = "ICRF"           # 国际天球参考系
    EME2000 = "EME2000"     # J2000 地心赤道惯性系
    J2000 = "J2000"         # 别名
    ITRF = "ITRF"           # 国际地球参考系（固连）
    TEME = "TEME"           # 真赤道平春分点（SGP4 用）
    LVLH = "LVLH"           # 局部水平局部垂直
    BODY_FIXED = "BodyFixed"
    RSW = "RSW"             # 沿迹坐标系
    TNW = "TNW"             # 法向坐标系


class FrameCenter(str, Enum):
    EARTH = "Earth"
    MOON = "Moon"
    MARS = "Mars"
    SUN = "Sun"
    SSB = "SSB"  # 太阳系质心
    MERCURY = "Mercury"
    VENUS = "Venus"
    JUPITER = "Jupiter"
    SATURN = "Saturn"


@dataclass
class Frame:
    """统一坐标系表示。

    Attributes:
        name: 坐标系名称
        center: 中心天体
        realization: 实现标准（IERS2010 等）
    """
    name: FrameName = FrameName.GCRF
    center: FrameCenter = FrameCenter.EARTH
    realization: str = "IERS2010"

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = FrameName(self.name)
        if isinstance(self.center, str):
            self.center = FrameCenter(self.center)

    def to_dict(self) -> dict:
        return {
            "name": self.name.value,
            "center": self.center.value,
            "realization": self.realization,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(
            name=FrameName(d.get("name", "GCRF")),
            center=FrameCenter(d.get("center", "Earth")),
            realization=d.get("realization", "IERS2010"),
        )

    @property
    def is_inertial(self) -> bool:
        """是否为惯性系。"""
        return self.name in (
            FrameName.GCRF, FrameName.ICRF, FrameName.EME2000, FrameName.J2000
        )

    @property
    def is_body_fixed(self) -> bool:
        """是否为固连系。"""
        return self.name in (FrameName.ITRF, FrameName.BODY_FIXED)
