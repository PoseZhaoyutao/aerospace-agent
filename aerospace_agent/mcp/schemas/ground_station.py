"""GroundStation — 地面站定义。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GroundStation:
    """地面站。

    Attributes:
        name: 站名
        latitude_deg: 纬度 deg（WGS84）
        longitude_deg: 经度 deg
        altitude_m: 海拔 m
        min_elevation_deg: 最小仰角 deg（可见性判定阈值）
    """
    name: str = "default_station"
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    altitude_m: float = 0.0
    min_elevation_deg: float = 5.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "GroundStation":
        return cls(**{k: d[k] for k in (
            "name", "latitude_deg", "longitude_deg", "altitude_m", "min_elevation_deg"
        ) if k in d})

    @classmethod
    def example(cls) -> "GroundStation":
        """示例：北京测控站。"""
        return cls(
            name="Beijing_Station",
            latitude_deg=40.0789, longitude_deg=116.5867,
            altitude_m=50.0, min_elevation_deg=5.0,
        )
