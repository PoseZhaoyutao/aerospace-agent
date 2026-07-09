"""Body — 天体物理参数。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Body:
    """天体定义。

    Attributes:
        name: 天体名（Earth/Moon/Mars...）
        mu: 引力参数 m³/s²
        radius: 平均半径 m
        gravity_model: 引力模型（point_mass/EGM96/EGM2008）
        rotation_rate_radps: 自转角速度 rad/s（固连系转换用）
    """
    name: str = "Earth"
    mu: float = 3.986004418e14  # m³/s²
    radius: float = 6378137.0   # m
    gravity_model: str = "point_mass"
    rotation_rate_radps: float = 7.2921159e-5  # Earth rad/s
    gm_sun: Optional[float] = 1.32712440018e20  # Sun mu m³/s²
    gm_moon: Optional[float] = 4.902800118e12   # Moon mu m³/s²

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mu": self.mu,
            "radius": self.radius,
            "gravity_model": self.gravity_model,
            "rotation_rate_radps": self.rotation_rate_radps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Body":
        return cls(
            name=d.get("name", "Earth"),
            mu=d.get("mu", 3.986004418e14),
            radius=d.get("radius", 6378137.0),
            gravity_model=d.get("gravity_model", "point_mass"),
            rotation_rate_radps=d.get("rotation_rate_radps", 7.2921159e-5),
        )

    @classmethod
    def earth(cls) -> "Body":
        return cls()

    @classmethod
    def moon(cls) -> "Body":
        return cls(
            name="Moon", mu=4.902800118e12, radius=1737400.0,
            gravity_model="point_mass", rotation_rate_radps=2.6617e-6,
        )

    @classmethod
    def mars(cls) -> "Body":
        return cls(
            name="Mars", mu=4.282837e13, radius=3389500.0,
            gravity_model="point_mass", rotation_rate_radps=7.0882e-5,
        )
