"""SpacecraftConfig — 航天器配置。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpacecraftConfig:
    """航天器配置（质量、面积、反射率等，用于力学模型与姿态仿真）。

    Attributes:
        name: 航天器名
        mass_kg: 质量 kg
        drag_area_m2: 阻力迎风面积 m²
        srp_area_m2: 光压受光面积 m²
        cd: 阻力系数
        cr: 光压系数
    """
    name: str = "default_sat"
    mass_kg: float = 100.0
    drag_area_m2: float = 1.0
    srp_area_m2: float = 1.0
    cd: float = 2.2
    cr: float = 1.3

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "SpacecraftConfig":
        return cls(**{k: d[k] for k in (
            "name", "mass_kg", "drag_area_m2", "srp_area_m2", "cd", "cr"
        ) if k in d})
