"""ForceModel — 统一力学模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DragModel:
    """大气阻力模型。

    Attributes:
        enabled: 是否启用
        atmosphere: 大气模型（NRLMSISE00/JB2008/exponential）
        cd: 阻力系数
        area_m2: 迎风面积 m²
        mass_kg: 质量 kg
    """
    enabled: bool = False
    atmosphere: str = "exponential"
    cd: float = 2.2
    area_m2: float = 1.0
    mass_kg: float = 100.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class SRPModel:
    """太阳光压模型。

    Attributes:
        enabled: 是否启用
        cr: 光压系数
        area_m2: 受光面积 m²
    """
    enabled: bool = False
    cr: float = 1.3
    area_m2: float = 1.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ThirdBody:
    """第三体摄动。"""
    name: str = "Moon"

    def to_dict(self) -> dict:
        return {"name": self.name}


@dataclass
class ForceModel:
    """统一力学模型。

    第一性原理：力的定义必须完整——中心天体 + 引力模型阶次 + 摄动项，
    否则传播结果的精度无从追溯。
    """
    central_body: str = "Earth"
    gravity: str = "point_mass"  # point_mass / spherical_harmonics
    degree: int = 0              # 球谐阶数
    order: int = 0               # 球谐次数
    drag: DragModel = field(default_factory=DragModel)
    srp: SRPModel = field(default_factory=SRPModel)
    third_body: List[ThirdBody] = field(default_factory=list)
    relativity: bool = False     # 广义相对论修正

    def to_dict(self) -> dict:
        return {
            "central_body": self.central_body,
            "gravity": self.gravity,
            "degree": self.degree,
            "order": self.order,
            "drag": self.drag.to_dict(),
            "srp": self.srp.to_dict(),
            "third_body": [t.to_dict() for t in self.third_body],
            "relativity": self.relativity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ForceModel":
        drag_d = d.get("drag", {})
        srp_d = d.get("srp", {})
        return cls(
            central_body=d.get("central_body", "Earth"),
            gravity=d.get("gravity", "point_mass"),
            degree=d.get("degree", 0),
            order=d.get("order", 0),
            drag=DragModel(**drag_d) if drag_d else DragModel(),
            srp=SRPModel(**srp_d) if srp_d else SRPModel(),
            third_body=[ThirdBody(**t) for t in d.get("third_body", [])],
            relativity=d.get("relativity", False),
        )

    @classmethod
    def two_body(cls) -> "ForceModel":
        """纯二体力学模型。"""
        return cls()

    @classmethod
    def high_fidelity(cls) -> "ForceModel":
        """高保真模型（球谐 + 阻力 + SRP + 三体）。"""
        return cls(
            gravity="spherical_harmonics", degree=70, order=70,
            drag=DragModel(enabled=True, atmosphere="NRLMSISE00"),
            srp=SRPModel(enabled=True),
            third_body=[ThirdBody("Sun"), ThirdBody("Moon")],
            relativity=True,
        )
