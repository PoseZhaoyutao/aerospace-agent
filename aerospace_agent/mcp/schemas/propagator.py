"""PropagatorConfig — 传播器配置。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PropagatorType(str, Enum):
    ANALYTICAL = "analytical"   # 解析（SGP4 / 二体）
    NUMERICAL = "numerical"     # 数值积分
    SGP4 = "sgp4"


class IntegratorType(str, Enum):
    DORMAND_PRINCE_853 = "DormandPrince853"
    RK4 = "RK4"
    RK45 = "RK45"
    GMAT_DEFAULT = "GMAT_Default"
    BULIRSCH_STOER = "BulirschStoer"


@dataclass
class PropagatorConfig:
    """传播器配置。

    Attributes:
        engine: 引擎（orekit/gmat/poliastro/stk/basilisk/auto）
        type: 传播类型
        integrator: 积分器
        step_s: 步长 s
        duration_s: 总传播时长 s
        tolerance: 容差（数值积分用）
        output_step_s: 输出采样间隔 s
    """
    engine: str = "auto"
    type: PropagatorType = PropagatorType.NUMERICAL
    integrator: IntegratorType = IntegratorType.DORMAND_PRINCE_853
    step_s: float = 60.0
    duration_s: float = 86400.0
    tolerance: float = 1e-10
    output_step_s: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = PropagatorType(self.type)
        if isinstance(self.integrator, str):
            self.integrator = IntegratorType(self.integrator)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "type": self.type.value,
            "integrator": self.integrator.value,
            "step_s": self.step_s,
            "duration_s": self.duration_s,
            "tolerance": self.tolerance,
            "output_step_s": self.output_step_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PropagatorConfig":
        return cls(
            engine=d.get("engine", "auto"),
            type=PropagatorType(d.get("type", "numerical")),
            integrator=IntegratorType(d.get("integrator", "DormandPrince853")),
            step_s=d.get("step_s", 60.0),
            duration_s=d.get("duration_s", 86400.0),
            tolerance=d.get("tolerance", 1e-10),
            output_step_s=d.get("output_step_s"),
        )
