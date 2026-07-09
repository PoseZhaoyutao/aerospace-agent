"""OrbitState — 统一轨道状态表示。

第一性原理：轨道状态 = (epoch, frame, representation, data)。
同一物理状态可用 cartesian 或 keplerian 表示，但必须显式声明。
单位一律 SI：位置 m，速度 m/s，角度 deg（输入），内部转 rad。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

from .time import Epoch
from .frame import Frame


class OrbitRepresentation(str, Enum):
    CARTESIAN = "cartesian"
    KEPLERIAN = "keplerian"
    EQUINOCTIAL = "equinoctial"
    TLE = "tle"


@dataclass
class KeplerianElements:
    """开普勒轨道根数（角度输入用 deg，存储转 rad 便于计算）。

    Attributes:
        a_m: 半长轴 m
        e: 偏心率（无量纲）
        i_deg: 倾角 deg
        raan_deg: 升交点赤经 deg
        argp_deg: 近地点幅角 deg
        ta_deg: 真近点角 deg
    """
    a_m: float = 0.0
    e: float = 0.0
    i_deg: float = 0.0
    raan_deg: float = 0.0
    argp_deg: float = 0.0
    ta_deg: float = 0.0

    def to_dict(self) -> dict:
        return {
            "a_m": self.a_m, "e": self.e,
            "i_deg": self.i_deg, "raan_deg": self.raan_deg,
            "argp_deg": self.argp_deg, "ta_deg": self.ta_deg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeplerianElements":
        return cls(**{k: d[k] for k in (
            "a_m", "e", "i_deg", "raan_deg", "argp_deg", "ta_deg"
        ) if k in d})


@dataclass
class OrbitState:
    """统一轨道状态。

    representation == cartesian 时用 position_m / velocity_mps；
    representation == keplerian 时用 elements；
    两者可共存（转换后同时填充）。
    """
    epoch: Epoch = field(default_factory=lambda: Epoch("2026-01-01T00:00:00"))
    frame: Frame = field(default_factory=Frame)
    representation: OrbitRepresentation = OrbitRepresentation.CARTESIAN
    position_m: Optional[List[float]] = None      # [x, y, z] m
    velocity_mps: Optional[List[float]] = None     # [vx, vy, vz] m/s
    elements: Optional[KeplerianElements] = None

    def __post_init__(self):
        if isinstance(self.representation, str):
            self.representation = OrbitRepresentation(self.representation)
        if isinstance(self.epoch, dict):
            self.epoch = Epoch.from_dict(self.epoch)
        if isinstance(self.frame, dict):
            self.frame = Frame.from_dict(self.frame)

    def to_dict(self) -> dict:
        d = {
            "epoch": self.epoch.to_dict(),
            "frame": self.frame.to_dict(),
            "representation": self.representation.value,
        }
        if self.position_m is not None:
            d["position_m"] = self.position_m
        if self.velocity_mps is not None:
            d["velocity_mps"] = self.velocity_mps
        if self.elements is not None:
            d["elements"] = self.elements.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OrbitState":
        return cls(
            epoch=Epoch.from_dict(d["epoch"]),
            frame=Frame.from_dict(d.get("frame", {})),
            representation=OrbitRepresentation(d.get("representation", "cartesian")),
            position_m=d.get("position_m"),
            velocity_mps=d.get("velocity_mps"),
            elements=KeplerianElements.from_dict(d["elements"]) if "elements" in d else None,
        )

    # ------------------------------------------------------------------
    # 笛卡尔 ↔ 开普勒 互转（二体解析，SI 单位）
    # ------------------------------------------------------------------
    def to_keplerian(self, mu: float = 3.986004418e14) -> "OrbitState":
        """笛卡尔 → 开普勒（二体）。返回新 OrbitState，elements 已填充。

        修复记录:
        - 使用 Laplace-Runge-Lenz 向量 (v×h)/μ - r/|r| 替代不稳定的 (v²-μ/r)r-(r·v)v 公式
        - 所有 acos 调用前加 clipping 防止浮点误差触发 domain error
        - 增加 eps 边界保护,与 physics/kepler.py state_to_elements 对齐
        """
        if self.position_m is None or self.velocity_mps is None:
            raise ValueError("to_keplerian 需要 position_m 和 velocity_mps")
        if self.elements is not None and self.representation == OrbitRepresentation.KEPLERIAN:
            return self
        eps = 1e-12
        rx, ry, rz = self.position_m
        vx, vy, vz = self.velocity_mps
        r_mag = math.sqrt(rx*rx + ry*ry + rz*rz)
        v_mag = math.sqrt(vx*vx + vy*vy + vz*vz)

        # 角动量 h = r × v
        hx = ry*vz - rz*vy
        hy = rz*vx - rx*vz
        hz = rx*vy - ry*vx
        h = math.sqrt(hx*hx + hy*hy + hz*hz)

        # 节点向量 n = K × h = (-h_y, h_x, 0)
        nx = -hy
        ny = hx
        nn = math.sqrt(nx*nx + ny*ny)

        # 偏心率向量 e_vec = (v × h)/μ - r/|r|  (Laplace-Runge-Lenz, 数值稳定)
        vch_x = vy*hz - vz*hy
        vch_y = vz*hx - vx*hz
        vch_z = vx*hy - vy*hx
        ex = vch_x / mu - rx / r_mag
        ey = vch_y / mu - ry / r_mag
        ez = vch_z / mu - rz / r_mag
        e = math.sqrt(ex*ex + ey*ey + ez*ez)

        # 半长轴
        denom = 2.0/r_mag - v_mag*v_mag/mu
        a = 1.0 / denom if abs(denom) > eps else math.inf

        # 安全 acos: clip 到 [-1, 1] 防止浮点误差触发 domain error
        def _acos_safe(x):
            return math.acos(max(-1.0, min(1.0, x)))

        # 倾角
        i = _acos_safe(hz / h) if h > eps else 0.0
        # RAAN
        if nn > eps:
            raan = _acos_safe(nx / nn)
            if ny < 0:
                raan = 2*math.pi - raan
        else:
            raan = 0.0
        # 近地点幅角
        if nn > eps and e > eps:
            argp = _acos_safe((nx*ex + ny*ey) / (nn * e))
            if ez < 0:
                argp = 2*math.pi - argp
        else:
            argp = 0.0
        # 真近点角
        if e > eps:
            ta = _acos_safe((ex*rx + ey*ry + ez*rz) / (e * r_mag))
            if (rx*vx + ry*vy + rz*vz) < 0:
                ta = 2*math.pi - ta
        else:
            ta = 0.0

        elems = KeplerianElements(
            a_m=a, e=e,
            i_deg=math.degrees(i), raan_deg=math.degrees(raan),
            argp_deg=math.degrees(argp), ta_deg=math.degrees(ta),
        )
        return OrbitState(
            epoch=self.epoch, frame=self.frame,
            representation=OrbitRepresentation.KEPLERIAN,
            position_m=self.position_m, velocity_mps=self.velocity_mps,
            elements=elems,
        )

    def to_cartesian(self, mu: float = 3.986004418e14) -> "OrbitState":
        """开普勒 → 笛卡尔（二体）。返回新 OrbitState，position/velocity 已填充。"""
        if self.elements is None:
            raise ValueError("to_cartesian 需要 elements")
        if self.position_m is not None and self.representation == OrbitRepresentation.CARTESIAN:
            return self
        el = self.elements
        a = el.a_m
        e = el.e
        i = math.radians(el.i_deg)
        raan = math.radians(el.raan_deg)
        argp = math.radians(el.argp_deg)
        ta = math.radians(el.ta_deg)

        p = a * (1 - e*e)
        r_orb = p / (1 + e * math.cos(ta))  # 近焦点系位置
        x_pf = r_orb * math.cos(ta)
        y_pf = r_orb * math.sin(ta)
        # 近焦点 → 惯性系
        cos_O, sin_O = math.cos(raan), math.sin(raan)
        cos_w, sin_w = math.cos(argp), math.sin(argp)
        cos_i, sin_i = math.cos(i), math.sin(i)

        R11 = cos_O*cos_w - sin_O*sin_w*cos_i
        R12 = -cos_O*sin_w - sin_O*cos_w*cos_i
        R21 = sin_O*cos_w + cos_O*sin_w*cos_i
        R22 = -sin_O*sin_w + cos_O*cos_w*cos_i
        R31 = sin_w*sin_i
        R32 = cos_w*sin_i

        rx = R11*x_pf + R12*y_pf
        ry = R21*x_pf + R22*y_pf
        rz = R31*x_pf + R32*y_pf

        # 速度
        h = math.sqrt(mu * p)
        vx_pf = -mu/h * math.sin(ta)
        vy_pf = mu/h * (e + math.cos(ta))
        vx = R11*vx_pf + R12*vy_pf
        vy = R21*vx_pf + R22*vy_pf
        vz = R31*vx_pf + R32*vy_pf

        return OrbitState(
            epoch=self.epoch, frame=self.frame,
            representation=OrbitRepresentation.CARTESIAN,
            position_m=[rx, ry, rz], velocity_mps=[vx, vy, vz],
            elements=self.elements,
        )
