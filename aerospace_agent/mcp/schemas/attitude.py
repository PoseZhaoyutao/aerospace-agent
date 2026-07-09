"""AttitudeState — 统一姿态表示。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .time import Epoch


class AttitudeRepresentation(str, Enum):
    QUATERNION = "quaternion"   # [q0, q1, q2, q3] 标量在前
    EULER = "euler"             # [roll, pitch, yaw] deg
    DCM = "dcm"                 # 3x3 方向余弦矩阵（行优先 9 元素）


@dataclass
class AttitudeState:
    """统一姿态状态。

    Attributes:
        epoch: 时间
        frame_from: 源系（通常 body）
        frame_to: 目标系（通常 GCRF）
        representation: 表示方式
        quaternion: [q0,q1,q2,q3]（representation=quaternion 时）
        euler_deg: [roll,pitch,yaw] deg（representation=euler 时）
        dcm: 3x3 行优先列表（representation=dcm 时）
        angular_velocity_radps: [wx,wy,wz] rad/s（可选）
    """
    epoch: Epoch = field(default_factory=lambda: Epoch("2026-01-01T00:00:00"))
    frame_from: str = "body"
    frame_to: str = "GCRF"
    representation: AttitudeRepresentation = AttitudeRepresentation.QUATERNION
    quaternion: Optional[List[float]] = None
    euler_deg: Optional[List[float]] = None
    dcm: Optional[List[float]] = None
    angular_velocity_radps: Optional[List[float]] = None

    def __post_init__(self):
        if isinstance(self.representation, str):
            self.representation = AttitudeRepresentation(self.representation)
        if isinstance(self.epoch, dict):
            self.epoch = Epoch.from_dict(self.epoch)

    def to_dict(self) -> dict:
        d = {
            "epoch": self.epoch.to_dict(),
            "frame_from": self.frame_from,
            "frame_to": self.frame_to,
            "representation": self.representation.value,
        }
        if self.quaternion is not None:
            d["quaternion"] = self.quaternion
        if self.euler_deg is not None:
            d["euler_deg"] = self.euler_deg
        if self.dcm is not None:
            d["dcm"] = self.dcm
        if self.angular_velocity_radps is not None:
            d["angular_velocity_radps"] = self.angular_velocity_radps
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AttitudeState":
        return cls(
            epoch=Epoch.from_dict(d.get("epoch", {})),
            frame_from=d.get("frame_from", "body"),
            frame_to=d.get("frame_to", "GCRF"),
            representation=AttitudeRepresentation(d.get("representation", "quaternion")),
            quaternion=d.get("quaternion"),
            euler_deg=d.get("euler_deg"),
            dcm=d.get("dcm"),
            angular_velocity_radps=d.get("angular_velocity_radps"),
        )
