"""Canonical Astrodynamics Model — 统一航天动力学中间层 Schema。

所有引擎（Orekit/GMAT/SpiceyPy/Astropy/poliastro/Basilisk/STK）的输入输出
都必须经过此层转换，保证：
  1. 物理量 SI 单位显式标注（position_m / velocity_mps / mu 等）
  2. 每个状态都带 epoch + frame 标签，杜绝裸数字
  3. 可序列化为 JSON（MCP 传输）
  4. canonical → engine → canonical 无损往返

模块导出：
    Epoch, Frame, Body, OrbitState, AttitudeState,
    ForceModel, PropagatorConfig, GroundStation, SpacecraftConfig,
    WorkflowSpec, WorkflowResult, ValidationReport
"""
from .time import Epoch, TimeScale, TimeFormat
from .frame import Frame, FrameName, FrameCenter
from .body import Body
from .orbit import OrbitState, OrbitRepresentation, KeplerianElements
from .attitude import AttitudeState, AttitudeRepresentation
from .force_model import ForceModel, DragModel, SRPModel, ThirdBody
from .propagator import PropagatorConfig, PropagatorType, IntegratorType
from .ground_station import GroundStation
from .spacecraft import SpacecraftConfig
from .workflow import (
    WorkflowSpec, WorkflowResult, ValidationReport,
    WorkflowStep, LoopLedgerEntry, LoopPhase,
)

__all__ = [
    # time
    "Epoch", "TimeScale", "TimeFormat",
    # frame
    "Frame", "FrameName", "FrameCenter",
    # body
    "Body",
    # orbit
    "OrbitState", "OrbitRepresentation", "KeplerianElements",
    # attitude
    "AttitudeState", "AttitudeRepresentation",
    # force model
    "ForceModel", "DragModel", "SRPModel", "ThirdBody",
    # propagator
    "PropagatorConfig", "PropagatorType", "IntegratorType",
    # ground station & spacecraft
    "GroundStation", "SpacecraftConfig",
    # workflow
    "WorkflowSpec", "WorkflowResult", "ValidationReport",
    "WorkflowStep", "LoopLedgerEntry", "LoopPhase",
]
