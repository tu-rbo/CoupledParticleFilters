"""Typed frame, measurement, and belief containers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class Frame:
    """One RGB-D observation; image coordinates are always ``(row, column)``."""

    timestamp_ns: int
    rgb: Tensor
    depth_m: Tensor
    intrinsics: Tensor
    flow_rc: Tensor

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError("rgb must have shape [height, width, 3]")
        if self.depth_m.shape != self.rgb.shape[:2]:
            raise ValueError("depth_m must match the RGB image dimensions")
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape [3, 3]")
        if self.flow_rc.shape != (*self.depth_m.shape, 2):
            raise ValueError("flow_rc must have shape [height, width, 2]")


@dataclass(frozen=True)
class HeatmapMeasurement:
    values: Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("heatmap must have shape [height, width]")


@dataclass(frozen=True)
class GraspMeasurement:
    points_xyz: Tensor
    transforms: Tensor
    scores: Tensor

    def __post_init__(self) -> None:
        count = self.points_xyz.shape[0]
        if self.points_xyz.ndim != 2 or self.points_xyz.shape[1] != 3:
            raise ValueError("grasp points must have shape [n, 3]")
        if self.transforms.shape != (count, 4, 4):
            raise ValueError("grasp transforms must have shape [n, 4, 4]")
        if self.scores.shape != (count,):
            raise ValueError("grasp scores must have shape [n]")


@dataclass
class Belief2D:
    particles_rc: Tensor
    weights: Tensor
    depth_m: Tensor

    def validate(self) -> None:
        count = self.particles_rc.shape[0]
        if self.particles_rc.shape != (count, 2):
            raise ValueError("2D particles must have shape [n, 2]")
        if self.weights.shape != (count,) or self.depth_m.shape != (count,):
            raise ValueError("2D weights/depth must align with particles")


@dataclass
class Belief3D:
    particles_xyz: Tensor
    weights: Tensor
    grasp_transforms: Tensor

    def validate(self) -> None:
        count = self.particles_xyz.shape[0]
        if self.particles_xyz.shape != (count, 3):
            raise ValueError("3D particles must have shape [n, 3]")
        if self.weights.shape != (count,) or self.grasp_transforms.shape != (count, 4, 4):
            raise ValueError("3D weights/transforms must align with particles")


Belief = Belief2D | Belief3D
Measurement = HeatmapMeasurement | GraspMeasurement


@dataclass(frozen=True)
class StepResult:
    frame_index: int
    timestamp_ns: int
    ready: bool
    beliefs: dict[str, Belief] = field(default_factory=dict)
    fused: bool = False
    fused_graspnet_sample: Tensor | None = None
    initialization_beliefs: dict[str, Belief] = field(default_factory=dict)
