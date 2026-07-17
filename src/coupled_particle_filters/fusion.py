"""Pure cross-belief density weighting strategies."""

from __future__ import annotations

import torch

from .config import FusionConfig, FusionStrategy, NumericalConfig, WeightRange
from .math import (
    backproject_rc,
    gather_image,
    normalize_max,
    pairwise_squared_distances,
    sanitize_weights,
)
from .types import Belief2D, Belief3D


class CrossBeliefCoupler:
    """Reweight HAP and GraspNet beliefs using their density in camera space."""

    def __init__(self, config: FusionConfig, numerical: NumericalConfig) -> None:
        self.config = config
        self.numerical = numerical

    def should_run(self, frame_index: int) -> bool:
        return (
            self.config.enabled
            and frame_index >= self.config.start_frame
            and (frame_index - self.config.start_frame) % self.config.every_n_frames == 0
        )

    def apply(
        self,
        hap: Belief2D,
        graspnet: Belief3D,
        intrinsics: torch.Tensor,
        frame_index: int,
        depth_image: torch.Tensor | None = None,
    ) -> None:
        hap.validate()
        graspnet.validate()
        if self.config.strategy == FusionStrategy.PAPER_MULTIPLY:
            if depth_image is None:
                raise ValueError("paper multiplication requires the current depth image")
            hap_depth = gather_image(depth_image, hap.particles_rc)
        else:
            hap_depth = hap.depth_m
        hap_xyz = backproject_rc(hap.particles_rc, hap_depth, intrinsics)
        grasp_density, hap_density = self.cross_density(graspnet.particles_xyz, hap_xyz)

        if self.config.targets == "both":
            update_grasp, update_hap = True, True
        else:
            occurrence = (frame_index - self.config.start_frame) // self.config.every_n_frames
            update_grasp, update_hap = occurrence % 2 == 0, occurrence % 2 == 1

        if update_grasp:
            graspnet.weights = self.combine(
                graspnet.weights,
                grasp_density,
                self.config.graspnet_ranges,
                frame_index,
            )
        if update_hap:
            hap.weights = self.combine(
                hap.weights,
                hap_density,
                self.config.hap_ranges,
                frame_index,
            )

    def cross_density(
        self, grasp_points: torch.Tensor, hap_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if grasp_points.numel() == 0 or hap_points.numel() == 0:
            raise ValueError("cannot couple empty beliefs")
        sigma2 = self.config.kernel_bandwidth_m**2
        grasp_density = grasp_points.new_empty(grasp_points.shape[0])
        hap_density = hap_points.new_zeros(hap_points.shape[0])
        offset = 0
        for chunk in grasp_points.split(self.numerical.distance_chunk_size):
            squared = pairwise_squared_distances(
                chunk, hap_points, self.numerical.distance_backend
            )
            kernel = torch.exp(-0.5 * squared / sigma2)
            grasp_density[offset : offset + chunk.shape[0]] = kernel.sum(dim=1)
            hap_density += kernel.sum(dim=0)
            offset += chunk.shape[0]
        return (
            normalize_max(grasp_density, self.numerical, name="GraspNet cross-density"),
            normalize_max(hap_density, self.numerical, name="HAP cross-density"),
        )

    def combine(
        self,
        measurement_weights: torch.Tensor,
        cross_density: torch.Tensor,
        ranges: WeightRange,
        frame_index: int,
    ) -> torch.Tensor:
        other = normalize_max(cross_density, self.numerical, name="cross-density")
        strategy = self.config.strategy

        if strategy == FusionStrategy.PAPER_MULTIPLY:
            # April 2025 multiply-rate-1 runs clipped the raw measurement
            # likelihood.  Normalizing it first materially changes the relative
            # strength of the two configured ranges.
            raw = sanitize_weights(
                measurement_weights, self.numerical, name="measurement weights"
            )
            # The historical implementation clipped only supported
            # measurements.  Exact zeros remained zero instead of being
            # promoted to the configured lower bound.
            own = raw.clone()
            supported = own != 0
            own[supported] = own[supported].clamp(*ranges.measurement)
            return own * other.clamp(*ranges.cross_density)

        own = normalize_max(measurement_weights, self.numerical, name="measurement weights")

        if strategy == FusionStrategy.MULTIPLY:
            return own * other
        if strategy == FusionStrategy.ADDITION:
            return (own + other) / 2
        if strategy == FusionStrategy.ALTERNATING:
            return self._alternating(own, other, frame_index)

        own_range = ranges.measurement
        other_range = ranges.cross_density
        if strategy in {FusionStrategy.CLIP_MULTIPLY, FusionStrategy.CLIP_ADDITION}:
            own = own.clamp(*own_range)
            other = other.clamp(*other_range)
        else:
            own = self._smooth_clip(own, own_range)
            other = self._smooth_clip(other, other_range)

        if strategy in {FusionStrategy.CLIP_MULTIPLY, FusionStrategy.SMOOTH_CLIP_MULTIPLY}:
            return own * other
        if strategy in {FusionStrategy.CLIP_ADDITION, FusionStrategy.SMOOTH_CLIP_ADDITION}:
            return (own + other) / 2
        if strategy == FusionStrategy.SMOOTH_CLIP_ALTERNATING:
            return self._alternating(own, other, frame_index)
        raise ValueError(f"unsupported fusion strategy: {strategy}")

    def _alternating(
        self, own: torch.Tensor, other: torch.Tensor, frame_index: int
    ) -> torch.Tensor:
        phase = frame_index % self.config.alternating_period
        multiply = phase == self.config.alternating_multiply_phase
        return own * other if multiply else (own + other) / 2

    def _smooth_clip(self, weights: torch.Tensor, limits: tuple[float, float]) -> torch.Tensor:
        low, high = limits
        if high <= low:
            raise ValueError("smooth clipping range must have positive width")
        normalized = (weights - low) / (high - low)
        sigmoid = torch.sigmoid(self.config.smooth_clip_steepness * (normalized - 0.5))
        return low + sigmoid * (high - low)
