"""Dimension-specific particle filters with a common lifecycle."""

from __future__ import annotations

import torch

from .config import AlgorithmProfile, FilterConfig, NumericalConfig
from .math import clamp_image_points, normalize_weights, sample_indices
from .types import Belief2D, Belief3D


class ParticleFilter2D:
    def __init__(
        self,
        config: FilterConfig,
        numerical: NumericalConfig,
        image_size: tuple[int, int],
        generator: torch.Generator,
        algorithm_profile: AlgorithmProfile = AlgorithmProfile.MODERN,
    ) -> None:
        self.config = config
        self.numerical = numerical
        self.height, self.width = image_size
        self.generator = generator
        self.algorithm_profile = algorithm_profile
        if self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
            # The archived 2D filter constructed an unused random belief before
            # replacing it with measurement samples.  Its draw is part of the
            # shared RNG order used by the paper run.
            torch.rand(
                (config.num_particles, 2),
                generator=self.generator,
                device="cpu",
            )
        self.belief: Belief2D | None = None

    def initialize(self, particles_rc: torch.Tensor, depth_m: torch.Tensor, weights: torch.Tensor) -> Belief2D:
        self.belief = Belief2D(particles_rc, weights, depth_m)
        self.belief.validate()
        return self.belief

    def predict(self, velocity_rc: torch.Tensor, new_depth_m: torch.Tensor, accept_depth: torch.Tensor) -> None:
        belief = self._require_belief()
        if velocity_rc.shape != belief.particles_rc.shape:
            raise ValueError("2D velocity must align with particles")
        if self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
            noise = torch.rand(
                belief.particles_rc.shape,
                generator=self.generator,
                device="cpu",
                dtype=belief.particles_rc.dtype,
            )
            noise = noise.to(belief.particles_rc.device)
            noise = (noise - 0.5) * self.config.motion_noise_std
        else:
            noise = torch.randn(
                belief.particles_rc.shape,
                generator=self.generator,
                device="cpu",
                dtype=belief.particles_rc.dtype,
            ).to(belief.particles_rc.device) * self.config.motion_noise_std
        belief.particles_rc = clamp_image_points(
            belief.particles_rc + velocity_rc + noise, self.height, self.width
        )
        belief.depth_m = torch.where(accept_depth, new_depth_m, belief.depth_m)

    def update(self, weights: torch.Tensor) -> None:
        belief = self._require_belief()
        if weights.shape != belief.weights.shape:
            raise ValueError("2D measurement weights must align with particles")
        belief.weights = weights

    def inject(self, particles_rc: torch.Tensor, depth_m: torch.Tensor, weights: torch.Tensor) -> None:
        belief = self._require_belief()
        belief.particles_rc = torch.cat((belief.particles_rc, particles_rc))
        belief.depth_m = torch.cat((belief.depth_m, depth_m))
        belief.weights = torch.cat((belief.weights, weights))
        belief.validate()

    def resample(self) -> None:
        belief = self._require_belief()
        probabilities = normalize_weights(belief.weights, self.numerical, name="2D particle weights")
        indices = _systematic_resample(probabilities, self.config.num_particles, self.generator)
        belief.particles_rc = belief.particles_rc[indices]
        belief.depth_m = belief.depth_m[indices]
        belief.weights = torch.full_like(probabilities[: self.config.num_particles], 1 / self.config.num_particles)
        belief.validate()

    def _require_belief(self) -> Belief2D:
        if self.belief is None:
            raise RuntimeError("2D filter has not been initialized")
        return self.belief


class ParticleFilter3D:
    def __init__(
        self,
        config: FilterConfig,
        numerical: NumericalConfig,
        generator: torch.Generator,
        algorithm_profile: AlgorithmProfile = AlgorithmProfile.MODERN,
    ) -> None:
        self.config = config
        self.numerical = numerical
        self.generator = generator
        self.algorithm_profile = algorithm_profile
        self.belief: Belief3D | None = None

    def initialize(
        self, particles_xyz: torch.Tensor, transforms: torch.Tensor, weights: torch.Tensor
    ) -> Belief3D:
        self.belief = Belief3D(particles_xyz, weights, transforms)
        self.belief.validate()
        return self.belief

    def predict(self, velocity_xyz: torch.Tensor) -> None:
        belief = self._require_belief()
        noise = torch.randn(
            belief.particles_xyz.shape,
            generator=self.generator,
            device="cpu",
            dtype=belief.particles_xyz.dtype,
        ).to(belief.particles_xyz.device) * self.config.motion_noise_std
        movement = velocity_xyz + noise
        if self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
            # Legacy gripper prediction generated zero-span Euler and
            # translation noise after translational motion.  Although both
            # tensors were numerically zero, normal_ advanced the global RNG.
            torch.randn(
                (belief.particles_xyz.shape[0], 3),
                generator=self.generator,
                device="cpu",
                dtype=belief.particles_xyz.dtype,
            )
            torch.randn(
                (belief.particles_xyz.shape[0], 3),
                generator=self.generator,
                device="cpu",
                dtype=belief.particles_xyz.dtype,
            )
        belief.particles_xyz = belief.particles_xyz + movement
        belief.grasp_transforms = belief.grasp_transforms.clone()
        belief.grasp_transforms[:, :3, 3] += movement

    def update(self, weights: torch.Tensor) -> None:
        belief = self._require_belief()
        if weights.shape != belief.weights.shape:
            raise ValueError("3D measurement weights must align with particles")
        belief.weights = weights

    def inject(self, points_xyz: torch.Tensor, transforms: torch.Tensor, weights: torch.Tensor) -> None:
        belief = self._require_belief()
        belief.particles_xyz = torch.cat((belief.particles_xyz, points_xyz))
        belief.grasp_transforms = torch.cat((belief.grasp_transforms, transforms))
        belief.weights = torch.cat((belief.weights, weights))
        belief.validate()

    def resample(self) -> None:
        belief = self._require_belief()
        probabilities = normalize_weights(belief.weights, self.numerical, name="3D particle weights")
        start_lower = (
            1e-6
            if self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025
            else 0.0
        )
        indices = _systematic_resample(
            probabilities,
            self.config.num_particles,
            self.generator,
            start_lower=start_lower,
        )
        belief.particles_xyz = belief.particles_xyz[indices]
        belief.grasp_transforms = belief.grasp_transforms[indices]
        belief.weights = torch.full_like(probabilities[: self.config.num_particles], 1 / self.config.num_particles)
        belief.validate()

    def _require_belief(self) -> Belief3D:
        if self.belief is None:
            raise RuntimeError("3D filter has not been initialized")
        return self.belief


def _systematic_resample(
    probabilities: torch.Tensor,
    count: int,
    generator: torch.Generator,
    *,
    start_lower: float = 0.0,
) -> torch.Tensor:
    """Systematic resampling with deterministic CPU sampling and source-device indices."""
    cumulative = probabilities.detach().cpu().cumsum(0)
    start = (
        torch.empty((), dtype=torch.float32)
        .uniform_(start_lower, 1.0, generator=generator)
        .item()
        / count
    )
    pointers = start + torch.arange(count, dtype=cumulative.dtype) / count
    indices = torch.searchsorted(cumulative, pointers).clamp(max=probabilities.numel() - 1)
    return indices.to(probabilities.device)


def sample_2d_belief(
    distribution: torch.Tensor,
    depth_image: torch.Tensor,
    count: int,
    numerical: NumericalConfig,
    generator: torch.Generator,
) -> Belief2D:
    indices = sample_indices(distribution.flatten(), count, numerical, generator)
    width = distribution.shape[1]
    points = torch.stack((indices // width, indices % width), dim=1).to(distribution.dtype)
    depths = depth_image[points[:, 0].long(), points[:, 1].long()]
    weights = distribution.flatten()[indices]
    belief = Belief2D(points, weights, depths)
    belief.validate()
    return belief


def sample_3d_belief(
    points: torch.Tensor,
    transforms: torch.Tensor,
    scores: torch.Tensor,
    count: int,
    numerical: NumericalConfig,
    generator: torch.Generator,
) -> Belief3D:
    indices = sample_indices(scores, count, numerical, generator)
    belief = Belief3D(points[indices], scores[indices], transforms[indices])
    belief.validate()
    return belief
