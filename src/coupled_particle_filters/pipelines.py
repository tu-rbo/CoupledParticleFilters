"""HAP and GraspNet estimator pipelines."""

from __future__ import annotations

from collections import deque
from typing import Protocol

import torch
import torch.nn.functional as functional

from .config import (
    AlgorithmProfile,
    FilterConfig,
    NumericalConfig,
    PipelineConfig,
    PipelineMode,
)
from .filters import ParticleFilter2D, ParticleFilter3D, sample_2d_belief, sample_3d_belief
from .math import (
    backproject_rc,
    clamp_image_points,
    gather_image,
    normalize_max,
    pairwise_distances,
    pairwise_squared_distances,
    sample_indices,
)
from .types import (
    Belief,
    Belief2D,
    Belief3D,
    Frame,
    GraspMeasurement,
    HeatmapMeasurement,
    Measurement,
)


class MeasurementProvider(Protocol):
    def get(self, frame: Frame) -> Measurement: ...


class EstimatorPipeline(Protocol):
    name: str

    def advance(self, frame: Frame) -> Belief | None: ...
    def resample(self) -> None: ...


class HapPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        numerical: NumericalConfig,
        image_size: tuple[int, int],
        provider: MeasurementProvider,
        generator: torch.Generator,
        algorithm_profile: AlgorithmProfile = AlgorithmProfile.MODERN,
        max_depth_m: float | None = None,
    ) -> None:
        self.name = config.name
        self.config = config
        self.numerical = numerical
        self.provider = provider
        self.generator = generator
        self.algorithm_profile = algorithm_profile
        self.max_depth_m = max_depth_m
        self.initialized_this_step = False
        self.paper_injection_ready = False
        self.frame_index = 0
        queue_length = config.filter.queue_length if config.filter else 1
        self.queue: deque[tuple[HeatmapMeasurement, torch.Tensor]] = deque(maxlen=queue_length)
        self.filter = (
            ParticleFilter2D(
                config.filter,
                numerical,
                image_size,
                generator,
                algorithm_profile,
            )
            if config.mode == PipelineMode.PARTICLE_FILTER and config.filter
            else None
        )

    def advance(self, frame: Frame) -> Belief2D | None:
        frame_index = self.frame_index
        self.frame_index += 1
        self.initialized_this_step = False
        measurement = self.provider.get(frame)
        if not isinstance(measurement, HeatmapMeasurement):
            raise TypeError("HAP provider must return HeatmapMeasurement")
        if measurement.values.shape != frame.depth_m.shape:
            raise ValueError("HAP heatmap and frame dimensions differ")

        if self.config.mode == PipelineMode.RAW_SAMPLED:
            return sample_2d_belief(
                measurement.values,
                frame.depth_m,
                self.config.raw_sample_count,
                self.numerical,
                self.generator,
            )

        assert self.filter is not None and self.config.filter is not None
        if not _append_measurement(
            self.queue,
            (measurement, frame.depth_m),
            self.config.filter.readiness_policy,
        ):
            return None
        heatmap = torch.stack([item[0].values for item in self.queue]).mean(0)
        depth = self.queue[-1][1]

        if self.filter.belief is None:
            clipped = (
                heatmap.clamp(*self.config.filter.initialization_clip)
                if self.config.filter.initialize_from_measurement
                else torch.ones_like(heatmap)
            )
            sampled = sample_2d_belief(
                clipped,
                depth,
                self.config.filter.num_particles,
                self.numerical,
                self.generator,
            )
            initialized = self.filter.initialize(
                sampled.particles_rc, sampled.depth_m, sampled.weights
            )
            self.initialized_this_step = True
            if self.config.filter.readiness_policy == "legacy_extra_frame":
                return None
            return initialized

        belief = self.filter.belief
        old_depth = belief.depth_m
        positions = clamp_image_points(belief.particles_rc, *frame.depth_m.shape)
        velocity = gather_image(frame.flow_rc, positions)
        motion_threshold = (
            1.0
            if self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025
            else self.config.filter.motion_threshold_px
        )
        velocity[velocity.norm(dim=1) < motion_threshold] = 0
        proposed = clamp_image_points(positions + velocity, *frame.depth_m.shape)
        new_depth = gather_image(depth, proposed)
        if self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
            reject = (
                (new_depth - old_depth).abs()
                > self.config.filter.max_particle_movement_m
            )
            if self.max_depth_m is not None:
                # The archived HAP path allowed motion into invalid/max-depth
                # pixels but rejected genuine foreground/background jumps.
                reject &= new_depth != self.max_depth_m
            velocity[reject] = 0
            accept = ~reject
        else:
            accept = (
                (new_depth - old_depth).abs()
                <= self.config.filter.max_particle_movement_m
            )
        self.filter.predict(velocity, new_depth, accept)

        injection_count = _injection_admitted_count(self.config.filter, frame_index)
        should_inject = injection_count and (
            self.algorithm_profile != AlgorithmProfile.PAPER_MULTIPLY_2025
            or self.paper_injection_ready
        )
        if should_inject:
            if self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
                adjusted = _paper_sparse_heatmap_distribution(
                    heatmap,
                    self.filter.belief.particles_rc,
                    self.config.filter.injection_mu_offset,
                    self.config.filter.injection_sigma_offset,
                    self.numerical,
                )
            else:
                adjusted = _sparse_heatmap_distribution(
                    heatmap,
                    self.filter.belief.particles_rc,
                    self.config.filter.injection_density_sigma,
                    self.config.filter.injection_mu_offset,
                    self.config.filter.injection_sigma_offset,
                    self.numerical,
                )
            injected = sample_2d_belief(
                adjusted,
                depth,
                _injection_draw_count(self.config.filter),
                self.numerical,
                self.generator,
            )
            count = injection_count
            self.filter.inject(
                injected.particles_rc[:count],
                injected.depth_m[:count],
                injected.weights[:count],
            )

        self.filter.update(gather_image(heatmap, self.filter.belief.particles_rc))
        if (
            self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025
            and self.config.filter.injection_count
        ):
            # The archived pipeline enabled injection only after completing its
            # first prediction/update step.
            self.paper_injection_ready = True
        return self.filter.belief

    def resample(self) -> None:
        if self.filter is not None and self.filter.belief is not None:
            self.filter.resample()


class GraspNetPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        numerical: NumericalConfig,
        provider: MeasurementProvider,
        generator: torch.Generator,
        max_depth_m: float,
        algorithm_profile: AlgorithmProfile = AlgorithmProfile.MODERN,
    ) -> None:
        self.name = config.name
        self.config = config
        self.numerical = numerical
        self.provider = provider
        self.generator = generator
        self.max_depth_m = max_depth_m
        self.algorithm_profile = algorithm_profile
        self.initialized_this_step = False
        self.paper_injection_ready = False
        self.frame_index = 0
        queue_length = config.filter.queue_length if config.filter else 1
        self.queue: deque[GraspMeasurement] = deque(maxlen=queue_length)
        self.filter = (
            ParticleFilter3D(config.filter, numerical, generator, algorithm_profile)
            if config.mode == PipelineMode.PARTICLE_FILTER and config.filter
            else None
        )

    def advance(self, frame: Frame) -> Belief3D | None:
        frame_index = self.frame_index
        self.frame_index += 1
        self.initialized_this_step = False
        measurement = self.provider.get(frame)
        if not isinstance(measurement, GraspMeasurement):
            raise TypeError("GraspNet provider must return GraspMeasurement")
        measurement = self._penalize_occlusion(measurement, frame)

        if self.config.mode == PipelineMode.RAW_SAMPLED:
            return sample_3d_belief(
                measurement.points_xyz,
                measurement.transforms,
                measurement.scores,
                self.config.raw_sample_count,
                self.numerical,
                self.generator,
            )

        assert self.filter is not None and self.config.filter is not None
        if not _append_measurement(
            self.queue,
            measurement,
            self.config.filter.readiness_policy,
        ):
            return None
        queued = GraspMeasurement(
            torch.cat([item.points_xyz for item in self.queue]),
            torch.cat([item.transforms for item in self.queue]),
            torch.cat([item.scores for item in self.queue]),
        )

        if self.filter.belief is None:
            scores = (
                queued.scores.clamp(*self.config.filter.initialization_clip)
                if self.config.filter.initialize_from_measurement
                else torch.ones_like(queued.scores)
            )
            sampled = sample_3d_belief(
                queued.points_xyz,
                queued.transforms,
                scores,
                self.config.filter.num_particles,
                self.numerical,
                self.generator,
            )
            initialized = self.filter.initialize(
                sampled.particles_xyz, sampled.grasp_transforms, sampled.weights
            )
            self.initialized_this_step = True
            if self.config.filter.readiness_policy == "legacy_extra_frame":
                return None
            return initialized

        belief = self.filter.belief
        projected_rc = _project_allow_invalid(belief.particles_xyz, frame.intrinsics)
        lookup_rc = clamp_image_points(projected_rc, *frame.depth_m.shape)
        flow = gather_image(frame.flow_rc, lookup_rc)
        if self.config.filter.flow_coordinates == "legacy_xy_as_row_column":
            # Stored artifacts have already been converted to row/column.  The
            # paper GraspNet path nevertheless added them as x/y.
            flow = flow[:, [1, 0]]
        if self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
            # Legacy code sampled depth at a clamped location, but used the
            # unclamped projected coordinate for backprojection.
            moved_rc = projected_rc + flow
            moved_lookup_rc = clamp_image_points(moved_rc, *frame.depth_m.shape)
            moved_depth = gather_image(frame.depth_m, moved_lookup_rc)
            moved_xyz = backproject_rc(moved_rc, moved_depth, frame.intrinsics)
        else:
            moved_rc = clamp_image_points(lookup_rc + flow, *frame.depth_m.shape)
            moved_depth = gather_image(frame.depth_m, moved_rc)
            moved_xyz = backproject_rc(moved_rc, moved_depth, frame.intrinsics)
        movement = moved_xyz - belief.particles_xyz
        movement[movement.norm(dim=1) > self.config.filter.max_particle_movement_m] = 0
        self.filter.predict(movement)

        injection_count = _injection_admitted_count(self.config.filter, frame_index)
        should_inject = injection_count and (
            self.algorithm_profile != AlgorithmProfile.PAPER_MULTIPLY_2025
            or self.paper_injection_ready
        )
        if should_inject:
            density = _point_density(
                queued.points_xyz,
                self.filter.belief.particles_xyz,
                self.config.filter.injection_density_sigma,
                self.numerical,
            )
            correction = (
                0
                if self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025
                else 1
            )
            mean = density.mean()
            std = density.std(correction=correction).clamp_min(
                self.numerical.normalization_epsilon
            )
            scaling = torch.exp(
                -0.5
                * ((density - mean + self.config.filter.injection_mu_offset) ** 2)
                / ((std + self.config.filter.injection_sigma_offset) ** 2)
            )
            injection_scores = queued.scores * scaling
            if self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
                injection_scores = _paper_grasp_injection_scores(
                    queued.scores, scaling
                )
            indices = sample_indices(
                injection_scores,
                _injection_draw_count(self.config.filter),
                self.numerical,
                self.generator,
            )
            indices = indices[:injection_count]
            self.filter.inject(
                queued.points_xyz[indices], queued.transforms[indices], queued.scores[indices]
            )

        weights = _grasp_measurement_weights(
            self.filter.belief,
            queued,
            self.config.filter.measurement_neighbor_radius_m,
            self.config.filter.measurement_distance_sigma,
            self.config.filter.orientation_weight,
            self.numerical,
        )
        self.filter.update(weights)
        if (
            self.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025
            and self.config.filter.injection_count
        ):
            self.paper_injection_ready = True
        return self.filter.belief

    def resample(self) -> None:
        if self.filter is not None and self.filter.belief is not None:
            self.filter.resample()

    def _penalize_occlusion(self, measurement: GraspMeasurement, frame: Frame) -> GraspMeasurement:
        config = self.config.filter
        if config is None:
            return measurement
        depth = frame.depth_m
        kernel = config.occlusion_kernel_size
        padded = depth.unsqueeze(0).unsqueeze(0)
        local_min = -functional.max_pool2d(-padded, kernel, 1, kernel // 2).squeeze()
        edge = (depth - local_min).abs() > config.occlusion_edge_max_diff_m
        invalid = ~torch.isfinite(depth) | (depth >= self.max_depth_m)
        invalid_dilated = functional.max_pool2d(
            invalid.float().unsqueeze(0).unsqueeze(0), kernel, 1, kernel // 2
        ).squeeze().bool()
        local_max = functional.max_pool2d(padded, kernel, 1, kernel // 2).squeeze()
        near_object = invalid_dilated & (
            (depth - local_max).abs() > config.occlusion_near_object_max_diff_m
        )
        mask = edge | invalid_dilated
        mask[near_object] = False
        border = config.image_border_px
        if border:
            mask[:border] = True
            mask[-border:] = True
            mask[:, :border] = True
            mask[:, -border:] = True
        projected = clamp_image_points(
            _project_allow_invalid(measurement.points_xyz, frame.intrinsics), *depth.shape
        )
        scores = measurement.scores.clone()
        scores[gather_image(mask, projected).bool()] = 0
        return GraspMeasurement(measurement.points_xyz, measurement.transforms, scores)


def _append_measurement(
    queue: deque,
    measurement,
    readiness_policy: str,
) -> bool:
    """Append a measurement and report whether prediction/update may run.

    ``legacy_extra_frame`` reproduces the research scripts: after the queue
    becomes full, one additional measurement is required before it is usable.
    ``when_full`` is the corrected low-latency behavior for new experiments.
    """

    if readiness_policy == "legacy_extra_frame":
        was_full = len(queue) == queue.maxlen
        queue.append(measurement)
        return was_full
    queue.append(measurement)
    return len(queue) == queue.maxlen


def _project_allow_invalid(points_xyz: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points_xyz[:, 2].clamp_min(torch.finfo(points_xyz.dtype).eps)
    col = intrinsics[0, 0] * points_xyz[:, 0] / z + intrinsics[0, 2]
    row = intrinsics[1, 1] * points_xyz[:, 1] / z + intrinsics[1, 2]
    return torch.stack((row, col), dim=1)


def _injection_draw_count(config: FilterConfig) -> int:
    """Return the RNG draw budget independently of admitted injection strength."""

    return config.injection_draw_count or config.injection_count


def _injection_admitted_count(config: FilterConfig, frame_index: int) -> int:
    if config.injection_warmup is not None:
        frames, count = config.injection_warmup
        if frame_index < frames:
            return count
    return config.injection_count


def _point_density(
    query: torch.Tensor,
    belief: torch.Tensor,
    sigma: float,
    numerical: NumericalConfig,
) -> torch.Tensor:
    result = query.new_empty(query.shape[0])
    offset = 0
    for chunk in query.split(numerical.distance_chunk_size):
        squared = pairwise_squared_distances(chunk, belief, numerical.distance_backend)
        result[offset : offset + chunk.shape[0]] = torch.exp(
            -0.5 * squared / sigma**2
        ).sum(1)
        offset += chunk.shape[0]
    return normalize_max(result, numerical, name="particle density")


def _grasp_measurement_weights(
    belief: Belief3D,
    measurement: GraspMeasurement,
    radius: float,
    sigma: float,
    orientation_weight: float,
    numerical: NumericalConfig,
) -> torch.Tensor:
    output = belief.weights.new_zeros(belief.particles_xyz.shape[0])
    offset = 0
    measurement_rotations = measurement.transforms[:, :3, :3].flatten(1)
    for particles in belief.particles_xyz.split(numerical.distance_chunk_size):
        end = offset + particles.shape[0]
        distances = pairwise_distances(
            particles, measurement.points_xyz, numerical.distance_backend
        )
        near = distances < radius
        distance_score = torch.exp(-0.5 * distances / sigma) * measurement.scores
        distance_score[~near] = 0
        particle_rotations = belief.grasp_transforms[offset:end, :3, :3].flatten(1)
        similarity = functional.cosine_similarity(
            particle_rotations[:, None, :], measurement_rotations[None, :, :], dim=2
        ).clamp(0, 1)
        similarity = similarity * measurement.scores
        similarity[~near] = 0
        output[offset:end] = (
            (1 - orientation_weight) * distance_score.max(1).values
            + orientation_weight * similarity.max(1).values
        )
        offset = end
    return output


def _sparse_heatmap_distribution(
    heatmap: torch.Tensor,
    particles_rc: torch.Tensor,
    sigma: float,
    mu_offset: float,
    sigma_offset: float,
    numerical: NumericalConfig,
) -> torch.Tensor:
    occupancy = torch.zeros_like(heatmap)
    points = clamp_image_points(particles_rc, *heatmap.shape).long()
    occupancy.index_put_((points[:, 0], points[:, 1]), torch.ones_like(points[:, 0], dtype=heatmap.dtype), accumulate=True)
    radius = max(1, int(round(3 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=heatmap.dtype, device=heatmap.device)
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).unsqueeze(0).unsqueeze(0)
    density = functional.conv2d(
        occupancy.unsqueeze(0).unsqueeze(0), kernel, padding=radius
    ).squeeze()
    density = normalize_max(density, numerical, name="2D particle density")
    mean, std = density.mean(), density.std().clamp_min(numerical.normalization_epsilon)
    scaling = torch.exp(-0.5 * ((density - mean + mu_offset) / (std + sigma_offset)).square())
    return heatmap * scaling


def _paper_sparse_heatmap_distribution(
    heatmap: torch.Tensor,
    particles_rc: torch.Tensor,
    mu_offset: float,
    sigma_offset: float,
    numerical: NumericalConfig,
) -> torch.Tensor:
    """April 2025 HAP injection density and Gaussian scaling.

    Duplicate particles overwrite the same binary pixel.  The historical blur
    selected its kernel from image area with ``k_ratio=6`` and used reflected
    padding.  ``scipy.stats.norm.fit`` used by the research script corresponds
    to the population standard deviation (``correction=0``).
    """

    occupancy = torch.zeros_like(heatmap)
    points = clamp_image_points(particles_rc, *heatmap.shape).long()
    occupancy[points[:, 0], points[:, 1]] = 1
    density = _paper_gaussian_blur(occupancy, kernel_ratio=6)
    density = normalize_max(density, numerical, name="legacy 2D particle density")
    mean = density.mean()
    std = density.std(correction=0).clamp_min(numerical.normalization_epsilon)
    scaling = torch.exp(
        -0.5 * (density - mean + mu_offset).square() / (std + sigma_offset).square()
    )
    return heatmap * scaling


def _paper_grasp_injection_scores(
    measurement_scores: torch.Tensor, scaling: torch.Tensor
) -> torch.Tensor:
    scores = measurement_scores * scaling
    scores = scores.clone()
    scores[measurement_scores < 0.1] = 0
    return scores


def _paper_gaussian_blur(image: torch.Tensor, kernel_ratio: int) -> torch.Tensor:
    height, width = image.shape
    kernel_size = max(1, int((height * width) ** 0.5 / kernel_ratio))
    kernel_size += 1 - kernel_size % 2
    # Reflection padding requires a half-kernel smaller than both dimensions.
    kernel_size = min(kernel_size, 2 * min(height, width) - 1)
    kernel_size -= 1 - kernel_size % 2
    kernel_size = max(1, kernel_size)
    if kernel_size == 1:
        return image.clone()

    sigma = 0.3 * (((kernel_size - 1) * 0.5) - 1) + 0.8
    radius = kernel_size // 2
    coordinates = torch.arange(
        -radius, radius + 1, dtype=image.dtype, device=image.device
    )
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel /= kernel.sum()
    values = image.unsqueeze(0).unsqueeze(0)
    values = functional.pad(values, (radius, radius, 0, 0), mode="reflect")
    values = functional.conv2d(values, kernel.view(1, 1, 1, -1))
    values = functional.pad(values, (0, 0, radius, radius), mode="reflect")
    return functional.conv2d(values, kernel.view(1, 1, -1, 1)).squeeze()
