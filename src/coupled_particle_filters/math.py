"""Numerically checked geometry and probability helpers."""

from __future__ import annotations

from typing import Literal

import torch

from .config import NumericalConfig


def sanitize_weights(weights: torch.Tensor, config: NumericalConfig, *, name: str) -> torch.Tensor:
    if config.validation == "fast":
        # Fast validation stays on-device.  Unlike strict mode, it repairs
        # non-finite and negative values instead of raising an exception.
        return torch.where(
            torch.isfinite(weights) & (weights >= 0), weights, torch.zeros_like(weights)
        )

    values = weights.clone()
    finite = torch.isfinite(values)
    if not finite.all():
        if config.nonfinite_policy == "error":
            raise ValueError(f"{name} contains non-finite values")
        values[~finite] = 0
    if (values < 0).any():
        raise ValueError(f"{name} contains negative values")
    return values


def normalize_weights(weights: torch.Tensor, config: NumericalConfig, *, name: str) -> torch.Tensor:
    values = sanitize_weights(weights, config, name=name)
    if config.validation == "fast":
        if values.numel() == 0:
            return values
        total = values.sum()
        normalized = values / total.clamp_min(config.normalization_epsilon)
        uniform = torch.full_like(values, 1.0 / values.numel())
        return torch.where(total > config.normalization_epsilon, normalized, uniform)

    total = values.sum()
    if total <= config.normalization_epsilon:
        if config.zero_mass_policy == "error":
            raise ValueError(f"{name} has zero probability mass")
        if values.numel() == 0:
            raise ValueError(f"{name} is empty")
        return torch.full_like(values, 1.0 / values.numel())
    return values / total


def normalize_max(weights: torch.Tensor, config: NumericalConfig, *, name: str) -> torch.Tensor:
    values = sanitize_weights(weights, config, name=name)
    if config.validation == "fast":
        if values.numel() == 0:
            return values
        maximum = values.max()
        normalized = values / maximum.clamp_min(config.normalization_epsilon)
        return torch.where(maximum > config.normalization_epsilon, normalized, torch.ones_like(values))

    maximum = values.max() if values.numel() else values.new_tensor(0)
    if maximum <= config.normalization_epsilon:
        if config.zero_mass_policy == "error":
            raise ValueError(f"{name} has zero maximum")
        return torch.ones_like(values)
    return values / maximum


def gather_image(image: torch.Tensor, particles_rc: torch.Tensor) -> torch.Tensor:
    rows = particles_rc[:, 0].long().clamp(0, image.shape[0] - 1)
    cols = particles_rc[:, 1].long().clamp(0, image.shape[1] - 1)
    return image[rows, cols]


def clamp_image_points(points_rc: torch.Tensor, height: int, width: int) -> torch.Tensor:
    result = points_rc.clone()
    result[:, 0].clamp_(0, height - 1)
    result[:, 1].clamp_(0, width - 1)
    return result


def backproject_rc(points_rc: torch.Tensor, depth_m: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    row, col = points_rc[:, 0], points_rc[:, 1]
    x = (col - intrinsics[0, 2]) * depth_m / intrinsics[0, 0]
    y = (row - intrinsics[1, 2]) * depth_m / intrinsics[1, 1]
    return torch.stack((x, y, depth_m), dim=1)


def project_xyz(points_xyz: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    z = points_xyz[:, 2]
    if (z <= 0).any():
        raise ValueError("cannot project points with non-positive depth")
    col = intrinsics[0, 0] * points_xyz[:, 0] / z + intrinsics[0, 2]
    row = intrinsics[1, 1] * points_xyz[:, 1] / z + intrinsics[1, 2]
    return torch.stack((row, col), dim=1)


@torch.jit.script
def fast_cdist(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """GEMM-based Euclidean distance from the original research implementation."""

    # Adapted from https://discuss.pytorch.org/t/understanding-cdist-function/76296/12.
    left_norm = left.pow(2).sum(dim=-1, keepdim=True)
    right_norm = right.pow(2).sum(dim=-1, keepdim=True)
    result = torch.addmm(
        right_norm.transpose(-2, -1), left, right.transpose(-2, -1), alpha=-2
    ).add_(left_norm)
    return result.clamp_min_(1e-30).sqrt_()


@torch.jit.script
def fast_squared_cdist(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """GEMM-based squared Euclidean distance without an unnecessary square root."""

    left_norm = left.pow(2).sum(dim=-1, keepdim=True)
    right_norm = right.pow(2).sum(dim=-1, keepdim=True)
    result = torch.addmm(
        right_norm.transpose(-2, -1), left, right.transpose(-2, -1), alpha=-2
    ).add_(left_norm)
    return result.clamp_min_(0)


def pairwise_distances(
    left: torch.Tensor, right: torch.Tensor, distance_backend: Literal["torch", "matmul"]
) -> torch.Tensor:
    """Dispatch a true-distance calculation to the configured backend."""

    if distance_backend == "torch":
        return torch.cdist(left, right)
    if distance_backend == "matmul":
        return fast_cdist(left, right)
    raise ValueError(f"unsupported distance backend: {distance_backend}")


def pairwise_squared_distances(
    left: torch.Tensor, right: torch.Tensor, distance_backend: Literal["torch", "matmul"]
) -> torch.Tensor:
    """Dispatch a squared-distance calculation to the configured backend."""

    if distance_backend == "torch":
        return torch.cdist(left, right).square()
    if distance_backend == "matmul":
        return fast_squared_cdist(left, right)
    raise ValueError(f"unsupported distance backend: {distance_backend}")


def squared_distances(
    left: torch.Tensor,
    right: torch.Tensor,
    chunk_size: int,
    distance_backend: Literal["torch", "matmul"] = "torch",
) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("distance inputs must have shape [n, dimensions]")
    chunks = [
        pairwise_squared_distances(part, right, distance_backend)
        for part in left.split(chunk_size)
    ]
    return torch.cat(chunks, dim=0) if chunks else left.new_empty((0, right.shape[0]))


def sample_indices(
    weights: torch.Tensor,
    count: int,
    numerical: NumericalConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    probabilities = normalize_weights(weights, numerical, name="sampling weights")
    return torch.multinomial(probabilities.cpu(), count, replacement=True, generator=generator).to(weights.device)
