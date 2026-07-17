"""Small compatibility helpers for the legacy geometry evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .math import project_xyz


@dataclass(frozen=True)
class ImageArtifact:
    data: object
    caption: str | None = None


class NullRunContext:
    def log(self, payload: object) -> None:
        return None


def combined_mesh_bounds(meshes: object) -> tuple[np.ndarray, np.ndarray]:
    """Return the axis-aligned union bounds without concatenating mesh visuals.

    ``trimesh`` concatenation also packs texture materials.  The legacy
    evaluator only needs the combined bounding box, so concatenation can spend
    minutes resizing textures while producing the same min/max coordinates.
    """

    bounds = np.asarray([np.asarray(mesh.bounds) for mesh in meshes], dtype=float)
    if bounds.ndim != 3 or bounds.shape[1:] != (2, 3) or not len(bounds):
        raise ValueError("mesh bounds must have shape (mesh_count, 2, 3)")
    return bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)


def simplify_mesh(mesh: object, face_count: int = 5000) -> object:
    """Simplify a Trimesh mesh across the legacy and current APIs.

    New Trimesh releases renamed ``simplify_quadratic_decimation`` to
    ``simplify_quadric_decimation`` and made the target face count a keyword
    argument. The legacy evaluator was written against the old API.
    """
    faces = getattr(mesh, "faces", None)
    if faces is not None and len(faces) <= face_count:
        return mesh

    simplify = getattr(mesh, "simplify_quadric_decimation", None)
    if callable(simplify):
        return simplify(face_count=face_count)

    simplify = getattr(mesh, "simplify_quadratic_decimation", None)
    if callable(simplify):
        return simplify(face_count)

    raise AttributeError("mesh does not provide a supported quadric decimation method")


def image_artifact(data: object, caption: str | None = None) -> ImageArtifact:
    return ImageArtifact(data=data, caption=caption)


def pxlpos2pcd(
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
    intrinsics: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    x = (pixel_x - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (pixel_y - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    return torch.stack((x, y, depth), dim=2).reshape(-1, 3)


def depth2pcd(
    depth_img: torch.Tensor,
    intrinsics: torch.Tensor,
    flow: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a depth image to a point cloud, optionally warping pixels by ``flow``."""
    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]
    # Project depth into 3D pointcloud in camera coordinates.
    pixel_y, pixel_x = torch.meshgrid(
        torch.linspace(0, img_h - 1, img_h),
        torch.linspace(0, img_w - 1, img_w),
        indexing="ij",
    )
    if flow is not None:
        pixel_x, pixel_y = torch.stack((pixel_x, pixel_y), dim=0) + flow.permute([2, 0, 1])
    return pxlpos2pcd(pixel_x, pixel_y, intrinsics, depth_img)


def torch_project_to_image_plane(
    points_xyz: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    return project_xyz(torch.as_tensor(points_xyz).float(), torch.as_tensor(intrinsics).float())


def compute_heatmap_torch(
    points: torch.Tensor,
    scores: torch.Tensor,
    image_size: torch.Tensor,
    k_ratio: float = 6.0,
    normalize: bool = False,
) -> torch.Tensor:
    height, width = (int(value) for value in image_size)
    heatmap = np.zeros((height, width), dtype=np.float32)
    points = torch.as_tensor(points).long().cpu()
    scores = torch.as_tensor(scores).float().cpu()
    valid = (
        (points[:, 0] >= 0)
        & (points[:, 0] < height)
        & (points[:, 1] >= 0)
        & (points[:, 1] < width)
    )
    valid_points = points[valid]
    np.add.at(
        heatmap,
        (valid_points[:, 0].numpy(), valid_points[:, 1].numpy()),
        scores[valid].numpy(),
    )
    kernel = max(1, int(np.sqrt(height * width) / k_ratio))
    kernel += 1 - kernel % 2
    heatmap = cv2.GaussianBlur(heatmap, (kernel, kernel), 0)
    if normalize and heatmap.max() > 0:
        heatmap /= heatmap.max()
    return torch.from_numpy(heatmap).unsqueeze(0)
