"""First-party adapter that runs the upstream Where2Act model on RBO point clouds.

Where2Act (https://github.com/daerduoCarey/where2act, MIT licensed) is vendored as a
git submodule at ``other_baseline/where2act``. This module is first-party glue code
-- it is not part of the upstream project -- that loads the pretrained network and
exposes the ``infer``/``infer_on_pcd`` interface expected by ``precompute_data.py``.

``other_baseline`` and ``other_baseline.where2act.code`` are implicit namespace
packages (no ``__init__.py``), matching how the upstream Where2Act checkout is laid
out, so the submodule can be updated without patching package metadata.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import other_baseline.where2act.code.models.model_3d_legacy as model_def


DEFAULT_CHECKPOINT_DIR = os.path.join(
    "other_baseline",
    "where2act",
    "code",
    "logs",
    "finalexp-model_all_final-pulling-None-train_all_v1",
)
DEFAULT_EPOCH = 81


class Where2Act_Runner:
    """Loads the pretrained Where2Act network and runs point-cloud inference."""

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        epoch: int = DEFAULT_EPOCH,
        force_cpu: bool = False,
    ) -> None:
        # Reproduce the deterministic sampling used by the upstream training/eval code.
        torch.manual_seed(777)
        np.random.seed(777)

        self.device = "cpu" if force_cpu else "cuda"

        train_conf = torch.load(os.path.join(checkpoint_dir, "conf.pth"))
        self.network = model_def.Network(
            train_conf.feat_dim, train_conf.rv_dim, train_conf.rv_cnt
        )
        data_to_restore = torch.load(
            os.path.join(checkpoint_dir, "ckpts", f"{epoch}-network.pth")
        )
        self.network.load_state_dict(data_to_restore, strict=False)
        self.network.to(self.device)
        self.network.eval()

    def test(self, pcd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pcd = pcd.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_6d = self.network.inference_actor(pcd)[0]  # RV_CNT x 6
            self.network.actor.bgs(pred_6d.reshape(-1, 3, 2))  # RV_CNT x 3 x 3
            pred_action_score_map = self.network.inference_action_score(pcd)[0]  # N
            pred_action_score_map = pred_action_score_map.cpu()

        return pcd, pred_action_score_map

    def infer(self, rgb, depth, cam_K) -> tuple[None, dict[str, torch.Tensor]]:
        depth = torch.Tensor(depth)
        K = torch.Tensor(cam_K)
        pcd = self.depth_to_pointcloud(depth, K).T
        pcd = pcd[~pcd.isnan().view((-1, 3)).any(1)]
        if pcd.shape[0] > 30000:
            pcd = pcd[torch.randperm(pcd.shape[0])[:30000]]

        mean = pcd.mean(0)
        pcd_ = pcd - mean
        pcd_p, scores = self.test(pcd_)
        pcd = pcd_p.cpu() + mean
        pcd = pcd.squeeze()
        output = {
            "cntct_pts": pcd,
            "predicted_grasps": torch.eye(4).unsqueeze(0).repeat(pcd.shape[0], 1, 1),
            "scores": scores,
        }

        return None, output

    def depth_to_pointcloud(self, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        h, w = depth.shape
        x = (torch.linspace(0, w - 1, w).repeat(h, 1) - K[0, 2]) / K[0, 0]
        y = (torch.linspace(0, h - 1, h).repeat(w, 1).t() - K[1, 2]) / K[1, 1]

        # Calculate the Z coordinate in the camera frame
        z = depth

        # Calculate the X and Y coordinates in the camera frame
        x = x * z
        y = y * z

        # Reshape the coordinates to (3, N)
        coords = torch.stack((x, y, z)).view(3, -1)
        return coords

    def infer_on_pcd(self, pcd: torch.Tensor) -> tuple[None, dict[str, torch.Tensor]]:
        pcd = pcd[~pcd.isnan().view((-1, 3)).any(1)]
        if pcd.shape[0] > 30000:
            pcd = pcd[torch.randperm(pcd.shape[0])[:30000]]

        mean = pcd.mean(0)
        pcd_ = pcd - mean
        pcd_p, scores = self.test(pcd_)
        pcd = pcd_p.cpu() + mean
        pcd = pcd.squeeze()
        output = {
            "cntct_pts": pcd,
            "predicted_grasps": torch.eye(4).unsqueeze(0).repeat(pcd.shape[0], 1, 1),
            "scores": scores,
        }

        return None, output
