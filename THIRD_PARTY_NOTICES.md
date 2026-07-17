# Third-Party Notices

The MIT license in this repository applies only to first-party code.

- Contact-GraspNet PyTorch fork: <https://github.com/tu-rbo/contact_graspnet_pytorch>,
  pinned from upstream commit `2d71da4e50a04aa353352d1cae99f20f7022145b`.
  The fork and checkpoint use the NVIDIA Source Code License for Contact-GraspNet,
  which limits use to non-commercial research/evaluation. The complete license is
  retained as `License.pdf` in the submodule.
- Original Contact-GraspNet: <https://github.com/NVlabs/contact_graspnet>
- PointNet/PointNet++ code referenced by Contact-GraspNet:
  <https://github.com/yanx27/Pointnet_Pointnet2_pytorch>
- Hands as Probes / ACP implementation: <https://github.com/uiuc-robovision/hands-as-probes>,
  pinned for provenance at commit `34d9b19158419c1d5badeb207c01ca476e670b27`.
  The upstream repository does not declare a license. Its code, this project's local
  adapter, checkpoints, and generated heatmaps are therefore excluded from the MIT source
  release unless the copyright holders grant explicit redistribution permission.
- RBO interaction dataset and annotations: distribute only under the dataset's terms.
- Where2Act (optional, experimental baseline): <https://github.com/daerduoCarey/where2act>,
  pinned as a shallow submodule at `other_baseline/where2act/`, upstream commit
  `1daf1a2`. Distributed under the MIT license per the upstream repository's README.
  Its pretrained checkpoints are not redistributed by this repository; obtain them
  from the upstream authors' request form and place them under
  `other_baseline/where2act/code/logs/` as described in
  [docs/external_repositories.md](docs/external_repositories.md).
- Optional FlowBot baseline is not included in the supported package.
- NVIDIA Kaolin 0.17.0 is installed only by the `evaluation` extra from NVIDIA's
  official wheel index and is used for mesh indexing and point-to-mesh distance. Its
  applicable core components are distributed under Apache License 2.0.
- fast-simplification 0.1.13 is installed by the `evaluation` extra as Trimesh's
  quadric mesh-decimation backend and is distributed under the MIT license.

Do not publish third-party checkpoints, datasets, nested repositories, or generated
outputs under the first-party MIT license. Record the exact commit, license, download
URL, and checksum for every release artifact.
