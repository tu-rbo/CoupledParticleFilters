# Coupled Particle Filters

Research code for **Coupled Particle Filters for Robust Affordance Estimation**.
The included workflow couples a HAP image-space affordance belief with a
Contact-GraspNet camera-space grasp belief.

## Environment

The project uses [uv](https://docs.astral.sh/uv/), Python 3.10, ROS 2 Humble,
and Torch 2.1. Source ROS before processing ROS bags:

```bash
source /opt/ros/humble/setup.bash
uv sync --extra evaluation
```

For model-backed precomputation, initialize the Contact-GraspNet submodule and
install the optional model dependencies:

```bash
git submodule update --init --depth 1 pytorch_cgn/contact_graspnet_pytorch
uv sync --extra models
uv pip install -e pytorch_cgn/contact_graspnet_pytorch
uv run python scripts/download_model_assets.py --asset contact_graspnet
```

Contact-GraspNet and its checkpoint are restricted to non-commercial
research/evaluation use under NVIDIA's license. The local HAP adapter and its
checkpoints are intentionally not distributed; use precomputed HAP heatmaps or
provide an authorized adapter through `cpf-precompute --hap-adapter-module`.

## Precompute model data

Precompute the flow, HAP, and GraspNet inputs for a ROS 2 bag. Generated data
is written below `data/` and is never tracked by Git.

```bash
uv run cpf-precompute \
  --bagfile data/rbo_dataset/interactions2/ikea/ikea03_o_ros2/ikea03_o_ros2.db3 \
  --models flow hap graspnet
```

Use `--categories`, `--all-bags`, `--output-root`, `--max-frames`, and
`--overwrite` as needed. Downloaded model assets can be checked without network
access using `uv run python scripts/download_model_assets.py --verify-only`.

## Run coupled fusion

The repository ships one experiment configuration:

```bash
uv run cpf-run --config configs/experiments/filter_fusion.yml --validate-only
uv run cpf-run --config configs/experiments/filter_fusion.yml
```

The configuration consumes precomputed flow, HAP, and GraspNet artifacts. Edit
its dataset paths and bag list for the local installation. Results are written
to `results/` and remain local-only.

## License and citation

First-party code is MIT licensed. External code, model weights, and datasets
retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Citation information is in [CITATION.cff](CITATION.cff). Dataset access and
layout are described in [DATASET.md](DATASET.md).
