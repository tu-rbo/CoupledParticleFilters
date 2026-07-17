# Coupled Particle Filters

Code for **Coupled Particle Filters for Robust Affordance Estimation**.
Read the paper for more details [Paper Link](https://arxiv.org/abs/2603.15223]

## Environment

The project uses [uv](https://docs.astral.sh/uv/), Python 3.10, ROS 2 Humble,
and Torch 2.1. Source ROS before processing ROS bags:

```bash
source /opt/ros/humble/setup.bash
uv sync --extra evaluation
```

To precompute the inference data, initialize the Contact-GraspNet submodule and
install the optional model dependencies:

```bash
git submodule update --init --depth 1 pytorch_cgn/contact_graspnet_pytorch
uv sync --extra models
uv pip install -e pytorch_cgn/contact_graspnet_pytorch
uv run python scripts/download_model_assets.py --asset contact_graspnet
```

You can precompute the HAP data with `cpf-precompute --hap-adapter-module`.

## Precompute model data

Precompute the flow, HAP, and GraspNet inputs for a ROS 2 bag and store them in /data.

```bash
uv run cpf-precompute \
  --bagfile data/rbo_dataset/interactions2/ikea/ikea03_o_ros2/ikea03_o_ros2.db3 \
  --models flow hap graspnet
```

Use `--categories`, `--all-bags`, `--output-root`, `--max-frames`, and
`--overwrite` as needed. Downloaded model assets can be checked without network
access using `uv run python scripts/download_model_assets.py --verify-only`.

## Run the coupled fusion experiment

```bash
uv run cpf-run --config configs/experiments/filter_fusion.yml
```

Edit the dataset paths and bag list for the local installation. Results are written
to `results/`.

## License and citation

First-party code is MIT licensed. External code, model weights, and datasets
retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Citation information is in [CITATION.cff](CITATION.cff). Dataset access and
layout are described in [DATASET.md](DATASET.md).
