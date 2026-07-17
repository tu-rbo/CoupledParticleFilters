"""Precompute model measurements and optical flow from ROS bag data."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from coupled_particle_filters.project_config import ConfigNode, load_project_config


LOGGER = logging.getLogger(__name__)

SUPPORTED_MODELS = (
    "flow",
    "graspnet",
    "hap",
    "where2act",
    "hrp",
    "attention",
    "affordance",
)

OUTPUT_SUBDIRECTORIES = {
    "flow": "flow",
    "graspnet": "GraspNet",
    "hap": "hap_output",
    "where2act": "where2act_output",
    "hrp": "hrp_output",
    "attention": "attention_output",
    "affordance": "affordance_output",
}

ARTIFACT_SUFFIXES = {
    "flow": ".npy",
    "graspnet": "_unprocessed_output.npy",
    "hap": "_heatmap.npy",
    "where2act": "_unprocessed_output.npy",
    "hrp": "_heatmaps.npy",
    "attention": "_heatmaps.npy",
    "affordance": "_heatmaps.npy",
}

DEFAULT_COLOR_TOPICS = {
    "/camera/rgb/image_raw",
    "/camera/color/image_raw",
}
DEFAULT_DEPTH_TOPICS = {
    "/camera/depth_registered/image_raw",
    "/camera/depth/image_rect_raw",
    "/camera/aligned_depth_to_color/image_raw",
}
DEFAULT_CAMERA_INFO_TOPICS = {
    "/camera/rgb/camera_info",
    "/camera/color/camera_info",
    "/camera/depth/camera_info",
    "/camera/depth_registered/camera_info",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute flow and affordance-model outputs from ROS bags."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=(*SUPPORTED_MODELS, "all"),
        help="Artifacts to compute. Use 'all' to run every supported model.",
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--bagfile",
        nargs="+",
        type=Path,
        help="One or more ROS1 .bag files, ROS2 .db3 files, or ROS2 bag directories.",
    )
    selection.add_argument(
        "--categories",
        nargs="+",
        help="Dataset categories or instance names to discover, for example ikea or ikea03.",
    )
    selection.add_argument(
        "--all-bags",
        action="store_true",
        help="Process every non-backup bag under paths.rbo_dataset.",
    )

    parser.add_argument("--project-config", default="configs/project.yml")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override all configured output directories, for example data or /data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument(
        "--flow-method",
        choices=("farneback", "rlof"),
        default="farneback",
        help="Dense-flow algorithm. RLOF requires opencv-contrib-python.",
    )
    parser.add_argument("--forward-passes", type=_positive_int, default=1)
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        help="Stop after this many synchronized frames per bag (useful for smoke tests).",
    )
    parser.add_argument("--graspnet-checkpoint", type=Path)
    parser.add_argument(
        "--hap-adapter-module",
        default="hotspots.hotspot_model",
        help=(
            "Import path for a local HAP adapter exposing HAPModel/hotspot_args or "
            "build_hap_model(project_config, device). The adapter is not distributed "
            "with the public repository."
        ),
    )
    parser.add_argument("--in-hrp", "--in_hrp", dest="in_hrp", type=Path)
    parser.add_argument("--eg-hrp", "--eg_hrp", dest="eg_hrp", type=Path)
    parser.add_argument("--vit-model-name", "--vit_model_name", default="VC1_hrp")
    parser.add_argument("--where-ckpt", "--where_ckpt", dest="where_ckpt", type=Path)
    return parser


def selected_models(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with individual model names")
        return SUPPORTED_MODELS
    return tuple(dict.fromkeys(values))


def configure_output_root(project_config: ConfigNode, root: Path | None) -> None:
    """Point every output at a stable subdirectory below ``root`` when supplied."""
    if root is None:
        return
    root = root.expanduser().resolve()
    for model_name, subdirectory in OUTPUT_SUBDIRECTORIES.items():
        project_config.outputs[model_name] = str(root / subdirectory)


def output_root(project_config: ConfigNode, model_name: str) -> Path:
    return Path(project_config.outputs[model_name]).expanduser()


def output_dir(
    project_config: ConfigNode,
    model_name: str,
    category: str,
    instance: str,
) -> Path:
    return output_root(project_config, model_name) / category / instance


def artifact_path(
    project_config: ConfigNode,
    model_name: str,
    category: str,
    instance: str,
    timestamp: int | str,
) -> Path:
    return output_dir(project_config, model_name, category, instance) / (
        f"{timestamp}{ARTIFACT_SUFFIXES[model_name]}"
    )


def make_dirs(
    category: str,
    instance: str,
    model_names: tuple[str, ...],
    project_config: ConfigNode,
) -> None:
    for model_name in model_names:
        output_dir(project_config, model_name, category, instance).mkdir(
            parents=True, exist_ok=True
        )


def file_exists(
    model_name: str,
    category: str,
    instance: str,
    timestamp: int | str,
    project_config: ConfigNode,
) -> bool:
    """Return whether this model's artifact exists, independently of other models."""
    return artifact_path(
        project_config, model_name, category, instance, timestamp
    ).is_file()


def artifact_context(bagfile: str | Path) -> tuple[str, str]:
    path = Path(bagfile)
    bag_name = path.name if path.is_dir() else path.stem
    instance = bag_name.split("_", maxsplit=1)[0]
    parent = path if path.is_dir() else path.parent
    if parent.name.endswith("_ros2") or parent.name.endswith(".orig_ros2"):
        category = parent.parent.name
    else:
        category = parent.name
    return category, instance


def discover_bagfiles(dataset_root: str | Path, selectors: list[str] | None) -> list[Path]:
    root = Path(dataset_root).expanduser()
    candidates = sorted((*root.rglob("*.db3"), *root.rglob("*.bag")))
    selected = []
    requested = set(selectors or ())
    for path in candidates:
        if any(part.endswith(".orig_ros2") for part in path.parts):
            continue
        category, instance = artifact_context(path)
        if requested and category not in requested and instance not in requested:
            continue
        selected.append(path)
    if not selected:
        suffix = f" for {sorted(requested)}" if requested else ""
        raise FileNotFoundError(f"no ROS bags found below {root}{suffix}")
    return selected


def _reader_path(bagfile: str | Path) -> Path:
    path = Path(bagfile).expanduser().resolve()
    if path.suffix == ".db3":
        return path.parent
    return path


def _decode_color(message: Any) -> np.ndarray:
    encoding = str(message.encoding).lower()
    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
    if encoding == "mono8":
        channels = 1
    rows = np.asarray(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, : message.width * channels].reshape(
        message.height, message.width, channels
    )
    if encoding in {"bgr8", "bgra8"}:
        image = image[..., [2, 1, 0, *([3] if channels == 4 else [])]]
    if channels == 4:
        image = image[..., :3]
    elif channels == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape != (message.height, message.width, 3):
        raise ValueError(f"unsupported color encoding: {message.encoding}")
    return np.ascontiguousarray(image)


def _decode_depth(message: Any) -> np.ndarray:
    encoding = str(message.encoding).upper()
    dtypes = {"16UC1": np.dtype("uint16"), "32FC1": np.dtype("float32")}
    if encoding not in dtypes:
        raise ValueError(f"unsupported depth encoding: {message.encoding}")
    dtype = dtypes[encoding].newbyteorder(">" if message.is_bigendian else "<")
    row_values = message.step // dtype.itemsize
    depth = np.frombuffer(np.asarray(message.data, dtype=np.uint8), dtype=dtype).reshape(
        message.height, row_values
    )[:, : message.width]
    depth = depth.astype(np.float32)
    if encoding == "16UC1":
        depth /= 1000.0
    return np.ascontiguousarray(depth)


def _camera_intrinsics(message: Any) -> np.ndarray:
    values = message.k if hasattr(message, "k") else message.K
    return np.asarray(values, dtype=np.float32).reshape(3, 3)


def _numpy_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _numpy_value(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        if value.dtype == np.float64:
            return value.astype(np.float32)
        if value.dtype == object:
            converted = np.empty_like(value)
            for index in np.ndindex(value.shape):
                converted[index] = _numpy_value(value[index])
            return converted
        return value
    if isinstance(value, dict):
        return {key: _numpy_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        result = np.empty(len(value), dtype=object)
        result[:] = [_numpy_value(item) for item in value]
        return result
    if isinstance(value, list):
        return [_numpy_value(item) for item in value]
    return value


def save_artifact(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, _numpy_value(value), allow_pickle=True)


def get_model_by_name(
    name: str,
    args: argparse.Namespace,
    project_config: ConfigNode,
) -> Any:
    device = "cpu" if args.force_cpu else None
    if name == "flow":
        return None
    if name == "graspnet":
        try:
            grasp_module = importlib.import_module("pytorch_cgn.run_grasp")
        except ModuleNotFoundError as exc:
            if exc.name != "contact_graspnet_pytorch":
                raise
            nested_package = (
                Path(__file__).resolve().parent
                / "pytorch_cgn"
                / "contact_graspnet_pytorch"
            )
            if not nested_package.is_dir():
                raise RuntimeError(
                    "Contact-GraspNet is not installed. Run "
                    "'pip install -e pytorch_cgn/contact_graspnet_pytorch' first."
                ) from exc
            sys.path.insert(0, str(nested_package))
            grasp_module = importlib.import_module("pytorch_cgn.run_grasp")
        GraspNet = grasp_module.GraspNet

        checkpoint = args.graspnet_checkpoint or Path(
            project_config.model_paths.contact_graspnet_checkpoint
        )
        return GraspNet(
            ckpt_dir=str(checkpoint),
            forward_passes=args.forward_passes,
            arg_configs=[],
            force_cpu=args.force_cpu,
        )
    if name == "hap":
        try:
            hap_adapter = importlib.import_module(args.hap_adapter_module)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "HAP generation needs a separately obtained local adapter because the "
                "upstream hands-as-probes repository has no redistribution license. "
                "Install an authorized adapter and pass --hap-adapter-module, or use "
                "the released precomputed HAP heatmaps."
            ) from exc
        if hasattr(hap_adapter, "build_hap_model"):
            return hap_adapter.build_hap_model(project_config, device)
        if not hasattr(hap_adapter, "hotspot_args") or not hasattr(hap_adapter, "HAPModel"):
            raise RuntimeError(
                f"HAP adapter {args.hap_adapter_module!r} must expose either "
                "build_hap_model(project_config, device) or HAPModel/hotspot_args"
            )
        hap_adapter.hotspot_args.model_dir = str(project_config.model_paths.hap_model_dir)
        hap_adapter.hotspot_args.grasp_info = str(project_config.model_paths.hap_grasp_info)
        return hap_adapter.HAPModel(device=device)
    if name == "where2act":
        from other_baseline.where2act_adapter import Where2Act_Runner

        return Where2Act_Runner(force_cpu=args.force_cpu)
    if name in {"hrp", "attention"}:
        from data4robotics import load_vit
        from data4robotics.examples.explore_attention_maps import Agent

        if name == "hrp":
            for checkpoint in (
                args.in_hrp or Path(project_config.model_paths.hrp_in_checkpoint),
                args.eg_hrp or Path(project_config.model_paths.hrp_eg_checkpoint),
            ):
                if not Path(checkpoint).exists():
                    LOGGER.warning("HRP checkpoint is not present locally: %s", checkpoint)
            model_name = "IN_hrp"
        else:
            model_name = args.vit_model_name
        transform, model = load_vit(model_name)
        model.eval()
        return Agent(model=model, transform=transform)
    if name == "affordance":
        affordance_dir = Path(project_config.model_paths.affordance_dir).resolve()
        if str(affordance_dir) not in sys.path:
            sys.path.insert(0, str(affordance_dir))
        from affordance_predictor import AffordanceKeypointPredictor

        checkpoint = args.where_ckpt or (
            affordance_dir / "output" / "release" / "layout" / "checkpoints" / "last.ckpt"
        )
        return AffordanceKeypointPredictor(
            where_ckpt=str(checkpoint),
            device="cpu" if args.force_cpu else "cuda",
            num_samples=5,
            side_x=256,
        )
    raise ValueError(f"unknown model: {name}")


def _compute_flow(
    previous_rgb: np.ndarray | None,
    rgb: np.ndarray,
    method: str = "farneback",
) -> np.ndarray:
    import cv2

    if previous_rgb is None:
        return np.zeros((*rgb.shape[:2], 2), dtype=np.float32)
    if method == "farneback":
        previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        ).astype(np.float32)
    if not hasattr(cv2, "optflow"):
        raise RuntimeError(
            "RLOF flow requires opencv-contrib-python; use --flow-method farneback "
            "with the standard OpenCV package"
        )
    parameters = cv2.optflow.RLOFOpticalFlowParameter_create()
    parameters.setMaxIteration(30)
    parameters.setNormSigma0(3.2)
    parameters.setNormSigma1(7.0)
    parameters.setLargeWinSize(21)
    parameters.setSmallWinSize(15)
    parameters.setMaxLevel(9)
    parameters.setMinEigenValue(0.0001)
    parameters.setCrossSegmentationThreshold(25)
    parameters.setGlobalMotionRansacThreshold(10.0)
    parameters.setUseIlluminationModel(True)
    parameters.setUseGlobalMotionPrior(False)
    return cv2.optflow.calcOpticalFlowDenseRLOF(
        previous_rgb, rgb, None, rlofParam=parameters
    ).astype(np.float32)


def _run_image_model(
    model_name: str,
    model: Any,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> Any:
    if model_name == "graspnet":
        return model.infer(rgb, depth, intrinsics)
    if model_name == "hap":
        return model.infer(rgb=rgb, depth=depth, K=intrinsics)
    return model.infer(rgb)


def _build_where2act_evaluator(
    instance: str, project_config: ConfigNode
) -> Any:
    from evaluate_particles import Evaluator3D

    return Evaluator3D(
        object_bag_name=f"{instance}_o",
        run_name="precompute_where2act",
        path2data=project_config.paths.evaluation_data,
        precompute=False,
    )


def precompute_bag(
    bagfile: str | Path,
    model_names: tuple[str, ...],
    models: dict[str, Any],
    project_config: ConfigNode,
    *,
    overwrite: bool = False,
    max_frames: int | None = None,
    flow_method: str = "farneback",
) -> dict[str, dict[str, int]]:
    """Read one bag once and write every requested model artifact."""
    from rosbags.highlevel import AnyReader

    category, instance = artifact_context(bagfile)
    make_dirs(category, instance, model_names, project_config)
    stats = {name: {"saved": 0, "skipped": 0} for name in model_names}
    evaluator = (
        _build_where2act_evaluator(instance, project_config)
        if "where2act" in model_names
        else None
    )
    previous_rgb: np.ndarray | None = None
    rgb = depth = intrinsics = None
    frame_count = 0

    reader_path = _reader_path(bagfile)
    with AnyReader([reader_path]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic
            in (DEFAULT_COLOR_TOPICS | DEFAULT_DEPTH_TOPICS | DEFAULT_CAMERA_INFO_TOPICS)
        ]
        if not connections:
            raise RuntimeError(f"no supported RGB-D topics found in {bagfile}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            try:
                message = reader.deserialize(rawdata, connection.msgtype)
                if connection.topic in DEFAULT_COLOR_TOPICS:
                    rgb = _decode_color(message)
                elif connection.topic in DEFAULT_DEPTH_TOPICS:
                    depth = _decode_depth(message)
                elif connection.topic in DEFAULT_CAMERA_INFO_TOPICS:
                    intrinsics = _camera_intrinsics(message)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"cannot decode {connection.topic} at {timestamp} in {bagfile}: {exc}"
                ) from exc

            if rgb is None or depth is None or intrinsics is None:
                continue

            frame_count += 1
            for model_name in model_names:
                destination = artifact_path(
                    project_config, model_name, category, instance, timestamp
                )
                if destination.exists() and not overwrite:
                    stats[model_name]["skipped"] += 1
                    continue

                if model_name == "flow":
                    output = _compute_flow(previous_rgb, rgb, method=flow_method)
                elif model_name == "where2act":
                    from coupled_particle_filters.evaluation_support import depth2pcd

                    pointcloud = depth2pcd(
                        torch.from_numpy(depth), torch.from_numpy(intrinsics)
                    )
                    pointcloud = evaluator.get_current_mesh_volume(float(timestamp), pointcloud)
                    _, output = models[model_name].infer_on_pcd(pointcloud)
                else:
                    output = _run_image_model(
                        model_name, models[model_name], rgb, depth, intrinsics
                    )
                save_artifact(destination, output)
                stats[model_name]["saved"] += 1

            previous_rgb = rgb
            rgb = depth = intrinsics = None
            if max_frames is not None and frame_count >= max_frames:
                break

    if frame_count == 0:
        raise RuntimeError(f"no synchronized RGB-D frames found in {bagfile}")
    LOGGER.info("Processed %d RGB-D frames from %s", frame_count, bagfile)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        model_names = selected_models(args.models)
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    project_config = load_project_config(args.project_config)
    configure_output_root(project_config, args.output_root)

    if args.bagfile:
        bagfiles = [path.expanduser() for path in args.bagfile]
    else:
        selectors = None if args.all_bags else args.categories
        bagfiles = discover_bagfiles(project_config.paths.rbo_dataset, selectors)

    models = {
        name: get_model_by_name(name, args, project_config)
        for name in model_names
        if name != "flow"
    }

    failures: list[tuple[Path, Exception]] = []
    totals = {name: {"saved": 0, "skipped": 0} for name in model_names}
    for bagfile in bagfiles:
        try:
            stats = precompute_bag(
                bagfile,
                model_names,
                models,
                project_config,
                overwrite=args.overwrite,
                max_frames=args.max_frames,
                flow_method=args.flow_method,
            )
            for name, counts in stats.items():
                for key, value in counts.items():
                    totals[name][key] += value
        except Exception as exc:  # Continue the selected dataset and summarize failures.
            LOGGER.exception("Failed to precompute %s", bagfile)
            failures.append((Path(bagfile), exc))

    for name, counts in totals.items():
        LOGGER.info(
            "%s: saved=%d skipped=%d", name, counts["saved"], counts["skipped"]
        )
    if failures:
        LOGGER.error("%d bag(s) failed", len(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
