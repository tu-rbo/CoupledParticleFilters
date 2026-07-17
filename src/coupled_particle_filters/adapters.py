"""Measurement, flow, and ROS bag adapters.

Heavy ROS/model dependencies are imported lazily so the core package remains
unit-testable on a CPU-only machine.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Generic, TypeVar

import cv2
import numpy as np
import torch

from .config import ExperimentConfig, PipelineConfig
from .bag_io import CorruptRosbagError, RGBDImageDecoder, ROS2BagReader
from .types import Frame, GraspMeasurement, HeatmapMeasurement


LoadedArtifact = TypeVar("LoadedArtifact")


class _PrefetchLoader(Generic[LoadedArtifact]):
    """Load the next timestamped artifact while the current frame is processed."""

    def __init__(
        self, files: list[Path], load: Callable[[Path], LoadedArtifact]
    ) -> None:
        self.files = files
        self._load = load
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpf-prefetch")
        self._future: Future[LoadedArtifact] | None = None
        self._future_index: int | None = None
        self._closed = False

    def get(self, index: int) -> LoadedArtifact:
        if not 0 <= index < len(self.files):
            raise IndexError(f"artifact index {index} is outside the available files")

        path = self.files[index]
        if self._future_index == index and self._future is not None:
            future = self._future
            self._future = None
            self._future_index = None
            try:
                loaded = future.result()
            except Exception:
                # Preserve the direct-load behavior if a queued task failed or
                # was cancelled before it reached the single worker.
                loaded = self._load(path)
        else:
            loaded = self._load(path)

        self._schedule_next(index)
        return loaded

    def _schedule_next(self, index: int) -> None:
        next_index = index + 1
        if next_index >= len(self.files) or self._future_index == next_index:
            return
        if self._future is not None:
            self._future.cancel()
        self._future = self._executor.submit(self._load, self.files[next_index])
        self._future_index = next_index

    def close(self) -> None:
        if not self._closed:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _cpu_tensor(
    values: np.ndarray, dtype: torch.dtype, *, pin_memory: bool
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=dtype)
    if pin_memory:
        try:
            return tensor.pin_memory()
        except RuntimeError:
            # CPU-only test environments cannot allocate CUDA-pinned memory.
            pass
    return tensor


def _load_hap_artifact(
    path: Path, dtype: torch.dtype, *, pin_memory: bool
) -> torch.Tensor:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    if isinstance(data, dict):
        data = np.stack([data[key] for key in sorted(data)]).mean(0)
    return _cpu_tensor(np.asarray(data).squeeze(), dtype, pin_memory=pin_memory)


def _load_graspnet_artifact(
    path: Path, dtype: torch.dtype, *, pin_memory: bool
) -> GraspMeasurement:
    loaded = np.load(path, allow_pickle=True)
    raw = loaded
    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        if loaded.shape and loaded.size == 2:
            raw = loaded[1]
        elif loaded.shape == ():
            raw = loaded.item()
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()
    if not isinstance(raw, dict):
        raise ValueError(f"unsupported GraspNet artifact structure in {path}")
    return GraspMeasurement(
        _cpu_tensor(raw["cntct_pts"], dtype, pin_memory=pin_memory),
        _cpu_tensor(raw["predicted_grasps"], dtype, pin_memory=pin_memory),
        _cpu_tensor(raw["scores"], dtype, pin_memory=pin_memory),
    )


def _load_optical_flow_artifact(path: Path, *, pin_memory: bool) -> torch.Tensor:
    # Stored flow is x/y in the legacy artifacts; the engine uses row/column.
    values = np.asarray(np.load(path))[..., [1, 0]]
    return _cpu_tensor(values, torch.float32, pin_memory=pin_memory)


class PrecomputedHapProvider:
    def __init__(self, directory: Path, device: torch.device, dtype: torch.dtype) -> None:
        self.directory = directory
        self.device = device
        self.dtype = dtype
        self.files = _timestamped_files(directory, ("*_heatmap.npy", "*_heatmaps.npy"))
        self._loader = _PrefetchLoader(
            [path for _, path in self.files],
            lambda path: _load_hap_artifact(
                path, dtype, pin_memory=device.type == "cuda" and torch.cuda.is_available()
            ),
        )

    def get(self, frame: Frame) -> HeatmapMeasurement:
        index = _nearest_file_index(self.files, frame.timestamp_ns, "HAP heatmap")
        values = self._loader.get(index).to(device=self.device, non_blocking=True)
        return HeatmapMeasurement(values)


class PrecomputedGraspNetProvider:
    def __init__(self, directory: Path, device: torch.device, dtype: torch.dtype) -> None:
        self.directory = directory
        self.device = device
        self.dtype = dtype
        self.files = _timestamped_files(directory, ("*_unprocessed_output.npy",))
        self._loader = _PrefetchLoader(
            [path for _, path in self.files],
            lambda path: _load_graspnet_artifact(
                path, dtype, pin_memory=device.type == "cuda" and torch.cuda.is_available()
            ),
        )

    def get(self, frame: Frame) -> GraspMeasurement:
        index = _nearest_file_index(self.files, frame.timestamp_ns, "GraspNet output")
        raw = self._loader.get(index)
        return GraspMeasurement(
            raw.points_xyz.to(device=self.device, non_blocking=True),
            raw.transforms.to(device=self.device, non_blocking=True),
            raw.scores.to(device=self.device, non_blocking=True),
        )


class OnlineHapProvider:
    def __init__(self, pipeline: PipelineConfig, device: torch.device, dtype: torch.dtype) -> None:
        from hotspots import hotspot_model

        model = pipeline.hap_model
        assert model is not None
        hotspot_model.hotspot_args.model_dir = str(model.model_dir)
        hotspot_model.hotspot_args.model_name = model.model_name
        hotspot_model.hotspot_args.ckpt = model.checkpoint_epoch
        hotspot_model.hotspot_args.grasp_info = str(model.grasp_info)
        hotspot_model.hotspot_args.imsize = model.image_size
        hotspot_model.hotspot_args.hand_cond = model.hand_conditioned
        hotspot_model.hotspot_args.mask_input = model.mask_input
        hotspot_model.hotspot_args.sym_encdec = model.symmetric_encoder_decoder
        hotspot_model.hotspot_args.two_heads = model.two_heads
        hotspot_model.hotspot_args.loss_masking = model.loss_masking
        self.model = hotspot_model.HAPModel(device=device)
        self.model.bs = model.batch_size
        self.model.mask_location = model.mask_location
        self.model.set_parameters({"scales": model.scales, "slack": model.slack})
        self.device = device
        self.dtype = dtype

    def get(self, frame: Frame) -> HeatmapMeasurement:
        values = self.model.infer(rgb=frame.rgb.detach().cpu().numpy().astype(np.uint8))
        return HeatmapMeasurement(values.to(device=self.device, dtype=self.dtype))


class OnlineGraspNetProvider:
    def __init__(self, pipeline: PipelineConfig, device: torch.device, dtype: torch.dtype) -> None:
        from pytorch_cgn.run_grasp import GraspNet

        model = pipeline.graspnet_model
        assert model is not None
        self.model = GraspNet(
            ckpt_dir=str(model.checkpoint_dir),
            forward_passes=model.forward_passes,
            arg_configs=model.argument_overrides,
            force_cpu=device.type == "cpu",
        )
        self.model.set_parameters(
            {
                "local_regions": model.local_regions,
                "skip_border_objects": model.skip_border_objects,
                "filter_grasps": model.filter_grasps,
                "z_range": list(model.z_range_m),
                "forward_passes": model.forward_passes,
                "all_pts": model.all_points,
                "scale_heatmap": model.scale_heatmap,
                "selected_threshold": model.selected_threshold,
            }
        )
        self.device = device
        self.dtype = dtype

    def get(self, frame: Frame) -> GraspMeasurement:
        _, raw = self.model.infer(
            frame.rgb.detach().cpu().numpy().astype(np.uint8),
            frame.depth_m.detach().cpu().numpy(),
            frame.intrinsics.detach().cpu().numpy(),
        )
        return GraspMeasurement(
            torch.as_tensor(raw["cntct_pts"], dtype=self.dtype, device=self.device),
            torch.as_tensor(raw["predicted_grasps"], dtype=self.dtype, device=self.device),
            torch.as_tensor(raw["scores"], dtype=self.dtype, device=self.device),
        )


def build_measurement_provider(
    pipeline: PipelineConfig,
    artifact_directory: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    if pipeline.measurement_source == "precomputed":
        if pipeline.estimator.value == "hap":
            return PrecomputedHapProvider(artifact_directory, device, dtype)
        return PrecomputedGraspNetProvider(artifact_directory, device, dtype)
    if pipeline.estimator.value == "hap":
        return OnlineHapProvider(pipeline, device, dtype)
    return OnlineGraspNetProvider(pipeline, device, dtype)


class OpticalFlowProvider:
    def __init__(
        self,
        config: ExperimentConfig,
        artifact_directory: Path,
        device: torch.device | None = None,
    ) -> None:
        self.config = config.optical_flow
        self.paper_profile = config.algorithm_profile.value == "paper_multiply_2025"
        self.artifact_directory = artifact_directory
        self.previous_rgb: np.ndarray | None = None
        self._flow_files: list[tuple[int, Path]] = []
        self._flow_indices: dict[int, int] = {}
        self._flow_loader: _PrefetchLoader[torch.Tensor] | None = None
        if self.config.source == "precomputed":
            self._flow_files = _timestamped_files(artifact_directory, ("*.npy",))
            self._flow_indices = {
                timestamp: index for index, (timestamp, _) in enumerate(self._flow_files)
            }
            pin_memory = (device is None or device.type == "cuda") and torch.cuda.is_available()
            self._flow_loader = _PrefetchLoader(
                [path for _, path in self._flow_files],
                lambda path: _load_optical_flow_artifact(path, pin_memory=pin_memory),
            )

    def get(self, timestamp_ns: int, rgb: torch.Tensor) -> torch.Tensor:
        height, width = rgb.shape[:2]
        if self.config.source == "zero":
            flow = torch.zeros((height, width, 2), dtype=torch.float32)
        elif self.config.source == "precomputed":
            path = self.artifact_directory / f"{timestamp_ns}.npy"
            index = self._flow_indices.get(timestamp_ns)
            if index is None and not path.exists():
                if self.config.missing_policy == "error":
                    raise FileNotFoundError(f"missing optical flow artifact: {path}")
                flow = torch.zeros((height, width, 2), dtype=torch.float32)
            else:
                if index is None:
                    flow = _load_optical_flow_artifact(
                        path, pin_memory=torch.cuda.is_available()
                    )
                else:
                    assert self._flow_loader is not None
                    flow = self._flow_loader.get(index)
        else:
            current = rgb.detach().cpu().numpy().astype(np.uint8)
            if self.previous_rgb is None:
                flow = torch.zeros((height, width, 2), dtype=torch.float32)
            else:
                parameters = self.config.farneback
                previous_gray = cv2.cvtColor(self.previous_rgb, cv2.COLOR_RGB2GRAY)
                current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
                xy = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    current_gray,
                    None,
                    parameters.pyr_scale,
                    parameters.levels,
                    parameters.winsize,
                    parameters.iterations,
                    parameters.poly_n,
                    parameters.poly_sigma,
                    parameters.flags,
                )
                flow = torch.as_tensor(xy[..., [1, 0]].copy(), dtype=torch.float32)
            self.previous_rgb = current
        if not self.paper_profile:
            flow[flow.norm(dim=2) < self.config.minimum_motion_px] = 0
        return flow


def iter_rosbag_frames(
    bag_path: Path,
    config: ExperimentConfig,
    flow_provider: OpticalFlowProvider,
    device: torch.device,
    dtype: torch.dtype,
) -> Iterator[Frame]:
    """Yield synchronized RGB/depth/intrinsics using the legacy bag decoder."""
    try:
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as exc:
        raise RuntimeError(
            "ROS bag playback requires ROS2 Python packages; install the 'ros' environment described in README.md"
        ) from exc

    reader = ROS2BagReader(bag_path)
    decoder = RGBDImageDecoder()
    rgb = depth = intrinsics = None
    all_topics = (
        config.input.topics.color
        + config.input.topics.depth
        + config.input.topics.camera_info
    )
    try:
        for topic, raw, timestamp in reader.read_messages(all_topics):
            if topic in config.input.topics.color:
                try:
                    message = deserialize_message(raw, Image)
                except Exception as exc:
                    raise CorruptRosbagError(
                        f"cannot deserialize color message on {topic} at {timestamp}: {exc}"
                    ) from exc
                rgb = decoder.imgmsg_to_cv2(message)
            elif topic in config.input.topics.depth:
                try:
                    message = deserialize_message(raw, Image)
                except Exception as exc:
                    raise CorruptRosbagError(
                        f"cannot deserialize depth message on {topic} at {timestamp}: {exc}"
                    ) from exc
                depth = decoder.depthmsg_to_cv2(message)
                if message.encoding == "16UC1":
                    depth *= config.input.image.depth_scale_16u / 0.001
            elif topic in config.input.topics.camera_info:
                try:
                    message = deserialize_message(raw, CameraInfo)
                except Exception as exc:
                    raise CorruptRosbagError(
                        f"cannot deserialize camera info on {topic} at {timestamp}: {exc}"
                    ) from exc
                values = message.k if hasattr(message, "k") else message.K
                intrinsics = torch.as_tensor(values, dtype=dtype).reshape(3, 3)
            if rgb is None or depth is None or intrinsics is None:
                continue
            depth = depth.to(dtype=dtype)
            invalid = ~torch.isfinite(depth) | (depth <= 0)
            if invalid.any():
                if config.input.image.invalid_depth_policy == "error":
                    raise ValueError(f"invalid depth values at timestamp {timestamp}")
                depth[invalid] = config.input.image.max_depth_m
            flow = flow_provider.get(timestamp, rgb).to(
                device=device, dtype=dtype, non_blocking=True
            )
            yield Frame(
                timestamp_ns=timestamp,
                rgb=rgb.to(device=device, dtype=dtype),
                depth_m=depth.to(device=device),
                intrinsics=intrinsics.to(device=device),
                flow_rc=flow,
            )
            rgb = depth = intrinsics = None
    finally:
        reader.close()


def artifact_context(bag_path: Path) -> tuple[str, str]:
    category = bag_path.parent.parent.name
    instance = bag_path.parent.name.split("_")[0]
    return category, instance


def _timestamped_files(directory: Path, patterns: tuple[str, ...]) -> list[tuple[int, Path]]:
    found: dict[int, Path] = {}
    for pattern in patterns:
        for path in directory.glob(pattern):
            try:
                timestamp = int(path.stem.split("_", maxsplit=1)[0])
            except ValueError:
                continue
            found[timestamp] = path
    return sorted(found.items())


def _nearest_file_index(files: list[tuple[int, Path]], timestamp: int, label: str) -> int:
    if not files:
        raise FileNotFoundError(f"no {label} artifacts found")
    timestamps = [item[0] for item in files]
    index = bisect_left(timestamps, timestamp)
    candidate_indices = range(max(0, index - 1), min(len(files), index + 1))
    return min(candidate_indices, key=lambda candidate: abs(files[candidate][0] - timestamp))


def _nearest_file(files: list[tuple[int, Path]], timestamp: int, label: str) -> Path:
    return files[_nearest_file_index(files, timestamp, label)][1]
