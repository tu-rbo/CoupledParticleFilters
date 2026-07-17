"""Offline experiment assembly and result sinks."""

from __future__ import annotations

import json
import hashlib
import csv
import fnmatch
import importlib
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .adapters import (
    OpticalFlowProvider,
    artifact_context,
    build_measurement_provider,
    iter_rosbag_frames,
)
from .bag_io import CorruptRosbagError
from .config import (
    AlgorithmProfile,
    Estimator,
    ExperimentConfig,
    PipelineConfig,
    dump_resolved_config,
)
from .engine import ExperimentEngine
from .math import clamp_image_points, project_xyz
from .pipelines import GraspNetPipeline, HapPipeline
from .types import Belief2D, Belief3D, Frame, StepResult


LOGGER = logging.getLogger(__name__)


def resolve_device(config: ExperimentConfig) -> torch.device:
    requested = config.runtime.device.value
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configuration requests CUDA, but CUDA is unavailable")
    return torch.device(requested)


def configure_runtime(config: ExperimentConfig) -> tuple[torch.device, torch.dtype]:
    if config.runtime.deterministic:
        # CUDA >= 10.2 requires a fixed cuBLAS workspace configuration before
        # deterministic matrix operations such as torch.cdist are first used.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)
    torch.manual_seed(config.runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.runtime.seed)
    torch.use_deterministic_algorithms(config.runtime.deterministic, warn_only=False)
    device = resolve_device(config)
    dtype = torch.float32 if config.runtime.dtype == "float32" else torch.float64
    return device, dtype


def discover_bags(config: ExperimentConfig) -> list[Path]:
    if config.input.bag_paths:
        return config.input.bag_paths
    bags = sorted(config.input.dataset_root.rglob("*.db3"))
    if config.input.categories:
        categories = set(config.input.categories)
        selected_instances: set[str] = set()
        if config.input.interactions_csv.exists():
            with config.input.interactions_csv.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("Name") in categories or row.get("Object") in categories:
                        if row.get("Name"):
                            selected_instances.add(row["Name"])
        bags = [
            bag
            for bag in bags
            if bag.parent.parent.name in categories
            or any(bag.parent.name.startswith(category) for category in categories)
            or any(bag.parent.name.startswith(instance) for instance in selected_instances)
        ]
    included = []
    for bag in bags:
        relative = str(bag.relative_to(config.input.dataset_root))
        pattern = next(
            (
                pattern
                for pattern in config.input.exclude_path_patterns
                if fnmatch.fnmatch(relative, pattern)
            ),
            None,
        )
        if pattern:
            LOGGER.warning("Skipping excluded ROS bag %s (matched %s)", bag, pattern)
        else:
            included.append(bag)
    bags = included
    if not bags:
        raise FileNotFoundError(
            f"no ROS2 bags selected under {config.input.dataset_root}; set input.bag_paths or categories"
        )
    return bags


def build_engine_for_bag(
    config: ExperimentConfig,
    bag_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> ExperimentEngine:
    category, instance = artifact_context(bag_path)
    image_size = (config.input.image.height, config.input.image.width)
    pipelines = []
    for pipeline_index, pipeline_config in enumerate(config.pipelines):
        if config.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
            pipeline_generator = generator
        else:
            pipeline_generator = torch.Generator(device="cpu").manual_seed(
                generator.initial_seed() + pipeline_index
            )
        artifact_directory = resolve_measurement_artifact_directory(
            pipeline_config, category, instance
        )
        provider = build_measurement_provider(
            pipeline_config, artifact_directory, device, dtype
        )
        if pipeline_config.estimator == Estimator.HAP:
            pipeline = HapPipeline(
                pipeline_config,
                config.numerics,
                image_size,
                provider,
                pipeline_generator,
                config.algorithm_profile,
                config.input.image.max_depth_m,
            )
        else:
            pipeline = GraspNetPipeline(
                pipeline_config,
                config.numerics,
                provider,
                pipeline_generator,
                config.input.image.max_depth_m,
                config.algorithm_profile,
            )
        pipelines.append(pipeline)
    return ExperimentEngine(config, pipelines, generator)


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    _validate_optional_dependencies(config)
    bags = discover_bags(config)
    _validate_precomputed_artifacts(config, bags)
    device, dtype = configure_runtime(config)
    run_root = config.output.root / config.name
    run_root.mkdir(parents=True, exist_ok=True)
    config.output.errors_dir.mkdir(parents=True, exist_ok=True)
    if config.output.save_resolved_config:
        dump_resolved_config(config, run_root / "resolved_config.yml")
    _write_metadata(config, run_root, device)

    summaries: dict[str, object] = {}
    skipped: dict[str, str] = {}
    failures: dict[str, str] = {}
    paper_generator = (
        torch.Generator(device="cpu").manual_seed(config.runtime.seed)
        if config.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025
        else None
    )
    for bag_path in bags:
        LOGGER.info("Running %s", bag_path)
        try:
            bag_offset = int.from_bytes(
                hashlib.sha256(str(bag_path).encode("utf-8")).digest()[:4], "big"
            )
            generator = paper_generator or torch.Generator(device="cpu").manual_seed(
                config.runtime.seed + bag_offset
            )
            summaries[str(bag_path)] = run_bag(
                config, bag_path, run_root, device, dtype, generator
            )
        except CorruptRosbagError as exc:
            message = str(exc)
            LOGGER.info("Skipping corrupted ROS bag %s: %s", bag_path, message)
            skipped[str(bag_path)] = message
            error_path = config.output.errors_dir / f"{bag_path.stem}.txt"
            error_path.write_text(f"CorruptRosbagError: {message}\n", encoding="utf-8")
        except Exception as exc:
            LOGGER.exception("Non-corruption failure while processing %s", bag_path)
            failures[str(bag_path)] = str(exc)
            error_path = config.output.errors_dir / f"{bag_path.stem}.txt"
            error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    result = {"bags": summaries, "skipped_corrupt_bags": skipped, "failures": failures}
    if config.output.save_metrics:
        (run_root / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    if failures:
        raise RuntimeError(f"{len(failures)} bag(s) failed; see {config.output.errors_dir}")
    return result


def _validate_optional_dependencies(config: ExperimentConfig) -> None:
    if not config.evaluation.enabled:
        return
    try:
        importlib.import_module("evaluate_particles")
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise RuntimeError(
            "evaluation.enabled is true, but evaluation cannot be imported; "
            f"missing module: {missing}. Install the evaluation environment or set "
            "evaluation.enabled: false. This is not a corrupted-bag error."
        ) from exc


def _validate_precomputed_artifacts(
    config: ExperimentConfig, bag_paths: list[Path]
) -> None:
    missing: list[str] = []
    for bag_path in bag_paths:
        category, instance = artifact_context(bag_path)
        if config.optical_flow.source == "precomputed":
            directory = resolve_flow_artifact_directory(
                config.optical_flow.artifact_root, category, instance
            )
            if not directory.exists() or not any(directory.glob("*.npy")):
                missing.append(f"optical flow: {directory}")
        for pipeline in config.pipelines:
            if pipeline.measurement_source != "precomputed":
                continue
            directory = resolve_measurement_artifact_directory(
                pipeline, category, instance
            )
            pattern = (
                "*_heatmap*.npy" if pipeline.estimator == Estimator.HAP else "*_unprocessed_output.npy"
            )
            if not directory.exists() or not any(directory.glob(pattern)):
                missing.append(f"{pipeline.name}: {directory}/{pattern}")
    if missing:
        details = "\n  - ".join(dict.fromkeys(missing))
        raise RuntimeError(
            "precomputed experiment artifacts are missing:\n  - "
            f"{details}\nGenerate the artifacts first, or select online/Farneback/zero sources "
            "in the experiment YAML. This is not a corrupted-bag error."
        )


def run_bag(
    config: ExperimentConfig,
    bag_path: Path,
    run_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> dict[str, object]:
    category, instance = artifact_context(bag_path)
    bag_root = run_root / category / instance
    bag_root.mkdir(parents=True, exist_ok=True)
    engine = build_engine_for_bag(config, bag_path, device, dtype, generator)
    flow = OpticalFlowProvider(
        config,
        resolve_flow_artifact_directory(
            config.optical_flow.artifact_root, category, instance
        ),
        device,
    )
    bag_seed_offset = generator.initial_seed() - config.runtime.seed
    evaluation_generator = torch.Generator(device="cpu").manual_seed(
        config.evaluation.seed + bag_seed_offset
    )
    sink = ResultSink(config, bag_root, instance, evaluation_generator)
    ready_frames = fused_frames = total_frames = 0
    try:
        for frame in iter_rosbag_frames(bag_path, config, flow, device, dtype):
            result = engine.step(frame)
            total_frames += 1
            ready_frames += int(result.ready)
            fused_frames += int(result.fused)
            sink.consume(frame, result)
            if (
                config.input.max_frames_per_bag is not None
                and total_frames >= config.input.max_frames_per_bag
            ):
                break
    finally:
        sink.close()
    return {
        "frames": total_frames,
        "ready_frames": ready_frames,
        "fused_frames": fused_frames,
        "metrics": sink.metrics,
    }


def resolve_measurement_artifact_directory(
    pipeline: PipelineConfig, category: str, instance: str
) -> Path:
    pattern = (
        "*_heatmap*.npy" if pipeline.estimator == Estimator.HAP else "*_unprocessed_output.npy"
    )
    canonical = pipeline.artifact_root / category / instance
    candidates = (
        canonical,
        canonical / instance,
        pipeline.artifact_root / instance,
        pipeline.artifact_root / f"{instance}_o_ros2" / instance,
    )
    return _select_artifact_directory(candidates, pattern, canonical)


def resolve_flow_artifact_directory(
    root: Path, category: str, instance: str
) -> Path:
    canonical = root / category / instance
    candidates = (
        canonical,
        root / instance,
        root / f"{instance}_o_ros2" / instance,
    )
    return _select_artifact_directory(candidates, "*.npy", canonical)


def _select_artifact_directory(
    candidates: tuple[Path, ...], pattern: str, canonical: Path
) -> Path:
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob(pattern)):
            if candidate != canonical:
                LOGGER.warning("Using legacy artifact layout %s", candidate)
            return candidate
    return canonical


class ResultSink:
    def __init__(
        self,
        config: ExperimentConfig,
        root: Path,
        instance: str,
        evaluation_generator: torch.Generator,
    ) -> None:
        self.config = config
        self.root = root
        self.instance = instance
        self.evaluation_generator = evaluation_generator
        self.metrics: dict[str, list[dict[str, object]]] = {
            pipeline.name: [] for pipeline in config.pipelines
        }
        self.paper_initialization_evaluated = False
        self.frames_directory = root / "frames"
        self.evaluator = self._build_evaluator() if config.evaluation.enabled else None

    def consume(self, frame: Frame, result: StepResult) -> None:
        if (
            self.evaluator is not None
            and self.config.algorithm_profile
            == AlgorithmProfile.PAPER_MULTIPLY_2025
            and result.initialization_beliefs
        ):
            self._evaluate_paper_initialization(frame, result.initialization_beliefs)
        if not result.ready:
            return
        if self.config.output.save_particles:
            self._save_particles(result)
        if self.evaluator is not None:
            self._evaluate(frame, result)
        if (
            self.config.output.render_enabled
            and result.frame_index % self.config.output.render_every_n_frames == 0
        ):
            self.frames_directory.mkdir(parents=True, exist_ok=True)
            if self.config.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
                image = render_paper_beliefs(
                    frame,
                    result,
                    self.config.output.heatmap_kernel_ratio,
                )
            else:
                image = render_beliefs(
                    frame,
                    result,
                    self.config.output.heatmap_kernel_ratio,
                )
            cv2.imwrite(
                str(self.frames_directory / f"frame_{result.frame_index:08d}.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
        if result.frame_index % self.config.output.log_every_n_frames == 0:
            LOGGER.info(
                "frame=%d timestamp=%d fused=%s",
                result.frame_index,
                result.timestamp_ns,
                result.fused,
            )

    def close(self) -> None:
        if self.config.output.video.enabled and self.frames_directory.exists():
            video = self.config.output.video
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(video.frame_rate),
                    "-pattern_type",
                    "glob",
                    "-i",
                    str(self.frames_directory / "frame_*.png"),
                    "-vcodec",
                    video.codec,
                    "-r",
                    str(video.output_rate),
                    "-q:v",
                    str(video.quality),
                    str(self.root / f"{self.instance}.mp4"),
                ],
                check=True,
            )
        if self.config.output.save_metrics:
            (self.root / "metrics.json").write_text(
                json.dumps(self.metrics, indent=2, sort_keys=True), encoding="utf-8"
            )

    def _build_evaluator(self):
        try:
            from evaluate_particles import Evaluator3D
        except ImportError as exc:
            raise RuntimeError("evaluation dependencies are not installed") from exc
        return Evaluator3D(
            object_bag_name=f"{self.instance}_o",
            path2data=str(self.config.evaluation.data_root),
            run_name=self.config.name,
        )

    def _evaluate(self, frame: Frame, result: StepResult) -> None:
        if self.config.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
            self._evaluate_paper(frame, result)
            return
        for name, belief in result.beliefs.items():
            particles = (
                belief.particles_rc if isinstance(belief, Belief2D) else belief.particles_xyz
            )
            if particles.shape[0] == 0:
                raise ValueError(f"cannot evaluate empty belief: {name}")
            indices = torch.randint(
                particles.shape[0],
                (self.config.evaluation.sample_count,),
                generator=self.evaluation_generator,
                device="cpu",
            ).to(particles.device)
            sampled_particles = particles[indices]
            self._evaluate_particles(name, sampled_particles, frame)

    def _evaluate_paper(self, frame: Frame, result: StepResult) -> None:
        if not self.paper_initialization_evaluated:
            # This fallback supports externally constructed engines/results.
            # The built-in engine normally reports the beliefs on their actual
            # initialization frame through ``initialization_beliefs``.
            self._evaluate_paper_initialization(frame, result.beliefs)

        sample = result.fused_graspnet_sample
        if sample is None:
            raise RuntimeError(
                "paper profile ready result is missing its fused GraspNet sample"
            )
        graspnet_name = next(
            pipeline.name
            for pipeline in self.config.pipelines
            if pipeline.estimator == Estimator.GRASPNET
        )
        self._evaluate_particles(graspnet_name, sample, frame)

    def _evaluate_paper_initialization(
        self, frame: Frame, beliefs: dict[str, Belief2D | Belief3D]
    ) -> None:
        if self.paper_initialization_evaluated:
            return
        for name, belief in beliefs.items():
            particles = (
                belief.particles_rc
                if isinstance(belief, Belief2D)
                else belief.particles_xyz
            )
            self._evaluate_particles(name, particles, frame)
        self.paper_initialization_evaluated = True

    def _evaluate_particles(
        self, name: str, particles: torch.Tensor, frame: Frame
    ) -> None:
        if particles.shape[0] == 0:
            raise ValueError(f"cannot evaluate empty belief: {name}")
        metrics = self.evaluator.evaluate(
            particles.detach().cpu(),
            frame.timestamp_ns / 1e9,
            frame.intrinsics.detach().cpu(),
            tresh=self.config.evaluation.distance_threshold_m,
            depth=frame.depth_m.detach().cpu(),
            render=self.config.evaluation.render,
        )
        if metrics:
            self.metrics[name].append(
                {key: _metric_value_to_json(value) for key, value in metrics.items()}
            )

    def _save_particles(self, result: StepResult) -> None:
        particles_root = self.root / "particles"
        particles_root.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for name, belief in result.beliefs.items():
            particles = (
                belief.particles_rc if isinstance(belief, Belief2D) else belief.particles_xyz
            )
            arrays[f"{name}_particles"] = particles.detach().cpu().numpy()
            arrays[f"{name}_weights"] = belief.weights.detach().cpu().numpy()
        np.savez_compressed(
            particles_root / f"{result.timestamp_ns}.npz",
            **arrays,
        )


def _metric_value_to_json(value: object) -> object:
    """Convert evaluator output, including nested per-mesh metrics, to JSON values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _metric_value_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metric_value_to_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported evaluation metric value: {type(value).__name__}")


def render_beliefs(frame: Frame, result: StepResult, kernel_ratio: int) -> np.ndarray:
    rgb = frame.rgb.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    kernel = max(1, int(np.sqrt(height * width) / kernel_ratio))
    kernel += 1 - kernel % 2
    panels = [_label_panel(rgb.copy(), "RGB")]
    combined = np.zeros((height, width), dtype=np.float32)
    for name, belief in result.beliefs.items():
        points = _belief_image_points(belief, frame)
        heat = _particle_heatmap(points, height, width, kernel)
        combined += heat
        panels.append(_label_panel(_overlay_heatmap(rgb, heat), name.upper()))
    panels.append(
        _label_panel(_overlay_heatmap(rgb, combined), "COMBINED")
    )
    return np.hstack(panels)


def render_paper_beliefs(
    frame: Frame, result: StepResult, kernel_ratio: int
) -> np.ndarray:
    """Render the April 2025 three-panel paper artifact.

    The first panel is the already-sampled 200-point fused GraspNet output, the
    second the full HAP belief, and the third the full GraspNet particle set.
    No sampling occurs here.
    """

    sample = result.fused_graspnet_sample
    if sample is None:
        raise ValueError("paper renderer requires the fused GraspNet sample")
    hap = next(
        (belief for belief in result.beliefs.values() if isinstance(belief, Belief2D)),
        None,
    )
    graspnet = next(
        (belief for belief in result.beliefs.values() if isinstance(belief, Belief3D)),
        None,
    )
    if hap is None or graspnet is None:
        raise ValueError("paper renderer requires both HAP and GraspNet beliefs")

    rgb = frame.rgb.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    intrinsics = frame.intrinsics.detach().cpu()
    sample_points = _paper_project_clamped(sample.detach().cpu(), intrinsics, height, width)
    hap_points = clamp_image_points(hap.particles_rc.detach().cpu(), height, width)
    grasp_points = _paper_project_clamped(
        graspnet.particles_xyz.detach().cpu(), intrinsics, height, width
    )

    sample_heat = _paper_particle_heatmap(
        sample_points, height, width, kernel_ratio
    )
    hap_heat = _paper_particle_heatmap(hap_points, height, width, kernel_ratio)
    grasp_heat = _paper_particle_heatmap(grasp_points, height, width, kernel_ratio)
    sample_panel = _paper_overlay_heatmap(rgb, sample_heat)
    hap_panel = _paper_overlay_heatmap(rgb, hap_heat)

    grasp_panel = _paper_overlay_heatmap(rgb, grasp_heat)
    return np.hstack((sample_panel, hap_panel, grasp_panel))


def _paper_project_clamped(
    points_xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    z = points_xyz[:, 2].clamp_min(torch.finfo(points_xyz.dtype).eps)
    column = intrinsics[0, 0] * points_xyz[:, 0] / z + intrinsics[0, 2]
    row = intrinsics[1, 1] * points_xyz[:, 1] / z + intrinsics[1, 2]
    return clamp_image_points(torch.stack((row, column), dim=1), height, width)


def _paper_particle_heatmap(
    points: torch.Tensor, height: int, width: int, kernel_ratio: int
) -> np.ndarray:
    heat = np.zeros((height, width), dtype=np.float32)
    points = clamp_image_points(points, height, width).long().numpy()
    # Advanced assignment intentionally overwrites duplicates instead of
    # accumulating them.
    heat[points[:, 0], points[:, 1]] = 1
    kernel = max(1, int(np.sqrt(height * width) / kernel_ratio))
    kernel += 1 - kernel % 2
    sigma = 0.3 * (((kernel - 1) * 0.5) - 1) + 0.8
    heat = cv2.GaussianBlur(
        heat,
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def _paper_overlay_heatmap(rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    colored = cv2.cvtColor(
        cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB,
    )
    return (0.65 * colored + 0.35 * rgb).astype(np.uint8)


def _belief_image_points(belief, frame: Frame) -> torch.Tensor:
    if isinstance(belief, Belief2D):
        return belief.particles_rc.detach().cpu()
    particles = belief.particles_xyz.detach().cpu()
    positive_depth = particles[:, 2] > 0
    return project_xyz(
        particles[positive_depth], frame.intrinsics.detach().cpu()
    )


def _particle_heatmap(
    points: torch.Tensor, height: int, width: int, kernel: int
) -> np.ndarray:
    heat = np.zeros((height, width), dtype=np.float32)
    points = points.long()
    valid = (
        (points[:, 0] >= 0)
        & (points[:, 0] < height)
        & (points[:, 1] >= 0)
        & (points[:, 1] < width)
    )
    points = points[valid]
    np.add.at(heat, (points[:, 0].numpy(), points[:, 1].numpy()), 1)
    heat = cv2.GaussianBlur(heat, (kernel, kernel), 0)
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def _overlay_heatmap(rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
    normalized = heat / heat.max() if heat.max() > 0 else heat
    colored = cv2.cvtColor(
        cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB,
    )
    return (0.35 * rgb + 0.65 * colored).astype(np.uint8)


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (180, 32), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _write_metadata(config: ExperimentConfig, root: Path, device: torch.device) -> None:
    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain=v1", "--untracked-files=normal")
    submodules = _git_output("submodule", "status", "--recursive")
    git_dirty = None if status is None else bool(status)
    status_lines = [] if not status else status.splitlines()
    submodule_lines = [] if not submodules else submodules.splitlines()
    metadata = {
        "experiment": config.name,
        "seed": config.runtime.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "git_commit": commit,
        "git_dirty": git_dirty,
        "git_status": status_lines,
        "git_submodules": submodule_lines,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _git_output(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
