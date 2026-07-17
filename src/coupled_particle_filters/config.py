"""Strict, self-contained experiment configuration.

An experiment is described by exactly one YAML file.  There is deliberately no
defaults overlay or legacy alias layer: the YAML validated here is also the
configuration copied into the result directory.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Device(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class PipelineMode(str, Enum):
    PARTICLE_FILTER = "particle_filter"
    RAW_SAMPLED = "raw_sampled"


class Estimator(str, Enum):
    HAP = "hap"
    GRASPNET = "graspnet"


class AlgorithmProfile(str, Enum):
    MODERN = "modern"
    PAPER_MULTIPLY_2025 = "paper_multiply_2025"


class FusionStrategy(str, Enum):
    MULTIPLY = "multiply"
    PAPER_MULTIPLY = "paper_multiply"
    ADDITION = "addition"
    ALTERNATING = "alternating_add_multiply"
    CLIP_MULTIPLY = "clip_multiply"
    CLIP_ADDITION = "clip_addition"
    SMOOTH_CLIP_MULTIPLY = "smooth_clip_multiply"
    SMOOTH_CLIP_ADDITION = "smooth_clip_addition"
    SMOOTH_CLIP_ALTERNATING = "smooth_clip_alternating"


class RuntimeConfig(StrictModel):
    seed: int
    deterministic: bool
    device: Device
    dtype: Literal["float32", "float64"]


class TopicsConfig(StrictModel):
    color: list[str]
    depth: list[str]
    camera_info: list[str]


class ImageConfig(StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    max_depth_m: float = Field(gt=0)
    depth_scale_16u: float = Field(gt=0)
    invalid_depth_policy: Literal["max_depth", "error"]


class InputConfig(StrictModel):
    source: Literal["rosbag2"]
    dataset_root: Path
    interactions_csv: Path
    bag_paths: list[Path]
    categories: list[str]
    max_frames_per_bag: int | None = Field(default=None, gt=0)
    exclude_path_patterns: list[str]
    topics: TopicsConfig
    image: ImageConfig


class FarnebackConfig(StrictModel):
    pyr_scale: float
    levels: int = Field(gt=0)
    winsize: int = Field(gt=0)
    iterations: int = Field(gt=0)
    poly_n: int = Field(gt=0)
    poly_sigma: float = Field(gt=0)
    flags: int


class OpticalFlowConfig(StrictModel):
    source: Literal["precomputed", "farneback", "zero"]
    artifact_root: Path
    missing_policy: Literal["error", "zero"]
    minimum_motion_px: float = Field(ge=0)
    farneback: FarnebackConfig


class HapModelConfig(StrictModel):
    model_dir: Path
    model_name: str
    checkpoint_epoch: int
    grasp_info: Path
    image_size: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    mask_location: str
    slack: int = Field(ge=0)
    scales: list[int]
    hand_conditioned: bool
    mask_input: bool
    symmetric_encoder_decoder: bool
    two_heads: bool
    loss_masking: bool


class GraspNetModelConfig(StrictModel):
    checkpoint_dir: Path
    forward_passes: int = Field(gt=0)
    local_regions: bool
    skip_border_objects: bool
    filter_grasps: bool
    z_range_m: tuple[float, float]
    all_points: bool
    scale_heatmap: bool
    selected_threshold: float = Field(ge=0, le=1)
    argument_overrides: list[str]


class FilterConfig(StrictModel):
    num_particles: int = Field(gt=0)
    queue_length: int = Field(gt=0)
    readiness_policy: Literal["when_full", "legacy_extra_frame"]
    flow_coordinates: Literal["row_column", "legacy_xy_as_row_column"]
    initialize_from_measurement: bool
    initialization_clip: tuple[float, float]
    motion_noise_std: float = Field(ge=0)
    # Pixel-space flow-magnitude threshold for the 2D filter's prediction step.
    # The default 0.0 defers to optical_flow.minimum_motion_px, matching previous
    # effective behavior; the paper profile always uses its historical 1.0 px.
    motion_threshold_px: float = Field(default=0.0, ge=0)
    max_particle_movement_m: float = Field(ge=0)
    injection_count: int = Field(ge=0)
    injection_draw_count: int | None = Field(default=None, gt=0)
    injection_warmup: tuple[int, int] | None = None
    injection_density_sigma: float = Field(gt=0)
    injection_mu_offset: float
    injection_sigma_offset: float = Field(gt=0)
    measurement_neighbor_radius_m: float = Field(gt=0)
    measurement_distance_sigma: float = Field(gt=0)
    orientation_weight: float = Field(ge=0, le=1)
    occlusion_kernel_size: int = Field(gt=0)
    occlusion_edge_max_diff_m: float = Field(ge=0)
    occlusion_near_object_max_diff_m: float = Field(ge=0)
    image_border_px: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "FilterConfig":
        low, high = self.initialization_clip
        if not 0 <= low <= high:
            raise ValueError("initialization_clip must be ordered and non-negative")
        if self.occlusion_kernel_size % 2 == 0:
            raise ValueError("occlusion_kernel_size must be odd")
        if (
            self.injection_draw_count is not None
            and self.injection_draw_count
            < max(
                self.injection_count,
                self.injection_warmup[1] if self.injection_warmup else 0,
            )
        ):
            raise ValueError("injection_draw_count must cover steady and warmup counts")
        if self.injection_warmup is not None:
            frames, count = self.injection_warmup
            if frames <= 0 or count < 0:
                raise ValueError("injection_warmup requires positive frames and non-negative count")
        return self


class PipelineConfig(StrictModel):
    name: str
    estimator: Estimator
    mode: PipelineMode
    measurement_source: Literal["precomputed", "online"]
    artifact_root: Path
    raw_sample_count: int = Field(gt=0)
    filter: FilterConfig | None
    hap_model: HapModelConfig | None
    graspnet_model: GraspNetModelConfig | None

    @model_validator(mode="after")
    def validate_pipeline(self) -> "PipelineConfig":
        if self.mode == PipelineMode.PARTICLE_FILTER and self.filter is None:
            raise ValueError("particle_filter mode requires filter settings")
        if self.estimator == Estimator.HAP and self.hap_model is None:
            raise ValueError("HAP pipeline requires hap_model settings")
        if self.estimator == Estimator.GRASPNET and self.graspnet_model is None:
            raise ValueError("GraspNet pipeline requires graspnet_model settings")
        if (
            self.estimator == Estimator.HAP
            and self.filter is not None
            and self.filter.flow_coordinates != "row_column"
        ):
            raise ValueError("HAP filters require row_column optical-flow coordinates")
        return self


class WeightRange(StrictModel):
    measurement: tuple[float, float]
    cross_density: tuple[float, float]


class FusionConfig(StrictModel):
    enabled: bool
    strategy: FusionStrategy
    kernel_bandwidth_m: float = Field(gt=0)
    start_frame: int = Field(ge=0)
    every_n_frames: int = Field(gt=0)
    targets: Literal["both", "alternating"]
    alternating_period: int = Field(gt=0)
    alternating_multiply_phase: int = Field(ge=0)
    smooth_clip_steepness: float = Field(gt=0)
    graspnet_ranges: WeightRange
    hap_ranges: WeightRange


class NumericalConfig(StrictModel):
    """Numerical safety and distance implementation settings.

    ``fast`` validation is an opt-in GPU-oriented mode that replaces non-finite
    or negative weights with zero and falls back to uniform/unit weights instead
    of raising.  ``strict`` retains the paper and preset behavior.
    """

    normalization_epsilon: float = Field(gt=0)
    zero_mass_policy: Literal["error", "uniform"]
    nonfinite_policy: Literal["error", "zero"]
    distance_chunk_size: int = Field(gt=0)
    validation: Literal["strict", "fast"] = "strict"
    distance_backend: Literal["torch", "matmul"] = "torch"


class EvaluationConfig(StrictModel):
    enabled: bool
    data_root: Path
    distance_threshold_m: float = Field(gt=0)
    render: bool
    sample_count: int = Field(gt=0)
    sampling: Literal["uniform_with_replacement"]
    sampling_stage: Literal["post_resample"]
    seed: int = Field(ge=0)


class VideoConfig(StrictModel):
    enabled: bool
    frame_rate: int = Field(gt=0)
    output_rate: int = Field(gt=0)
    codec: str
    quality: int = Field(gt=0)


class OutputConfig(StrictModel):
    root: Path
    errors_dir: Path
    save_resolved_config: bool
    save_metrics: bool
    save_particles: bool
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    log_every_n_frames: int = Field(gt=0)
    render_enabled: bool
    render_every_n_frames: int = Field(gt=0)
    heatmap_kernel_ratio: int = Field(gt=0)
    video: VideoConfig


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    name: str
    description: str
    algorithm_profile: AlgorithmProfile = AlgorithmProfile.MODERN
    runtime: RuntimeConfig
    input: InputConfig
    optical_flow: OpticalFlowConfig
    pipelines: list[PipelineConfig]
    fusion: FusionConfig
    numerics: NumericalConfig
    evaluation: EvaluationConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_experiment(self) -> "ExperimentConfig":
        names = [pipeline.name for pipeline in self.pipelines]
        if not names or len(names) != len(set(names)):
            raise ValueError("pipeline names must be non-empty and unique")
        estimators = {pipeline.estimator for pipeline in self.pipelines}
        if self.fusion.enabled:
            if estimators != {Estimator.HAP, Estimator.GRASPNET} or len(self.pipelines) != 2:
                raise ValueError("fusion requires exactly one HAP and one GraspNet pipeline")
            if any(p.mode != PipelineMode.PARTICLE_FILTER for p in self.pipelines):
                raise ValueError("fusion requires particle_filter mode for both pipelines")
        return self


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config and resolve every relative path from its declared root."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = ExperimentConfig.model_validate(raw)
    base = config_path.parent

    def resolve(value: Path) -> Path:
        expanded = Path(os.path.expandvars(str(value))).expanduser()
        return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()

    config.input.dataset_root = resolve(config.input.dataset_root)
    config.input.interactions_csv = resolve(config.input.interactions_csv)
    config.input.bag_paths = [resolve(path) for path in config.input.bag_paths]
    config.optical_flow.artifact_root = resolve(config.optical_flow.artifact_root)
    config.evaluation.data_root = resolve(config.evaluation.data_root)
    config.output.root = resolve(config.output.root)
    config.output.errors_dir = resolve(config.output.errors_dir)
    for pipeline in config.pipelines:
        pipeline.artifact_root = resolve(pipeline.artifact_root)
        if pipeline.hap_model:
            pipeline.hap_model.model_dir = resolve(pipeline.hap_model.model_dir)
            pipeline.hap_model.grasp_info = resolve(pipeline.hap_model.grasp_info)
        if pipeline.graspnet_model:
            pipeline.graspnet_model.checkpoint_dir = resolve(pipeline.graspnet_model.checkpoint_dir)
    return config


def dump_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
