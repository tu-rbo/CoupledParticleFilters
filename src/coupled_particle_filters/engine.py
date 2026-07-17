"""Transport-independent experiment engine."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .config import AlgorithmProfile, ExperimentConfig
from .fusion import CrossBeliefCoupler
from .math import sample_indices
from .pipelines import EstimatorPipeline
from .types import Belief2D, Belief3D, Frame, StepResult


class ExperimentEngine:
    """Run predict/update/couple/resample without knowing where frames came from."""

    def __init__(
        self,
        config: ExperimentConfig,
        pipelines: Iterable[EstimatorPipeline],
        generator: torch.Generator | None = None,
    ) -> None:
        self.config = config
        self.pipelines = list(pipelines)
        expected = [pipeline.name for pipeline in config.pipelines]
        actual = [pipeline.name for pipeline in self.pipelines]
        if actual != expected:
            raise ValueError(f"pipeline order/names {actual} do not match config {expected}")
        self.coupler = CrossBeliefCoupler(config.fusion, config.numerics)
        self.generator = generator
        if (
            self.generator is None
            and config.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025
            and self.pipelines
        ):
            self.generator = getattr(self.pipelines[0], "generator", None)
        self.frame_index = 0

    def step(self, frame: Frame) -> StepResult:
        beliefs = {}
        initialization_beliefs = {}
        for pipeline in self.pipelines:
            belief = pipeline.advance(frame)
            if belief is not None:
                beliefs[pipeline.name] = belief
            elif getattr(pipeline, "initialized_this_step", False):
                particle_filter = getattr(pipeline, "filter", None)
                initialized = getattr(particle_filter, "belief", None)
                if initialized is not None:
                    initialization_beliefs[pipeline.name] = initialized

        ready = len(beliefs) == len(self.pipelines)
        fused = False
        if ready and self.coupler.should_run(self.frame_index):
            hap = next((belief for belief in beliefs.values() if isinstance(belief, Belief2D)), None)
            graspnet = next(
                (belief for belief in beliefs.values() if isinstance(belief, Belief3D)), None
            )
            if hap is None or graspnet is None:
                raise RuntimeError("validated fusion experiment did not produce both belief types")
            self.coupler.apply(
                hap,
                graspnet,
                frame.intrinsics,
                self.frame_index,
                frame.depth_m,
            )
            fused = True

        fused_graspnet_sample = None
        if ready:
            for pipeline in self.pipelines:
                pipeline.resample()
            # Beliefs are mutable containers; resampling updates the returned views.
            if self.config.algorithm_profile == AlgorithmProfile.PAPER_MULTIPLY_2025:
                graspnet = next(
                    (
                        belief
                        for belief in beliefs.values()
                        if isinstance(belief, Belief3D)
                    ),
                    None,
                )
                if graspnet is None:
                    raise RuntimeError("paper profile requires a GraspNet belief")
                if self.generator is None:
                    raise RuntimeError("paper profile requires a shared random generator")
                uniform = torch.ones_like(graspnet.weights)
                indices = sample_indices(
                    uniform,
                    self.config.evaluation.sample_count,
                    self.config.numerics,
                    self.generator,
                )
                fused_graspnet_sample = graspnet.particles_xyz[indices]

        result = StepResult(
            frame_index=self.frame_index,
            timestamp_ns=frame.timestamp_ns,
            ready=ready,
            beliefs=beliefs,
            fused=fused,
            fused_graspnet_sample=fused_graspnet_sample,
            initialization_beliefs=initialization_beliefs,
        )
        self.frame_index += 1
        return result
