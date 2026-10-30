"""Typed configuration for the Model 1 research MVP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    frame_size: int
    clip_steps: int
    prediction_steps: int
    levels: tuple[str, ...]
    counts: tuple[int, int, int]
    quick_counts: tuple[int, int, int]
    max_attempts: int
    velocity_bounds: tuple[float, float]
    initial_state: dict[str, tuple[float, float]]
    occluder: dict[str, float]

    def __post_init__(self) -> None:
        if min(self.frame_size, self.clip_steps) < 1:
            raise ValueError("dataset dimensions must be positive")
        if self.prediction_steps < 2:
            raise ValueError("prediction_steps must be at least 2")
        if len(self.counts) != 3 or len(self.quick_counts) != 3:
            raise ValueError("dataset counts must contain train, validation, and test")
        if any(value < 0 for value in (*self.counts, *self.quick_counts, self.max_attempts)):
            raise ValueError("dataset counts must be nonnegative")
        if not self.levels:
            raise ValueError("at least one world level is required")
        if self.velocity_bounds[0] >= self.velocity_bounds[1]:
            raise ValueError("velocity bounds must increase")


@dataclass(frozen=True)
class ModelConfig:
    modes: int
    hidden_size: int
    frame_channels: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.modes < 1 or self.hidden_size < 1 or not self.frame_channels:
            raise ValueError("model dimensions must be positive")


@dataclass(frozen=True)
class LossConfig:
    trajectory_weight: float
    mode_weight: float
    state_weight: float
    pre_belief_weight: float
    post_belief_weight: float

    def weights(self) -> tuple[float, ...]:
        return (
            self.trajectory_weight,
            self.mode_weight,
            self.state_weight,
            self.pre_belief_weight,
            self.post_belief_weight,
        )

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.weights()):
            raise ValueError("loss weights must be nonnegative")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("training dimensions must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("invalid optimizer settings")


@dataclass(frozen=True)
class EvaluationConfig:
    calibration_radius: float
    calibration_bins: int
    mixture_sigma: float

    def __post_init__(self) -> None:
        if min(self.calibration_radius, self.mixture_sigma) <= 0:
            raise ValueError("evaluation radii must be positive")
        if self.calibration_bins < 1:
            raise ValueError("calibration bins must be positive")


@dataclass(frozen=True)
class Model1Config:
    schema_version: str
    dataset: DatasetConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def _pairs(values: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {key: (float(value[0]), float(value[1])) for key, value in values.items()}


def load_model1_config(path: Path) -> Model1Config:
    payload = yaml.safe_load(Path(path).read_text())
    dataset = dict(payload["dataset"])
    dataset["levels"] = tuple(dataset["levels"])
    dataset["counts"] = tuple(int(value) for value in dataset["counts"])
    dataset["quick_counts"] = tuple(int(value) for value in dataset["quick_counts"])
    dataset["velocity_bounds"] = tuple(float(value) for value in dataset["velocity_bounds"])
    dataset["initial_state"] = _pairs(dataset["initial_state"])
    dataset["occluder"] = {key: float(value) for key, value in dataset["occluder"].items()}
    model = dict(payload["model"])
    model["frame_channels"] = tuple(int(value) for value in model["frame_channels"])
    return Model1Config(
        schema_version=str(payload["schema_version"]),
        dataset=DatasetConfig(**dataset),
        model=ModelConfig(**model),
        loss=LossConfig(**payload["loss"]),
        training=TrainingConfig(**payload["training"]),
        evaluation=EvaluationConfig(**payload["evaluation"]),
    )
