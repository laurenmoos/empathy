"""Streaming predict-record-reveal-update training for Model 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from perception_env.families import load_family, sample_worlds
from perception_env.simulate import BallState

from .config import Model1Config
from .data import Model1ExampleRecord, make_model1_record


@dataclass(frozen=True)
class StreamTrainingConfig:
    schema_version: str
    levels: tuple[str, ...]
    environment_count: int
    episodes_per_environment: int
    max_attempts_per_environment: int
    replay_capacity: int
    replay_batch_size: int
    update_steps_per_environment: int

    def __post_init__(self) -> None:
        if self.schema_version != "model1-stream-v1":
            raise ValueError("unsupported Model 1 stream schema")
        values = (
            self.environment_count,
            self.episodes_per_environment,
            self.max_attempts_per_environment,
            self.replay_capacity,
            self.replay_batch_size,
            self.update_steps_per_environment,
        )
        if not self.levels or min(values) < 1:
            raise ValueError("stream training values and levels must be positive")
        if self.replay_batch_size > self.replay_capacity:
            raise ValueError("replay batch size cannot exceed replay capacity")


def load_stream_training_config(path: Path) -> StreamTrainingConfig:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "stream"}:
        raise ValueError("stream configuration must contain schema_version and stream")
    values = dict(payload["stream"])
    expected = {
        "levels",
        "environment_count",
        "episodes_per_environment",
        "max_attempts_per_environment",
        "replay_capacity",
        "replay_batch_size",
        "update_steps_per_environment",
    }
    if set(values) != expected:
        raise ValueError("stream configuration fields are unsupported or missing")
    values["levels"] = tuple(str(level) for level in values["levels"])
    return StreamTrainingConfig(schema_version=str(payload["schema_version"]), **values)


@dataclass(frozen=True, eq=False)
class EnvironmentBatch:
    environment_index: int
    level: str
    world_id: str
    records: tuple[Model1ExampleRecord, ...]
    rejection_counts: dict[str, int]


def _sample_initial(config: Model1Config, rng: np.random.Generator) -> BallState:
    ranges = config.dataset.initial_state
    return BallState(
        *(float(rng.uniform(*ranges[name])) for name in ("x", "y", "vx", "vy"))
    )


def generate_environment_batches(
    config: Model1Config,
    stream_config: StreamTrainingConfig,
    family_config: Path,
    *,
    seed: int,
) -> tuple[EnvironmentBatch, ...]:
    batches: list[EnvironmentBatch] = []
    for environment_index in range(stream_config.environment_count):
        level = stream_config.levels[environment_index % len(stream_config.levels)]
        family = load_family(family_config, level)
        environment_seed = seed + environment_index * 1_000_000
        world = sample_worlds(family, 1, environment_seed)[0]
        rng = np.random.default_rng(environment_seed + 17)
        records: list[Model1ExampleRecord] = []
        rejections: dict[str, int] = {}
        for attempt in range(stream_config.max_attempts_per_environment):
            if len(records) == stream_config.episodes_per_environment:
                break
            episode_index = len(records)
            record, reason = make_model1_record(
                config,
                world,
                _sample_initial(config, rng),
                example_id=f"stream-{environment_index:03d}-{episode_index:03d}",
                split="stream",
                family_level=level,
                seed=environment_seed + attempt,
            )
            if record is None:
                key = reason or "unknown"
                rejections[key] = rejections.get(key, 0) + 1
            else:
                records.append(record)
        if len(records) != stream_config.episodes_per_environment:
            raise RuntimeError(
                f"environment {environment_index} exhausted attempt budget: {rejections}"
            )
        batches.append(
            EnvironmentBatch(
                environment_index,
                level,
                records[0].world_id,
                tuple(records),
                rejections,
            )
        )
    return tuple(batches)


@dataclass(frozen=True, eq=False)
class ReplayEntry:
    environment_index: int
    record: Model1ExampleRecord


class Model1ReplayBuffer:
    def __init__(self, capacity: int, *, seed: int) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._entries: list[ReplayEntry] = []
        self._seen = 0

    @property
    def entries(self) -> tuple[ReplayEntry, ...]:
        return tuple(self._entries)

    def add(self, environment_index: int, record: Model1ExampleRecord) -> None:
        self._seen += 1
        entry = ReplayEntry(environment_index, record)
        if len(self._entries) < self.capacity:
            self._entries.append(entry)
            return
        index = int(self._rng.integers(0, self._seen))
        if index < self.capacity:
            self._entries[index] = entry

    def sample(self, count: int) -> tuple[ReplayEntry, ...]:
        if count < 1 or count > len(self._entries):
            raise ValueError("sample count outside replay population")
        indices = self._rng.choice(len(self._entries), size=count, replace=False)
        return tuple(self._entries[int(index)] for index in indices)
