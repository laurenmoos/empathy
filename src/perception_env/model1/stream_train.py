"""Streaming predict-record-reveal-update training for Model 1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from perception_env.families import load_family, sample_worlds
from perception_env.simulate import BallState

from .config import Model1Config
from .data import Model1ExampleRecord, collate_model1, make_model1_record
from .model import Model1
from .rasterize import rasterize_mixture
from .train import _run_batch, save_checkpoint, seed_model1


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
        if self.replay_batch_size > self.episodes_per_environment:
            raise ValueError("replay batch must fit in the first environment")
        if self.environment_count > 1 and self.replay_batch_size < 2:
            raise ValueError("multiple environments require at least two replay samples")


def load_stream_training_config(path: Path) -> StreamTrainingConfig:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "stream"}:
        raise ValueError("stream configuration must contain schema_version and stream")
    if not isinstance(payload["stream"], dict):
        raise ValueError("stream configuration stream must be a mapping")
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

    def add_environment(
        self,
        environment_index: int,
        records: list[Model1ExampleRecord] | tuple[Model1ExampleRecord, ...],
    ) -> None:
        if not records:
            raise ValueError("environment replay batch cannot be empty")
        prior = [
            entry
            for entry in self._entries
            if entry.environment_index < environment_index
        ]
        protected_prior = (
            prior[int(self._rng.integers(0, len(prior)))] if prior else None
        )
        for record in records:
            self.add(environment_index, record)
        if protected_prior is None:
            return
        previous_indices = [
            index
            for index, entry in enumerate(self._entries)
            if entry.environment_index < environment_index
        ]
        if not previous_indices:
            replace_index = int(self._rng.integers(0, len(self._entries)))
            self._entries[replace_index] = protected_prior
            previous_indices = [replace_index]
        if not any(
            entry.environment_index == environment_index for entry in self._entries
        ):
            candidates = [
                index
                for index in range(len(self._entries))
                if index != previous_indices[0]
            ]
            replace_index = int(self._rng.choice(candidates))
            record = records[int(self._rng.integers(0, len(records)))]
            self._entries[replace_index] = ReplayEntry(environment_index, record)

    def sample(self, count: int) -> tuple[ReplayEntry, ...]:
        if count < 1 or count > len(self._entries):
            raise ValueError("sample count outside replay population")
        indices = self._rng.choice(len(self._entries), size=count, replace=False)
        return tuple(self._entries[int(index)] for index in indices)

    def sample_for_update(
        self, count: int, *, current_environment: int
    ) -> tuple[ReplayEntry, ...]:
        if count < 1 or count > len(self._entries):
            raise ValueError("sample count outside replay population")
        current = [
            index
            for index, entry in enumerate(self._entries)
            if entry.environment_index == current_environment
        ]
        previous = [
            index
            for index, entry in enumerate(self._entries)
            if entry.environment_index < current_environment
        ]
        selected: list[int] = []
        if current:
            selected.append(int(self._rng.choice(current)))
        if previous and len(selected) < count:
            selected.append(int(self._rng.choice(previous)))
        available = [index for index in range(len(self._entries)) if index not in selected]
        remaining = count - len(selected)
        if remaining:
            selected.extend(int(index) for index in self._rng.choice(available, remaining, False))
        return tuple(self._entries[index] for index in selected)


@dataclass(frozen=True, eq=False)
class RecordedBeliefEvent:
    environment_index: int
    event_id: str
    weight_update_count: int
    path: Path


@dataclass(frozen=True, eq=False)
class StreamTrainingResult:
    events: tuple[RecordedBeliefEvent, ...]
    history: tuple[dict[str, Any], ...]
    checkpoint: Path
    summary: dict[str, Any]


def _model_hash(model: Model1) -> str:
    digest = hashlib.sha256()
    for name, values in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(values.detach().cpu().numpy())
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _loss_values(terms) -> dict[str, float]:
    return {
        name: float(getattr(terms, name).detach().cpu())
        for name in ("total", "trajectory", "mode", "state", "pre_belief", "post_belief")
    }


def _record_frozen_batch(
    model: Model1,
    batch: EnvironmentBatch,
    config: Model1Config,
    events_root: Path,
    *,
    device: torch.device,
    weight_update_count: int,
) -> tuple[RecordedBeliefEvent, ...]:
    collated = collate_model1(batch.records, config).to(device)
    model.eval()
    with torch.inference_mode():
        pre_prediction = model(
            collated.pre_frames, collated.pre_times, collated.pre_mask
        )
        post_prediction = model(
            collated.post_frames, collated.post_times, collated.post_mask
        )
        pre = rasterize_mixture(
            pre_prediction,
            height=config.dataset.frame_size,
            width=config.dataset.frame_size,
            step_mask=collated.target_mask,
        ).collapsed.cpu().numpy()
        post = rasterize_mixture(
            post_prediction,
            height=config.dataset.frame_size,
            width=config.dataset.frame_size,
            step_mask=collated.target_mask,
        ).collapsed.cpu().numpy()
    events: list[RecordedBeliefEvent] = []
    for index, record in enumerate(batch.records):
        path = events_root / f"{record.example_id}.npz"
        np.savez_compressed(
            path,
            pre_belief=pre[index].astype(np.float32),
            post_belief=post[index].astype(np.float32),
            belief_difference=(post[index] - pre[index]).astype(np.float32),
            environment_index=np.asarray(batch.environment_index, dtype=np.int64),
            weight_update_count=np.asarray(weight_update_count, dtype=np.int64),
        )
        events.append(
            RecordedBeliefEvent(
                batch.environment_index,
                record.example_id,
                weight_update_count,
                path,
            )
        )
    return tuple(events)


def train_model1_stream(
    config: Model1Config,
    stream_config: StreamTrainingConfig,
    family_config: Path,
    output: Path,
    *,
    device: torch.device,
    seed: int,
) -> StreamTrainingResult:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    batches = generate_environment_batches(
        config, stream_config, family_config, seed=seed
    )
    output.mkdir(parents=True)
    events_root = output / "events"
    events_root.mkdir()
    generator = seed_model1(seed)
    del generator
    model = Model1(config.model, prediction_steps=config.dataset.prediction_steps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    replay = Model1ReplayBuffer(stream_config.replay_capacity, seed=seed + 1)
    initial_hash = _model_hash(model)
    events: list[RecordedBeliefEvent] = []
    history: list[dict[str, Any]] = []
    weight_update_count = 0
    best_loss = float("inf")
    history_path = output / "history.jsonl"
    environment_summaries: list[dict[str, Any]] = []
    for environment in batches:
        frozen_events = _record_frozen_batch(
            model,
            environment,
            config,
            events_root,
            device=device,
            weight_update_count=weight_update_count,
        )
        events.extend(frozen_events)
        replay.add_environment(environment.environment_index, environment.records)
        event_rows: list[dict[str, Any]] = []
        for update_index in range(stream_config.update_steps_per_environment):
            sampled = replay.sample_for_update(
                stream_config.replay_batch_size,
                current_environment=environment.environment_index,
            )
            raw_batch = collate_model1([entry.record for entry in sampled], config)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            terms = _run_batch(model, raw_batch.to(device), config)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip
            )
            optimizer.step()
            weight_update_count += 1
            values = _loss_values(terms)
            best_loss = min(best_loss, values["total"])
            row: dict[str, Any] = {
                "environment_index": environment.environment_index,
                "update_index": update_index,
                "weight_update_count": weight_update_count,
                "replay_event_ids": [entry.record.example_id for entry in sampled],
                "replay_environment_indices": [
                    entry.environment_index for entry in sampled
                ],
                **values,
            }
            history.append(row)
            event_rows.append(row)
            with history_path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        environment_summaries.append(
            {
                "environment_index": environment.environment_index,
                "level": environment.level,
                "world_id": environment.world_id,
                "event_ids": [record.example_id for record in environment.records],
                "recorded_weight_update_count": frozen_events[0].weight_update_count,
                "completed_weight_update_count": event_rows[-1][
                    "weight_update_count"
                ],
                "rejection_counts": environment.rejection_counts,
            }
        )
    checkpoint = save_checkpoint(
        output / "stream-last.pt",
        model,
        optimizer,
        config,
        epoch=weight_update_count,
        seed=seed,
        best_validation_loss=best_loss,
    )
    model.eval()
    summary: dict[str, Any] = {
        "schema_version": stream_config.schema_version,
        "seed": seed,
        "environment_count": len(batches),
        "event_count": len(events),
        "weight_update_count": weight_update_count,
        "replay_size": len(replay.entries),
        "initial_model_hash": initial_hash,
        "final_model_hash": _model_hash(model),
        "environments": environment_summaries,
    }
    (output / "stream-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return StreamTrainingResult(tuple(events), tuple(history), checkpoint, summary)
