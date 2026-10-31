"""Environment-backed records and batching for Model 1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from perception_env.beliefs import superpose
from perception_env.config import RenderConfig, SimulationConfig
from perception_env.events import extract_reappearance
from perception_env.families import load_family, sample_worlds
from perception_env.reachability import EntryStateGrid, enumerate_reachable
from perception_env.render import Occluder, render_sequence
from perception_env.simulate import BallState, rollout
from perception_env.world import WorldCode, WorldSchema

from .config import Model1Config
from .types import Model1Batch


@dataclass(frozen=True, eq=False)
class Model1ExampleRecord:
    example_id: str
    split: str
    family_level: str
    seed: int
    world_id: str
    world_flat: np.ndarray
    frames: np.ndarray
    hidden_start: int
    hidden_end: int
    reappearance_step: int
    target_states: np.ndarray
    target_mask: np.ndarray
    oracle_pre_belief: np.ndarray
    actual_path_belief: np.ndarray


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    seed: int
    split_ids: dict[str, tuple[str, ...]]
    rejection_counts: dict[str, int]


def save_example(record: Model1ExampleRecord, root: Path) -> Path:
    path = Path(root) / record.example_id
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    metadata = {
        key: value
        for key, value in asdict(record).items()
        if not isinstance(value, np.ndarray)
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    np.savez_compressed(
        path / "arrays.npz",
        world_flat=record.world_flat,
        frames=record.frames,
        target_states=record.target_states,
        target_mask=record.target_mask,
        oracle_pre_belief=record.oracle_pre_belief,
        actual_path_belief=record.actual_path_belief,
    )
    return path


def load_example(path: Path) -> Model1ExampleRecord:
    payload = json.loads((Path(path) / "metadata.json").read_text())
    with np.load(Path(path) / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return Model1ExampleRecord(**payload, **arrays)


class Model1Dataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str) -> None:
        manifest = json.loads((Path(root) / "manifest.json").read_text())
        self.root = Path(root) / "records"
        self.ids = tuple(manifest["split_ids"][split])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Model1ExampleRecord:
        return load_example(self.root / self.ids[index])


def _path_belief(states: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    belief = np.zeros((size, size), dtype=np.float32)
    for state, valid in zip(states, mask, strict=True):
        if not valid:
            continue
        column = int(round(float(np.clip(state[0], 0, 1)) * (size - 1)))
        row = int(round((1.0 - float(np.clip(state[1], 0, 1))) * (size - 1)))
        belief[row, column] = 1.0
    return belief


def _world_id(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()[:16]


def make_model1_record(
    config: Model1Config,
    world: WorldCode,
    initial: BallState,
    *,
    example_id: str,
    split: str,
    family_level: str,
    seed: int,
) -> tuple[Model1ExampleRecord | None, str | None]:
    schema = WorldSchema.v1()
    schema.validate(world)
    simulation = SimulationConfig(dt=0.1, steps=40, seed=seed)
    trajectory = rollout(initial, world, simulation)
    if not trajectory.valid:
        return None, trajectory.termination_reason.value
    render_config = RenderConfig(
        width=config.dataset.frame_size,
        height=config.dataset.frame_size,
        ball_radius=2,
        min_hidden_steps=2,
    )
    occluder = Occluder(**config.dataset.occluder)
    sequence = render_sequence(trajectory, (occluder,), render_config)
    event = extract_reappearance(sequence)
    if event is None:
        return None, sequence.rejection_reason or "no_reappearance"
    horizon = config.dataset.prediction_steps
    target = np.zeros((horizon, 4), dtype=np.float32)
    target_mask = np.zeros(horizon, dtype=bool)
    available = trajectory.states[event.hidden_start : event.hidden_start + horizon]
    for target_index, state in enumerate(available):
        target[target_index] = (state.x, state.y, state.vx, state.vy)
        target_mask[target_index] = True
    low, high = config.dataset.velocity_bounds
    outside_velocity_support = np.any(target[target_mask, 2:] < low) or np.any(
        target[target_mask, 2:] > high
    )
    if not target_mask.any() or outside_velocity_support:
        return None, "velocity_outside_normalization"
    entry = trajectory.states[event.hidden_start]
    grid = EntryStateGrid(
        (max(0.0, entry.x - 0.02), min(1.0, entry.x + 0.02)),
        (max(0.0, entry.y - 0.02), min(1.0, entry.y + 0.02)),
        (entry.vx - 0.02, entry.vx + 0.02),
        (entry.vy - 0.02, entry.vy + 0.02),
        (2, 2, 2, 2),
    )
    local_simulation = SimulationConfig(dt=simulation.dt, steps=horizon - 1, seed=seed)
    reachability = enumerate_reachable(
        grid, world, (horizon - 1,), local_simulation, context_id=example_id
    )
    valid_paths = [item for item in reachability.trajectories if item.valid]
    if not valid_paths:
        return None, "no_reachable_paths"
    oracle = superpose(valid_paths, np.ones(len(valid_paths)), render_config)
    flat = world.to_flat(schema).astype(np.float32)
    return (
        Model1ExampleRecord(
            example_id=example_id,
            split=split,
            family_level=family_level,
            seed=seed,
            world_id=_world_id(flat),
            world_flat=flat,
            frames=sequence.frames,
            hidden_start=event.hidden_start,
            hidden_end=event.hidden_end,
            reappearance_step=event.step,
            target_states=target,
            target_mask=target_mask,
            oracle_pre_belief=oracle,
            actual_path_belief=_path_belief(target, target_mask, config.dataset.frame_size),
        ),
        None,
    )


def _make_record(
    config: Model1Config,
    family_config: Path,
    level: str,
    split: str,
    index: int,
    seed: int,
) -> tuple[Model1ExampleRecord | None, str | None]:
    rng = np.random.default_rng(seed)
    family = load_family(family_config, level)
    world = sample_worlds(family, 1, seed)[0]
    ranges = config.dataset.initial_state
    initial = BallState(
        *(float(rng.uniform(*ranges[name])) for name in ("x", "y", "vx", "vy"))
    )
    return make_model1_record(
        config,
        world,
        initial,
        example_id=f"{split}-{index:06d}",
        split=split,
        family_level=level,
        seed=seed,
    )


def generate_model1_dataset(
    config: Model1Config,
    family_config: Path,
    output: Path,
    *,
    seed: int,
    quick: bool = False,
) -> DatasetManifest:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    records_root = output / "records"
    records_root.mkdir()
    counts = config.dataset.quick_counts if quick else config.dataset.counts
    split_ids: dict[str, tuple[str, ...]] = {}
    rejections: dict[str, int] = {}
    attempt = 0
    split_counts = zip(("train", "validation", "test"), counts, strict=True)
    for split_index, (split, count) in enumerate(split_counts):
        accepted: list[str] = []
        while len(accepted) < count:
            if attempt >= config.dataset.max_attempts:
                raise RuntimeError(f"generation attempt budget exhausted: {rejections}")
            level = config.dataset.levels[len(accepted) % len(config.dataset.levels)]
            example_seed = seed + split_index * 1_000_000 + attempt
            record, reason = _make_record(
                config, Path(family_config), level, split, len(accepted), example_seed
            )
            attempt += 1
            if record is None:
                key = reason or "unknown"
                rejections[key] = rejections.get(key, 0) + 1
                continue
            save_example(record, records_root)
            accepted.append(record.example_id)
        split_ids[split] = tuple(accepted)
    manifest = DatasetManifest(config.schema_version, seed, split_ids, rejections)
    payload: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "seed": seed,
        "split_ids": {key: list(value) for key, value in split_ids.items()},
        "rejection_counts": rejections,
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest


def _clip(
    record: Model1ExampleRecord, end: int, config: Model1Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(end + 1, dtype=np.int64)[-config.dataset.clip_steps :]
    frames = record.frames[indices].astype(np.float32) / 255.0
    output = np.zeros(
        (config.dataset.clip_steps, 1, config.dataset.frame_size, config.dataset.frame_size),
        dtype=np.float32,
    )
    times = np.zeros(config.dataset.clip_steps, dtype=np.float32)
    mask = np.zeros(config.dataset.clip_steps, dtype=bool)
    length = len(indices)
    output[:length, 0] = frames
    times[:length] = (indices - record.hidden_start) / config.dataset.prediction_steps
    mask[:length] = True
    return output, times, mask


def collate_model1(records: Sequence[Model1ExampleRecord], config: Model1Config) -> Model1Batch:
    if not records:
        raise ValueError("cannot collate an empty batch")
    pre = [_clip(record, record.hidden_end, config) for record in records]
    post = [_clip(record, record.reappearance_step, config) for record in records]
    states = np.stack([record.target_states[0] for record in records]).astype(np.float32)
    low, high = config.dataset.velocity_bounds
    states[:, 2:] = (states[:, 2:] - low) / (high - low)
    return Model1Batch(
        pre_frames=torch.from_numpy(np.stack([item[0] for item in pre])),
        pre_times=torch.from_numpy(np.stack([item[1] for item in pre])),
        pre_mask=torch.from_numpy(np.stack([item[2] for item in pre])),
        post_frames=torch.from_numpy(np.stack([item[0] for item in post])),
        post_times=torch.from_numpy(np.stack([item[1] for item in post])),
        post_mask=torch.from_numpy(np.stack([item[2] for item in post])),
        entry_state=torch.from_numpy(states),
        target_positions=torch.from_numpy(
            np.stack([record.target_states[:, :2] for record in records])
        ),
        target_mask=torch.from_numpy(np.stack([record.target_mask for record in records])),
        oracle_pre_belief=torch.from_numpy(
            np.stack([record.oracle_pre_belief for record in records])
        ),
        actual_path_belief=torch.from_numpy(
            np.stack([record.actual_path_belief for record in records])
        ),
        example_ids=tuple(record.example_id for record in records),
        family_levels=tuple(record.family_level for record in records),
    )
