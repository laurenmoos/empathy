"""Durable benchmark records and belief-only Model 2 inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .beliefs import BeliefEvent
from .config import RenderConfig, SimulationConfig
from .reachability import EntryStateGrid, ReachabilityResult
from .render import Occluder, RenderedSequence
from .simulate import BallState, CollisionEvent, TerminationReason, Trajectory
from .world import WorldCode, WorldSchema


def _states_to_array(states: tuple[BallState, ...]) -> np.ndarray:
    return np.asarray([[state.x, state.y, state.vx, state.vy] for state in states], dtype=float)


def _states_from_array(values: np.ndarray) -> tuple[BallState, ...]:
    return tuple(BallState(*(float(value) for value in row)) for row in values)


def belief_entropy(values: np.ndarray) -> np.ndarray:
    """Return one entropy scalar per belief-history plane."""
    beliefs = np.asarray(values)
    if beliefs.ndim != 3:
        raise ValueError("belief history must have shape [history, height, width]")
    clipped = np.clip(beliefs, 1.0e-6, 1.0 - 1.0e-6)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    return entropy.mean(axis=(-2, -1)).astype(np.float32)


@dataclass(frozen=True, eq=False)
class BenchmarkRecord:
    record_id: str
    schema_version: str
    family_level: str
    seed: int
    rejection_reason: str | None
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @classmethod
    def from_components(
        cls,
        *,
        record_id: str,
        family_level: str,
        schema: WorldSchema,
        world: WorldCode,
        trajectory: Trajectory,
        sequence: RenderedSequence,
        reachability: ReachabilityResult,
        belief: BeliefEvent,
        rejection_reason: str | None = None,
    ) -> BenchmarkRecord:
        schema.validate(world)
        endpoint_rows: list[list[float]] = []
        for step in reachability.query_steps:
            endpoint_rows.extend(
                [float(step), state.x, state.y, state.vx, state.vy]
                for state in reachability.endpoints[step]
            )
        metadata: dict[str, Any] = {
            "simulation_config": asdict(reachability.config),
            "render_config": asdict(sequence.config),
            "occluders": [asdict(item) for item in sequence.occluders],
            "entry_grid": reachability.entry_grid.to_metadata(),
            "query_steps": list(reachability.query_steps),
            "context_id": reachability.context_id,
            "trajectory_valid": trajectory.valid,
            "termination_reason": trajectory.termination_reason.value,
            "collisions": [
                {"step": item.step, "operator": item.operator, "position": item.position}
                for item in trajectory.collisions
            ],
            "belief_step": belief.step,
            "reachability_termination_reasons": [
                item.termination_reason.value for item in reachability.trajectories
            ],
        }
        trajectory_lengths = np.asarray(
            [len(item.states) for item in reachability.trajectories], dtype=np.int64
        )
        max_length = int(trajectory_lengths.max(initial=0))
        reachability_states = np.zeros(
            (len(reachability.trajectories), max_length, 4), dtype=np.float64
        )
        for index, item in enumerate(reachability.trajectories):
            reachability_states[index, : len(item.states)] = _states_to_array(item.states)
        arrays = {
            "world_flat": world.to_flat(schema),
            "latent_states": _states_to_array(trajectory.states),
            "frames": sequence.frames,
            "ball_masks": sequence.ball_masks,
            "visible_ball_masks": sequence.visible_ball_masks,
            "visible": sequence.visible,
            "fully_hidden": sequence.fully_hidden,
            "partially_occluded": sequence.partially_occluded,
            "endpoint_states": np.asarray(endpoint_rows, dtype=float).reshape(-1, 5),
            "pre_belief": belief.pre,
            "post_belief": belief.post,
            "belief_difference": belief.difference,
            "reachability_states": reachability_states,
            "reachability_lengths": trajectory_lengths,
            "reachability_valid": np.asarray(
                [item.valid for item in reachability.trajectories], dtype=bool
            ),
            "reachability_seeds": np.asarray(
                [item.seed for item in reachability.trajectories], dtype=np.int64
            ),
        }
        return cls(
            record_id=record_id,
            schema_version=schema.version,
            family_level=family_level,
            seed=trajectory.seed,
            rejection_reason=rejection_reason,
            metadata=metadata,
            arrays=arrays,
        )

    def world(self, schema: WorldSchema) -> WorldCode:
        if schema.version != self.schema_version:
            raise ValueError("record and schema versions differ")
        return schema.from_flat(self.arrays["world_flat"])

    def trajectory(self) -> Trajectory:
        collisions = tuple(
            CollisionEvent(int(item["step"]), str(item["operator"]), float(item["position"]))
            for item in self.metadata["collisions"]
        )
        return Trajectory(
            states=_states_from_array(self.arrays["latent_states"]),
            collisions=collisions,
            valid=bool(self.metadata["trajectory_valid"]),
            seed=self.seed,
            termination_reason=TerminationReason(self.metadata["termination_reason"]),
        )

    def sequence(self) -> RenderedSequence:
        return RenderedSequence(
            frames=self.arrays["frames"],
            ball_masks=self.arrays["ball_masks"],
            visible_ball_masks=self.arrays["visible_ball_masks"],
            visible=self.arrays["visible"],
            fully_hidden=self.arrays["fully_hidden"],
            partially_occluded=self.arrays["partially_occluded"],
            latent_states=self.trajectory().states,
            occluders=tuple(Occluder(**values) for values in self.metadata["occluders"]),
            config=RenderConfig(**self.metadata["render_config"]),
            seed=self.seed,
            rejection_reason=self.rejection_reason,
        )

    def belief_event(self) -> BeliefEvent:
        return BeliefEvent(
            int(self.metadata["belief_step"]),
            self.arrays["pre_belief"],
            self.arrays["post_belief"],
        )

    def reachability(self) -> ReachabilityResult:
        schema = WorldSchema.v1()
        grid = self.metadata["entry_grid"]
        entry_grid = EntryStateGrid(
            tuple(grid["x_bounds"]),
            tuple(grid["y_bounds"]),
            tuple(grid["vx_bounds"]),
            tuple(grid["vy_bounds"]),
            tuple(grid["resolution"]),
        )
        query_steps = tuple(int(value) for value in self.metadata["query_steps"])
        endpoints: dict[int, tuple[BallState, ...]] = {}
        rows = self.arrays["endpoint_states"]
        for step in query_steps:
            endpoints[step] = _states_from_array(rows[rows[:, 0] == step, 1:])
        trajectories = tuple(
            Trajectory(
                states=_states_from_array(self.arrays["reachability_states"][index, :length]),
                collisions=(),
                valid=bool(self.arrays["reachability_valid"][index]),
                seed=int(self.arrays["reachability_seeds"][index]),
                termination_reason=TerminationReason(
                    self.metadata["reachability_termination_reasons"][index]
                ),
            )
            for index, length in enumerate(self.arrays["reachability_lengths"])
        )
        return ReachabilityResult(
            context_id=str(self.metadata["context_id"]),
            entry_grid=entry_grid,
            world=self.world(schema),
            query_steps=query_steps,
            trajectories=trajectories,
            endpoints=endpoints,
            config=SimulationConfig(**self.metadata["simulation_config"]),
        )


@dataclass(frozen=True, eq=False)
class Model2InputRecord:
    pre_belief: np.ndarray
    post_belief: np.ndarray
    belief_difference: np.ndarray
    entropy: np.ndarray
    history_mask: np.ndarray
    target_world: np.ndarray
    channel_names: tuple[str, str, str] = (
        "pre_belief",
        "post_belief",
        "belief_difference",
    )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Model2InputRecord:
        allowed = {
            "pre_belief",
            "post_belief",
            "belief_difference",
            "entropy",
            "history_mask",
            "target_world",
        }
        unsupported = set(values) - allowed
        if unsupported:
            raise ValueError(f"unsupported or first-order fields: {sorted(unsupported)}")
        missing = allowed - set(values)
        if missing:
            raise ValueError(f"missing Model 2 fields: {sorted(missing)}")
        arrays = {name: np.asarray(values[name]) for name in allowed}
        belief_shape = arrays["pre_belief"].shape
        if len(belief_shape) != 3 or belief_shape[0] != 4:
            raise ValueError("belief-history fields must have shape [4, height, width]")
        if (
            arrays["post_belief"].shape != belief_shape
            or arrays["belief_difference"].shape != belief_shape
        ):
            raise ValueError("belief-history fields must have the same shape")
        if not all(np.isfinite(arrays[name]).all() for name in allowed):
            raise ValueError("Model 2 tensors must contain only finite values")
        if not np.allclose(
            arrays["belief_difference"],
            arrays["post_belief"] - arrays["pre_belief"],
        ):
            raise ValueError("belief_difference must equal post_belief - pre_belief")
        if arrays["entropy"].shape != (4,) or not np.allclose(
            arrays["entropy"], belief_entropy(arrays["pre_belief"])
        ):
            raise ValueError("entropy must be derived from pre_belief")
        if arrays["history_mask"].shape != (4,) or arrays["history_mask"].dtype != np.bool_:
            raise ValueError("history_mask must be a boolean vector of length 4")
        if arrays["target_world"].shape != (len(WorldSchema.v1().field_order),):
            raise ValueError("target_world does not match the v1 world schema")
        return cls(**arrays)


def save_record(record: BenchmarkRecord, root: Path) -> Path:
    path = Path(root) / record.record_id
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    payload = {
        "record_id": record.record_id,
        "schema_version": record.schema_version,
        "family_level": record.family_level,
        "seed": record.seed,
        "rejection_reason": record.rejection_reason,
        "metadata": record.metadata,
    }
    (path / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    np.savez_compressed(path / "arrays.npz", **record.arrays)
    return path


def load_record(path: Path) -> BenchmarkRecord:
    path = Path(path)
    payload = json.loads((path / "metadata.json").read_text())
    with np.load(path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return BenchmarkRecord(
        record_id=str(payload["record_id"]),
        schema_version=str(payload["schema_version"]),
        family_level=str(payload["family_level"]),
        seed=int(payload["seed"]),
        rejection_reason=payload["rejection_reason"],
        metadata=dict(payload["metadata"]),
        arrays=arrays,
    )

