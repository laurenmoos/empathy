"""Durable benchmark records and belief-only Model 2 inputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

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
        }
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
        return ReachabilityResult(
            context_id=str(self.metadata["context_id"]),
            entry_grid=entry_grid,
            world=self.world(schema),
            query_steps=query_steps,
            trajectories=(),
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
        if arrays["post_belief"].shape != belief_shape or arrays["belief_difference"].shape != belief_shape:
            raise ValueError("belief-history fields must have the same shape")
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

