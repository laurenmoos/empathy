"""Strict input-only arrays and separately typed supervision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from perception_env.physics.world import WorldSchema
from perception_env.reasoning.records import Model2InputRecord

if TYPE_CHECKING:
    import torch


def _readonly(values: np.ndarray, dtype) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class SchemaLayout:
    activity_indices: tuple[int, ...]
    parameter_indices: tuple[int, ...]
    parameter_to_slot: tuple[int, ...]

    @classmethod
    def from_schema(cls, schema: WorldSchema) -> SchemaLayout:
        activity, parameters, slots = [], [], []
        cursor = 0
        for index, spec in enumerate(schema.specs):
            activity.append(cursor)
            parameters.extend(range(cursor + 1, cursor + 1 + len(spec.parameters)))
            slots.extend([index] * len(spec.parameters))
            cursor += 1 + len(spec.parameters)
        return cls(tuple(activity), tuple(parameters), tuple(slots))


@dataclass(frozen=True, eq=False)
class BeliefInputs:
    beliefs: np.ndarray
    history_mask: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.beliefs)
        mask = np.asarray(self.history_mask)
        if (
            values.ndim != 5
            or values.shape[0] < 1
            or values.shape[1:3] != (4, 3)
            or min(values.shape[-2:]) < 8
        ):
            raise ValueError("beliefs must have shape [N,4,3,H,W], N>0 and H,W>=8")
        if mask.shape != values.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("history_mask must be boolean [N,4]")
        if not mask.any(axis=1).all() or (np.diff(mask.astype(int), axis=1) < 0).any():
            raise ValueError("history_mask must be a nonempty valid suffix")
        if not np.isfinite(values).all():
            raise ValueError("beliefs must be finite")
        if ((values[:, :, :2] < 0) | (values[:, :, :2] > 1)).any():
            raise ValueError("pre/post beliefs must lie in [0,1]")
        if not np.allclose(values[:, :, 2], values[:, :, 1] - values[:, :, 0], atol=1e-6, rtol=0):
            raise ValueError("difference must equal post minus pre")
        if np.any(values[~mask] != 0):
            raise ValueError("padded events must be zero")
        object.__setattr__(self, "beliefs", _readonly(values, np.float32))
        object.__setattr__(self, "history_mask", _readonly(mask, bool))

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> BeliefInputs:
        if set(values) != {"beliefs", "history_mask"}:
            raise ValueError("input fields must be exactly beliefs and history_mask")
        return cls(values["beliefs"], values["history_mask"])


@dataclass(frozen=True, eq=False)
class WorldTargets:
    world_flat: np.ndarray
    parameter_identifiable: np.ndarray

    def __post_init__(self) -> None:
        flat = np.asarray(self.world_flat)
        mask = np.asarray(self.parameter_identifiable)
        if flat.ndim != 2 or flat.shape[1] != 12 or flat.shape[0] < 1:
            raise ValueError("world targets must have shape [N,12]")
        if mask.shape != (len(flat), 7) or mask.dtype != np.bool_:
            raise ValueError("identifiability must be boolean [N,7]")
        if not np.isfinite(flat).all():
            raise ValueError("targets must be finite")
        schema = WorldSchema.v1()
        layout = SchemaLayout.from_schema(schema)
        if not np.isin(flat[:, layout.activity_indices], [0, 1]).all():
            raise ValueError("activity targets must be binary")
        for row in flat:
            canonical = schema.from_flat(row).to_flat(schema)
            if not np.allclose(row, canonical, rtol=0, atol=1e-6):
                raise ValueError("target must be a canonical world code")
        object.__setattr__(self, "world_flat", _readonly(flat, np.float32))
        object.__setattr__(self, "parameter_identifiable", _readonly(mask, bool))


def from_legacy(record: Model2InputRecord) -> tuple[BeliefInputs, WorldTargets]:
    checked = Model2InputRecord.from_mapping(
        {
            name: getattr(record, name)
            for name in (
                "pre_belief",
                "post_belief",
                "belief_difference",
                "entropy",
                "history_mask",
                "target_world",
            )
        }
    )
    beliefs = np.stack(
        (checked.pre_belief, checked.post_belief, checked.belief_difference), axis=1
    )[None]
    return (
        BeliefInputs(beliefs, checked.history_mask[None]),
        WorldTargets(checked.target_world[None], np.zeros((1, 7), bool)),
    )


@dataclass(frozen=True)
class Model2Batch:
    beliefs: torch.Tensor
    history_mask: torch.Tensor
    world_flat: torch.Tensor
    parameter_identifiable: torch.Tensor


@dataclass(frozen=True)
class WorldPrediction:
    activity_logits: torch.Tensor
    normalized_parameters: torch.Tensor
