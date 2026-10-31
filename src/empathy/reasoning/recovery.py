"""Matched possibility evaluation for true and recovered latent worlds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..physics.config import SimulationConfig
from ..physics.world import WorldCode
from .reachability import EntryStateGrid, Probe, enumerate_reachable


@dataclass(frozen=True)
class EvaluationContext:
    context_id: str
    entry_grid: EntryStateGrid
    simulation: SimulationConfig
    query_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError("context identifier must not be empty")
        if not self.query_steps:
            raise ValueError("query steps must not be empty")
        if len(set(self.query_steps)) != len(self.query_steps):
            raise ValueError("query steps must be unique")


@dataclass(frozen=True, eq=False)
class RecoveredWorldBank:
    worlds: tuple[WorldCode | None, ...]
    valid: np.ndarray

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid)
        if valid.dtype != np.bool_ or valid.shape != (len(self.worlds),):
            raise ValueError("recovered-world validity must be a boolean event vector")
        if any((world is not None) != bool(is_valid) for world, is_valid in zip(
            self.worlds, valid, strict=True
        )):
            raise ValueError("recovered worlds and validity flags disagree")
        object.__setattr__(self, "valid", valid.copy())


@dataclass(frozen=True, eq=False)
class MatchedPossibilityBank:
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    valid_events: np.ndarray
    event_ids: tuple[str, ...]
    probe_ids: tuple[str, ...]
    probes: tuple[Probe, ...]
    recovered: RecoveredWorldBank
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        truth = np.asarray(self.true_labels)
        predicted = np.asarray(self.predicted_labels)
        valid = np.asarray(self.valid_events)
        expected = (len(self.event_ids), len(self.probes))
        if truth.dtype != np.bool_ or predicted.dtype != np.bool_:
            raise ValueError("possibility labels must be boolean")
        if truth.shape != expected or predicted.shape != expected:
            raise ValueError("possibility labels must match event and probe axes")
        if valid.dtype != np.bool_ or valid.shape != (len(self.event_ids),):
            raise ValueError("valid events must be a boolean event vector")
        if len(self.probe_ids) != len(self.probes):
            raise ValueError("probe identifiers must match probes")
        if len(set(self.probe_ids)) != len(self.probe_ids):
            raise ValueError("probe identifiers must be unique")
        object.__setattr__(self, "true_labels", truth.copy())
        object.__setattr__(self, "predicted_labels", predicted.copy())
        object.__setattr__(self, "valid_events", valid.copy())


def _probe_id(probe: Probe) -> str:
    return ":".join(
        (
            probe.context_id,
            str(probe.step),
            repr(float(probe.x)),
            repr(float(probe.y)),
            repr(float(probe.tolerance)),
        )
    )


def _labels_for_world(
    world: WorldCode,
    contexts: dict[str, EvaluationContext],
    probes: tuple[Probe, ...],
) -> np.ndarray:
    results = {
        context_id: enumerate_reachable(
            context.entry_grid,
            world,
            context.query_steps,
            context.simulation,
            context_id=context_id,
        )
        for context_id, context in contexts.items()
    }
    return np.asarray(
        [results[probe.context_id].label(probe) for probe in probes], dtype=bool
    )


def build_matched_possibility_bank(
    true_worlds: Sequence[WorldCode],
    predicted_worlds: Sequence[WorldCode | None],
    contexts: Sequence[EvaluationContext],
    probes: Sequence[Probe],
) -> MatchedPossibilityBank:
    """Evaluate true and recovered worlds with one shared reachability route."""
    true_tuple = tuple(true_worlds)
    predicted_tuple = tuple(predicted_worlds)
    probe_tuple = tuple(probes)
    if len(true_tuple) != len(predicted_tuple):
        raise ValueError("true and predicted worlds must have one value per event")
    if not true_tuple:
        raise ValueError("at least one event world is required")
    if not probe_tuple:
        raise ValueError("at least one probe is required")
    context_map: dict[str, EvaluationContext] = {}
    for context in contexts:
        if context.context_id in context_map:
            raise ValueError(f"duplicate context identifier: {context.context_id}")
        context_map[context.context_id] = context
    if not context_map:
        raise ValueError("at least one evaluation context is required")
    probe_ids = tuple(_probe_id(probe) for probe in probe_tuple)
    if len(set(probe_ids)) != len(probe_ids):
        raise ValueError("probe identifiers must be unique")
    for probe in probe_tuple:
        if probe.context_id not in context_map:
            raise ValueError(f"probe context is not registered: {probe.context_id}")
        if probe.step not in context_map[probe.context_id].query_steps:
            raise ValueError(
                f"probe query step {probe.step} is not registered for {probe.context_id}"
            )

    true_labels = np.stack(
        [_labels_for_world(world, context_map, probe_tuple) for world in true_tuple]
    )
    valid = np.asarray([world is not None for world in predicted_tuple], dtype=bool)
    predicted_labels = np.zeros_like(true_labels)
    for index, world in enumerate(predicted_tuple):
        if world is not None:
            predicted_labels[index] = _labels_for_world(world, context_map, probe_tuple)

    context_metadata = {
        identifier: {
            "entry_grid": context.entry_grid.to_metadata(),
            "simulation": asdict(context.simulation),
            "query_steps": list(context.query_steps),
        }
        for identifier, context in context_map.items()
    }
    recovered = RecoveredWorldBank(predicted_tuple, valid)
    return MatchedPossibilityBank(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        valid_events=valid,
        event_ids=tuple(f"event-{index}" for index in range(len(true_tuple))),
        probe_ids=probe_ids,
        probes=probe_tuple,
        recovered=recovered,
        metadata={
            "contexts": context_metadata,
            "probes": [asdict(probe) for probe in probe_tuple],
        },
    )
