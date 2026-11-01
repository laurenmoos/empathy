"""Activation-only boundary for counterfactual Model 1 calls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from empathy.physics.world import WorldCode

from .activation_adapter import Model1ActivationPacket


class CounterfactualUnavailable(ValueError):
    """A candidate world could not produce a comparable Model 1 event."""


class CounterfactualActivationSource(Protocol):
    """Run frozen Model 1 privately and expose only detached activations."""

    def activations_for(
        self, world: WorldCode
    ) -> tuple[Model1ActivationPacket, ...]: ...


def _event_vector(packet: Model1ActivationPacket) -> np.ndarray:
    return np.concatenate(
        (
            packet.pre_activation,
            packet.post_activation,
            packet.activation_difference,
        )
    ).astype(np.float64, copy=False)


class ActivationImaginationLoop:
    """Compare observed activations with Model 1's counterfactual activations."""

    def __init__(
        self,
        observed: Sequence[Model1ActivationPacket],
        source: CounterfactualActivationSource,
    ) -> None:
        if not observed:
            raise ValueError("imagination requires observed Model 1 activations")
        checkpoint_ids = {event.checkpoint_id for event in observed}
        if len(checkpoint_ids) != 1:
            raise ValueError("observed activations must use one Model 1 checkpoint")
        self.observed = tuple(observed)
        self.source = source

    def discrepancies(self, worlds: Sequence[WorldCode]) -> np.ndarray:
        result = np.empty((len(worlds), len(self.observed)), dtype=np.float64)
        for world_index, world in enumerate(worlds):
            try:
                imagined = self.source.activations_for(world)
                if len(imagined) != len(self.observed):
                    raise CounterfactualUnavailable("counterfactual event count differs")
                for event_index, (actual, candidate) in enumerate(
                    zip(self.observed, imagined, strict=True)
                ):
                    if candidate.checkpoint_id != actual.checkpoint_id:
                        raise CounterfactualUnavailable(
                            "counterfactual used a different Model 1 checkpoint"
                        )
                    actual_vector = _event_vector(actual)
                    candidate_vector = _event_vector(candidate)
                    if candidate_vector.shape != actual_vector.shape:
                        raise CounterfactualUnavailable(
                            "counterfactual activation shape differs"
                        )
                    scale = max(float(np.mean(actual_vector**2)), 1.0e-6)
                    result[world_index, event_index] = float(
                        np.mean((actual_vector - candidate_vector) ** 2) / scale
                    )
            except CounterfactualUnavailable:
                result[world_index] = 1.0e6
        return result

