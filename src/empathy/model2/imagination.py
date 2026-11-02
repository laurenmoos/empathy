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


def _normalized_mse(actual: np.ndarray, candidate: np.ndarray) -> float:
    if actual.shape != candidate.shape:
        raise CounterfactualUnavailable("counterfactual activation shape differs")
    actual_values = actual.astype(np.float64, copy=False)
    candidate_values = candidate.astype(np.float64, copy=False)
    scale = max(float(np.mean(actual_values**2)), 1.0e-6)
    return float(np.mean((actual_values - candidate_values) ** 2) / scale)


def _event_discrepancy(
    actual: Model1ActivationPacket,
    candidate: Model1ActivationPacket,
) -> float:
    guess = _normalized_mse(actual.guess_activation, candidate.guess_activation)
    before = _normalized_mse(actual.pre_activation, candidate.pre_activation)
    after = _normalized_mse(actual.post_activation, candidate.post_activation)
    return guess + 0.25 * (before + after)


class ActivationImaginationLoop:
    """Compare observed activations with Model 1's counterfactual activations."""

    def __init__(
        self,
        observed: Sequence[Model1ActivationPacket],
        source: CounterfactualActivationSource,
    ) -> None:
        if not observed:
            raise ValueError("imagination requires observed Model 1 activations")
        if not all(type(event) is Model1ActivationPacket for event in observed):
            raise TypeError("observed events must be Model1ActivationPacket instances")
        checkpoint_ids = {event.checkpoint_id for event in observed}
        if len(checkpoint_ids) != 1:
            raise ValueError("observed activations must use one Model 1 checkpoint")
        self.observed = tuple(observed)
        self.source = source

    def discrepancies(self, worlds: Sequence[WorldCode]) -> np.ndarray:
        result = np.empty((len(worlds), len(self.observed)), dtype=np.float64)
        activations_for_many = getattr(self.source, "activations_for_many", None)
        if callable(activations_for_many):
            imagined_worlds = activations_for_many(tuple(worlds))
            if len(imagined_worlds) != len(worlds):
                raise CounterfactualUnavailable(
                    "batched counterfactual world count differs"
                )
        else:
            imagined_worlds = []
            for world in worlds:
                try:
                    imagined_worlds.append(self.source.activations_for(world))
                except CounterfactualUnavailable:
                    imagined_worlds.append(None)

        for world_index, imagined in enumerate(imagined_worlds):
            try:
                if imagined is None:
                    raise CounterfactualUnavailable(
                        "candidate produced no comparable activations"
                    )
                if not all(
                    type(event) is Model1ActivationPacket for event in imagined
                ):
                    raise CounterfactualUnavailable(
                        "counterfactual source returned a non-activation packet"
                    )
                if len(imagined) != len(self.observed):
                    raise CounterfactualUnavailable("counterfactual event count differs")
                for event_index, (actual, candidate) in enumerate(
                    zip(self.observed, imagined, strict=True)
                ):
                    if (
                        candidate.episode_id != actual.episode_id
                        or candidate.event_ordinal != actual.event_ordinal
                    ):
                        raise CounterfactualUnavailable(
                            "counterfactual event identity differs"
                        )
                    if candidate.checkpoint_id != actual.checkpoint_id:
                        raise CounterfactualUnavailable(
                            "counterfactual used a different Model 1 checkpoint"
                        )
                    result[world_index, event_index] = _event_discrepancy(
                        actual, candidate
                    )
            except CounterfactualUnavailable:
                result[world_index] = np.inf
        return result

