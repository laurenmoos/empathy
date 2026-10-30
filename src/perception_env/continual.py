"""Small replay learner with separately auditable latent and weight updates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .streaming import StreamUpdate


@dataclass(frozen=True, eq=False)
class ReplayExample:
    event_id: str
    pre_belief: np.ndarray
    post_belief: np.ndarray
    belief_difference: np.ndarray

    @classmethod
    def from_update(cls, update: StreamUpdate) -> ReplayExample:
        if update.status == "rejected":
            raise ValueError("rejected updates cannot enter replay")
        return cls(
            update.event_id,
            update.event.pre.copy(),
            update.event.post.copy(),
            update.event.difference.copy(),
        )


class ReplayBuffer:
    def __init__(self, capacity: int, *, seed: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._examples: list[ReplayExample] = []
        self._seen = 0

    @property
    def examples(self) -> tuple[ReplayExample, ...]:
        return tuple(self._examples)

    def add(self, example: ReplayExample) -> None:
        self._seen += 1
        if len(self._examples) < self.capacity:
            self._examples.append(example)
            return
        index = int(self._rng.integers(0, self._seen))
        if index < self.capacity:
            self._examples[index] = example

    def sample(self, count: int) -> tuple[ReplayExample, ...]:
        if count < 1 or count > len(self._examples):
            raise ValueError("sample count outside replay population")
        indices = self._rng.choice(len(self._examples), size=count, replace=False)
        return tuple(self._examples[int(index)] for index in indices)


@dataclass(frozen=True)
class ContinualLog:
    observed_events: int
    latent_update_count: int
    weight_update_count: int
    metrics: dict[str, Any] | None


class ContinualLearner:
    def __init__(
        self,
        buffer: ReplayBuffer,
        train_step: Callable[[Sequence[ReplayExample]], dict[str, Any]],
        *,
        warmup: int,
        batch_size: int,
        update_every: int,
    ) -> None:
        if min(warmup, batch_size, update_every) < 1:
            raise ValueError("continual cadence values must be positive")
        self.buffer = buffer
        self.train_step = train_step
        self.warmup = warmup
        self.batch_size = batch_size
        self.update_every = update_every
        self.latent_update_count = 0
        self.weight_update_count = 0
        self._last_weight_observation = 0
        self.logs: list[ContinualLog] = []

    def observe(self, update: StreamUpdate) -> None:
        self.latent_update_count += 1
        if update.status != "rejected":
            self.buffer.add(ReplayExample.from_update(update))
        self.logs.append(
            ContinualLog(
                self.latent_update_count,
                self.latent_update_count,
                self.weight_update_count,
                None,
            )
        )

    def maybe_train(self) -> dict[str, Any] | None:
        if len(self.buffer.examples) < max(self.warmup, self.batch_size):
            return None
        if self.latent_update_count - self._last_weight_observation < self.update_every:
            return None
        metrics = self.train_step(self.buffer.sample(self.batch_size))
        self.weight_update_count += 1
        self._last_weight_observation = self.latent_update_count
        self.logs.append(
            ContinualLog(
                self.latent_update_count,
                self.latent_update_count,
                self.weight_update_count,
                dict(metrics),
            )
        )
        return metrics

