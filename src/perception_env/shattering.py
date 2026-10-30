"""Search-relative empirical growth and shattering estimates."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from .reachability import Probe


@dataclass(frozen=True, eq=False)
class LabelMatrix:
    values: np.ndarray
    world_ids: tuple[str, ...]
    probes: tuple[Probe, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=bool)
        if values.ndim != 2:
            raise ValueError("label matrix must be two-dimensional")
        if values.shape != (len(self.world_ids), len(self.probes)):
            raise ValueError("label matrix dimensions must match worlds and probes")
        object.__setattr__(self, "values", values.copy())


@dataclass(frozen=True, eq=False)
class ShatteringEstimate:
    dimension: int
    selected_probe_indices: tuple[int, ...]
    realized_labelings: int
    possible_labelings: int
    metric_name: str
    exhaustive: bool
    matrix: LabelMatrix
    metadata: dict[str, Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShatteringEstimate):
            return NotImplemented
        return (
            self.dimension == other.dimension
            and self.selected_probe_indices == other.selected_probe_indices
            and self.realized_labelings == other.realized_labelings
            and self.possible_labelings == other.possible_labelings
            and self.metric_name == other.metric_name
            and self.exhaustive == other.exhaustive
            and self.metadata == other.metadata
            and self.matrix.world_ids == other.matrix.world_ids
            and self.matrix.probes == other.matrix.probes
            and np.array_equal(self.matrix.values, other.matrix.values)
        )

    def as_context_conditioned(self) -> ShatteringEstimate:
        contexts = {self.matrix.probes[index].context_id for index in self.selected_probe_indices}
        if self.metric_name != "context_conditioned" or len(contexts) > 1:
            raise ValueError("a mixed-context estimate cannot be labeled context-conditioned")
        return self


@dataclass(frozen=True)
class RepeatedShatteringSummary:
    seeds: tuple[int, ...]
    dimensions: tuple[int, ...]
    metric_name: str


def build_label_matrix(
    worlds: Sequence[Any],
    probes: Sequence[Probe],
    *,
    labeler: Callable[[Any, Probe], bool],
    metadata: dict[str, Any] | None = None,
) -> LabelMatrix:
    values = np.asarray(
        [[bool(labeler(world, probe)) for probe in probes] for world in worlds],
        dtype=bool,
    ).reshape(len(worlds), len(probes))
    world_ids = tuple(f"world-{index}" for index in range(len(worlds)))
    return LabelMatrix(values, world_ids, tuple(probes), dict(metadata or {}))


def count_labelings(matrix: LabelMatrix) -> int:
    if matrix.values.shape[0] == 0:
        return 0
    return int(np.unique(matrix.values, axis=0).shape[0])


def _context_groups(matrix: LabelMatrix) -> list[tuple[int, ...]]:
    by_context: dict[str, list[int]] = {}
    for index, probe in enumerate(matrix.probes):
        by_context.setdefault(probe.context_id, []).append(index)
    return [tuple(indices) for indices in by_context.values()]


def _candidate_count(matrix: LabelMatrix, size: int, same_context: bool) -> int:
    if not same_context:
        return math.comb(len(matrix.probes), size)
    return sum(
        math.comb(len(indices), size)
        for indices in _context_groups(matrix)
        if len(indices) >= size
    )


def _candidate_subsets(
    matrix: LabelMatrix,
    size: int,
    same_context: bool,
    budget: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, ...]], bool]:
    count = _candidate_count(matrix, size, same_context)
    if count <= budget:
        if not same_context:
            return list(combinations(range(len(matrix.probes)), size)), True
        candidates: set[tuple[int, ...]] = set()
        for indices in _context_groups(matrix):
            candidates.update(combinations(indices, size))
        return sorted(candidates), True

    groups = (
        [tuple(range(len(matrix.probes)))]
        if not same_context
        else [indices for indices in _context_groups(matrix) if len(indices) >= size]
    )
    candidates: set[tuple[int, ...]] = set()
    attempts = 0
    while len(candidates) < budget and attempts < budget * 20:
        group = groups[int(rng.integers(len(groups)))]
        chosen = rng.choice(group, size=size, replace=False)
        candidates.add(tuple(sorted(int(index) for index in chosen)))
        attempts += 1
    return sorted(candidates), False


def _realized(matrix: LabelMatrix, selected: tuple[int, ...]) -> int:
    if not selected:
        return 1
    return int(np.unique(matrix.values[:, selected], axis=0).shape[0])


def search_shattered_set(
    matrix: LabelMatrix,
    max_n: int,
    budget: int,
    seed: int,
    same_context: bool = True,
) -> ShatteringEstimate:
    if max_n < 0 or budget < 1:
        raise ValueError("max_n must be nonnegative and budget must be positive")
    rng = np.random.default_rng(seed)
    metadata = {
        "operator_support": matrix.metadata.get("operator_support", "unspecified"),
        "uncertainty_grid": matrix.metadata.get("uncertainty_grid", "unspecified"),
        "tolerance": matrix.metadata.get("tolerance", "unspecified"),
        "world_count": matrix.values.shape[0],
        "probe_count": matrix.values.shape[1],
        "algorithm": "exhaustive_or_seeded_subset_search",
        "budget": budget,
        "seeds": [seed],
    }
    metric_name = "context_conditioned" if same_context else "global_multi_context"
    all_exhaustive = True
    world_capacity = int(math.floor(math.log2(max(matrix.values.shape[0], 1))))
    upper = min(max_n, matrix.values.shape[1], world_capacity)
    for size in range(upper, 0, -1):
        candidates, exhaustive = _candidate_subsets(
            matrix, size, same_context, budget, rng
        )
        all_exhaustive = all_exhaustive and exhaustive
        for selected in candidates:
            realized = _realized(matrix, selected)
            if realized == 2**size:
                return ShatteringEstimate(
                    dimension=size,
                    selected_probe_indices=selected,
                    realized_labelings=realized,
                    possible_labelings=2**size,
                    metric_name=metric_name,
                    exhaustive=all_exhaustive,
                    matrix=matrix,
                    metadata=metadata,
                )
    return ShatteringEstimate(
        dimension=0,
        selected_probe_indices=(),
        realized_labelings=1,
        possible_labelings=1,
        metric_name=metric_name,
        exhaustive=all_exhaustive,
        matrix=matrix,
        metadata=metadata,
    )


def repeat_shattering(
    matrix: LabelMatrix,
    *,
    max_n: int,
    budget: int,
    seeds: tuple[int, ...],
    same_context: bool = True,
) -> RepeatedShatteringSummary:
    estimates = tuple(
        search_shattered_set(matrix, max_n, budget, seed, same_context)
        for seed in seeds
    )
    return RepeatedShatteringSummary(
        seeds=seeds,
        dimensions=tuple(item.dimension for item in estimates),
        metric_name="context_conditioned" if same_context else "global_multi_context",
    )

