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


@dataclass(frozen=True)
class ShatteringInterval:
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class ComplexityRecoveryEstimate:
    true_dimensions: tuple[int, ...]
    recovered_dimensions: tuple[int, ...]
    true_median: float
    recovered_median: float
    true_interval: ShatteringInterval
    recovered_interval: ShatteringInterval
    ratio: float | None
    ratio_interval: ShatteringInterval
    ratio_status: str
    label_agreement: float
    metric_name: str
    metadata: dict[str, Any]


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


def _canonical_matrix(matrix: LabelMatrix, world_ids: tuple[str, ...]) -> LabelMatrix:
    if len(set(matrix.world_ids)) != len(matrix.world_ids):
        raise ValueError("world identifiers must be unique")
    index_by_id = {world_id: index for index, world_id in enumerate(matrix.world_ids)}
    if set(index_by_id) != set(world_ids):
        raise ValueError("true and recovered matrices must share world identifiers")
    indices = [index_by_id[world_id] for world_id in world_ids]
    return LabelMatrix(
        matrix.values[indices],
        world_ids,
        matrix.probes,
        dict(matrix.metadata),
    )


def _bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> ShatteringInterval:
    if values.size == 0:
        return ShatteringInterval(None, None)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(values, size=values.size, replace=True)
        bootstrap[index] = float(np.median(selected))
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    return ShatteringInterval(
        float(min(lower, values.min())),
        float(max(upper, values.max())),
    )


def estimate_complexity_recovery(
    true_matrix: LabelMatrix,
    recovered_matrix: LabelMatrix,
    *,
    max_n: int,
    budget: int,
    search_seeds: tuple[int, ...],
    bootstrap_seed: int,
    bootstrap_samples: int,
    same_context: bool = True,
) -> ComplexityRecoveryEstimate:
    """Pair empirical shattering estimates for true and recovered world banks."""
    if not search_seeds:
        raise ValueError("at least one search seed is required")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if true_matrix.probes != recovered_matrix.probes:
        raise ValueError("true and recovered matrices must share probes")
    world_ids = tuple(sorted(true_matrix.world_ids))
    truth = _canonical_matrix(true_matrix, world_ids)
    recovered = _canonical_matrix(recovered_matrix, world_ids)
    true_estimates = tuple(
        search_shattered_set(truth, max_n, budget, seed, same_context)
        for seed in search_seeds
    )
    recovered_estimates = tuple(
        search_shattered_set(recovered, max_n, budget, seed, same_context)
        for seed in search_seeds
    )
    true_dimensions = tuple(item.dimension for item in true_estimates)
    recovered_dimensions = tuple(item.dimension for item in recovered_estimates)
    true_values = np.asarray(true_dimensions, dtype=np.float64)
    recovered_values = np.asarray(recovered_dimensions, dtype=np.float64)
    usable = true_values > 0
    if usable.any():
        ratios = recovered_values[usable] / true_values[usable]
        ratio = float(np.median(ratios))
        ratio_interval = _bootstrap_interval(
            ratios, seed=bootstrap_seed, samples=bootstrap_samples
        )
        ratio_status = "defined"
    else:
        ratio = None
        ratio_interval = ShatteringInterval(None, None)
        ratio_status = "undefined_zero_true_dimension"
    metadata = {
        "algorithm": "paired_exhaustive_or_seeded_subset_search",
        "budget": budget,
        "max_n": max_n,
        "search_seeds": list(search_seeds),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "interval_method": "paired_percentile_95_expanded_to_observed_range",
        "operator_support": truth.metadata.get("operator_support", "unspecified"),
        "uncertainty_grid": truth.metadata.get("uncertainty_grid", "unspecified"),
        "tolerance": truth.metadata.get("tolerance", "unspecified"),
        "world_count": len(world_ids),
        "probe_count": len(truth.probes),
    }
    return ComplexityRecoveryEstimate(
        true_dimensions=true_dimensions,
        recovered_dimensions=recovered_dimensions,
        true_median=float(np.median(true_values)),
        recovered_median=float(np.median(recovered_values)),
        true_interval=_bootstrap_interval(
            true_values, seed=bootstrap_seed + 1, samples=bootstrap_samples
        ),
        recovered_interval=_bootstrap_interval(
            recovered_values, seed=bootstrap_seed + 2, samples=bootstrap_samples
        ),
        ratio=ratio,
        ratio_interval=ratio_interval,
        ratio_status=ratio_status,
        label_agreement=float((truth.values == recovered.values).mean()),
        metric_name="context_conditioned" if same_context else "global_multi_context",
        metadata=metadata,
    )

