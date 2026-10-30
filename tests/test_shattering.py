from itertools import product

import numpy as np
import pytest

from perception_env.reachability import Probe
from perception_env.shattering import (
    LabelMatrix,
    build_label_matrix,
    count_labelings,
    repeat_shattering,
    search_shattered_set,
)


def probes(count: int, contexts: list[str] | None = None) -> list[Probe]:
    contexts = contexts or ["clip-a"] * count
    return [Probe(contexts[index], index / max(count, 1), 0.5, 1, 0.05) for index in range(count)]


def matrix(values: list[list[int]], contexts: list[str] | None = None) -> LabelMatrix:
    array = np.asarray(values, dtype=bool)
    return LabelMatrix(
        values=array,
        world_ids=tuple(f"w{index}" for index in range(array.shape[0])),
        probes=tuple(probes(array.shape[1], contexts)),
        metadata={"operator_support": "fixture", "uncertainty_grid": "fixture", "tolerance": 0.05},
    )


def test_build_label_matrix_uses_explicit_oracle_and_counts_distinct_rows() -> None:
    worlds = [0, 1, 2]
    items = probes(2)

    labels = build_label_matrix(
        worlds,
        items,
        labeler=lambda world, probe: world > round(probe.x * 2),
        metadata={"operator_support": ["threshold"]},
    )

    assert labels.values.tolist() == [[False, False], [True, False], [True, True]]
    assert count_labelings(labels) == 3


def test_constant_classifiers_shatter_one_probe() -> None:
    labels = matrix([[0, 0, 0], [1, 1, 1]])

    estimate = search_shattered_set(labels, max_n=3, budget=100, seed=7)

    assert estimate.dimension == 1
    assert estimate.realized_labelings == 2
    assert estimate.possible_labelings == 2
    assert estimate.metric_name == "context_conditioned"


def test_threshold_classifiers_have_empirical_dimension_one() -> None:
    labels = matrix([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])

    estimate = search_shattered_set(labels, max_n=3, budget=100, seed=3)

    assert estimate.dimension == 1
    assert estimate.exhaustive is True


def test_boolean_fixture_recovers_known_dimension_three() -> None:
    labels = matrix([list(bits) for bits in product([0, 1], repeat=3)])

    estimate = search_shattered_set(labels, max_n=3, budget=100, seed=5)

    assert estimate.dimension == 3
    assert estimate.selected_probe_indices == (0, 1, 2)
    assert estimate.realized_labelings == 8


def test_budgeted_search_and_repeated_summary_are_deterministic() -> None:
    rng = np.random.default_rng(2)
    labels = matrix(rng.integers(0, 2, size=(16, 8)).tolist())

    first = search_shattered_set(labels, max_n=5, budget=5, seed=11)
    second = search_shattered_set(labels, max_n=5, budget=5, seed=11)
    summary = repeat_shattering(labels, max_n=5, budget=5, seeds=(11, 12, 13))

    assert first == second
    assert first.metadata["budget"] == 5
    assert first.metadata["seeds"] == [11]
    assert summary.seeds == (11, 12, 13)
    assert len(summary.dimensions) == 3


def test_multi_context_search_is_never_named_context_conditioned() -> None:
    labels = matrix(
        [list(bits) for bits in product([0, 1], repeat=2)],
        contexts=["clip-a", "clip-b"],
    )

    estimate = search_shattered_set(
        labels, max_n=2, budget=100, seed=1, same_context=False
    )

    assert estimate.dimension == 2
    assert estimate.metric_name == "global_multi_context"
    with pytest.raises(ValueError, match="mixed-context"):
        estimate.as_context_conditioned()


def test_estimate_persists_required_search_metadata() -> None:
    labels = matrix([[0], [1]])

    estimate = search_shattered_set(labels, max_n=1, budget=8, seed=17)

    assert estimate.metadata == {
        "operator_support": "fixture",
        "uncertainty_grid": "fixture",
        "tolerance": 0.05,
        "world_count": 2,
        "probe_count": 1,
        "algorithm": "exhaustive_or_seeded_subset_search",
        "budget": 8,
        "seeds": [17],
    }

