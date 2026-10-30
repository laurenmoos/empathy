import numpy as np

from perception_env.reachability import Probe
from perception_env.shattering import LabelMatrix, estimate_complexity_recovery

PROBES = (
    Probe("context", 0.1, 0.1, 1, 0.01),
    Probe("context", 0.2, 0.2, 1, 0.01),
)


def _matrix(values: list[list[bool]], world_ids: tuple[str, ...] | None = None) -> LabelMatrix:
    ids = world_ids or tuple(f"world-{index}" for index in range(len(values)))
    return LabelMatrix(
        np.asarray(values, dtype=bool),
        ids,
        PROBES,
        {"operator_support": "fixture", "uncertainty_grid": "fixed", "tolerance": 0.01},
    )


def _estimate(true: LabelMatrix, recovered: LabelMatrix):
    return estimate_complexity_recovery(
        true,
        recovered,
        max_n=2,
        budget=100,
        search_seeds=(3, 5, 7),
        bootstrap_seed=11,
        bootstrap_samples=200,
    )


def test_identical_dimension_two_classes_recover_full_complexity() -> None:
    matrix = _matrix([[False, False], [False, True], [True, False], [True, True]])

    estimate = _estimate(matrix, matrix)

    assert estimate.true_dimensions == (2, 2, 2)
    assert estimate.recovered_dimensions == (2, 2, 2)
    assert estimate.true_median == 2.0
    assert estimate.recovered_median == 2.0
    assert estimate.ratio == 1.0
    assert estimate.ratio_status == "defined"
    assert estimate.ratio_interval.lower == 1.0
    assert estimate.ratio_interval.upper == 1.0
    assert estimate.label_agreement == 1.0
    assert estimate.metric_name == "context_conditioned"
    assert estimate.metadata["budget"] == 100
    assert estimate.metadata["search_seeds"] == [3, 5, 7]


def test_collapsed_recovered_class_has_lower_dimension() -> None:
    true = _matrix([[False, False], [False, True], [True, False], [True, True]])
    recovered = _matrix([[False, False], [False, False], [True, False], [True, False]])

    estimate = _estimate(true, recovered)

    assert estimate.true_median == 2.0
    assert estimate.recovered_median == 1.0
    assert estimate.ratio == 0.5
    assert estimate.label_agreement == 0.75


def test_intervals_contain_every_repeated_estimate() -> None:
    true = _matrix([[False, False], [False, True], [True, False], [True, True]])
    recovered = _matrix([[False, False], [False, False], [True, False], [True, False]])

    estimate = _estimate(true, recovered)

    assert estimate.true_interval.lower <= min(estimate.true_dimensions)
    assert estimate.true_interval.upper >= max(estimate.true_dimensions)
    assert estimate.recovered_interval.lower <= min(estimate.recovered_dimensions)
    assert estimate.recovered_interval.upper >= max(estimate.recovered_dimensions)


def test_zero_true_dimension_has_undefined_ratio() -> None:
    matrix = _matrix([[False, False], [False, False]])

    estimate = _estimate(matrix, matrix)

    assert estimate.true_dimensions == (0, 0, 0)
    assert estimate.ratio is None
    assert estimate.ratio_status == "undefined_zero_true_dimension"
    assert estimate.ratio_interval.lower is None
    assert estimate.ratio_interval.upper is None


def test_world_row_order_does_not_change_results() -> None:
    true = _matrix(
        [[False, False], [False, True], [True, False], [True, True]],
        ("d", "b", "a", "c"),
    )
    recovered = _matrix(
        [[True, False], [False, False], [True, True], [False, True]],
        ("a", "d", "c", "b"),
    )

    estimate = _estimate(true, recovered)

    assert estimate.true_dimensions == estimate.recovered_dimensions == (2, 2, 2)
    assert estimate.label_agreement == 1.0


def test_global_and_context_conditioned_metrics_remain_distinct() -> None:
    matrix = _matrix([[False, False], [False, True], [True, False], [True, True]])

    estimate = estimate_complexity_recovery(
        matrix,
        matrix,
        max_n=2,
        budget=100,
        search_seeds=(3,),
        bootstrap_seed=11,
        bootstrap_samples=20,
        same_context=False,
    )

    assert estimate.metric_name == "global_multi_context"
