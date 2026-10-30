import numpy as np

from perception_env.coherence import (
    CapacityThresholds,
    TemporalCoherenceReport,
    find_capacity_breakpoint,
)
from perception_env.shattering import ComplexityRecoveryEstimate, ShatteringInterval


def _coherence(value: float | None) -> TemporalCoherenceReport:
    if value is None:
        return TemporalCoherenceReport(
            ("event",),
            ("probe",),
            np.asarray([np.nan]),
            None,
            None,
            None,
            0,
            0,
            "insufficient_data",
        )
    return TemporalCoherenceReport(
        ("event",),
        ("probe",),
        np.asarray([value]),
        value,
        1.0,
        1.0,
        1,
        1,
        "ok",
    )


def _complexity(ratio_lower: float | None, *, zero_true: bool = False):
    if zero_true:
        return ComplexityRecoveryEstimate(
            (0,),
            (0,),
            0.0,
            0.0,
            ShatteringInterval(0.0, 0.0),
            ShatteringInterval(0.0, 0.0),
            None,
            ShatteringInterval(None, None),
            "undefined_zero_true_dimension",
            1.0,
            "context_conditioned",
            {},
        )
    assert ratio_lower is not None
    return ComplexityRecoveryEstimate(
        (2,),
        (2,),
        2.0,
        2.0,
        ShatteringInterval(2.0, 2.0),
        ShatteringInterval(2.0, 2.0),
        ratio_lower,
        ShatteringInterval(ratio_lower, 1.0),
        "defined",
        1.0,
        "context_conditioned",
        {},
    )


THRESHOLDS = CapacityThresholds(min_coherence=0.90, min_shattering_ratio=0.80)


def test_frontier_has_no_breakpoint_when_every_level_passes() -> None:
    frontier = find_capacity_breakpoint(
        ("W0", "W1"),
        {"W0": _coherence(1.0), "W1": _coherence(0.95)},
        {"W0": _complexity(None, zero_true=True), "W1": _complexity(0.85)},
        THRESHOLDS,
    )

    assert frontier.breakpoint is None
    assert [item.passed for item in frontier.levels] == [True, True]


def test_frontier_marks_coherence_only_crossing() -> None:
    frontier = find_capacity_breakpoint(
        ("W0",),
        {"W0": _coherence(0.89)},
        {"W0": _complexity(0.9)},
        THRESHOLDS,
    )

    assert frontier.breakpoint == "W0"
    assert frontier.levels[0].reasons == ("coherence_below_threshold",)


def test_frontier_marks_complexity_only_and_simultaneous_crossings() -> None:
    frontier = find_capacity_breakpoint(
        ("W0", "W1"),
        {"W0": _coherence(0.95), "W1": _coherence(0.85)},
        {"W0": _complexity(0.79), "W1": _complexity(0.70)},
        THRESHOLDS,
    )

    assert frontier.breakpoint == "W0"
    assert frontier.levels[0].reasons == ("shattering_ratio_below_threshold",)
    assert frontier.levels[1].reasons == (
        "coherence_below_threshold",
        "shattering_ratio_below_threshold",
    )


def test_insufficient_data_is_explicit_but_not_a_measured_breakpoint() -> None:
    frontier = find_capacity_breakpoint(
        ("W0",),
        {"W0": _coherence(None)},
        {"W0": _complexity(0.9)},
        THRESHOLDS,
    )

    assert frontier.breakpoint is None
    assert frontier.levels[0].passed is None
    assert frontier.levels[0].reasons == ("insufficient_data",)


def test_missing_complexity_interval_is_not_reported_as_a_failure() -> None:
    unavailable = ComplexityRecoveryEstimate(
        (),
        (),
        0.0,
        0.0,
        ShatteringInterval(None, None),
        ShatteringInterval(None, None),
        None,
        ShatteringInterval(None, None),
        "insufficient_data",
        0.0,
        "context_conditioned",
        {},
    )

    frontier = find_capacity_breakpoint(
        ("W0",),
        {"W0": _coherence(0.95)},
        {"W0": unavailable},
        THRESHOLDS,
    )

    assert frontier.breakpoint is None
    assert frontier.levels[0].passed is None
    assert frontier.levels[0].reasons == ("insufficient_data",)


def test_zero_true_dimension_is_descriptive_and_does_not_fail_ratio() -> None:
    frontier = find_capacity_breakpoint(
        ("W0",),
        {"W0": _coherence(0.95)},
        {"W0": _complexity(None, zero_true=True)},
        THRESHOLDS,
    )

    assert frontier.breakpoint is None
    assert frontier.levels[0].passed is True
    assert frontier.levels[0].shattering_ratio_lower is None


def test_first_crossing_survives_non_monotonic_later_recovery() -> None:
    frontier = find_capacity_breakpoint(
        ("W0", "W1", "W2"),
        {level: _coherence(0.95) for level in ("W0", "W1", "W2")},
        {
            "W0": _complexity(0.9),
            "W1": _complexity(0.7),
            "W2": _complexity(0.9),
        },
        THRESHOLDS,
    )

    assert frontier.breakpoint == "W1"
    assert frontier.warnings == ("non_monotonic_recovery_after_breakpoint",)


def test_non_finite_coherence_is_insufficient_not_a_pass() -> None:
    invalid = TemporalCoherenceReport(
        ("event",),
        ("probe",),
        np.asarray([np.nan]),
        float("nan"),
        1.0,
        1.0,
        1,
        1,
        "ok",
    )

    frontier = find_capacity_breakpoint(
        ("W0",),
        {"W0": invalid},
        {"W0": _complexity(0.9)},
        THRESHOLDS,
    )

    assert frontier.breakpoint is None
    assert frontier.levels[0].passed is None
    assert frontier.levels[0].reasons == ("insufficient_data",)


def test_non_finite_complexity_bound_is_insufficient_not_a_pass() -> None:
    invalid = _complexity(float("nan"))

    frontier = find_capacity_breakpoint(
        ("W0",),
        {"W0": _coherence(0.95)},
        {"W0": invalid},
        THRESHOLDS,
    )

    assert frontier.breakpoint is None
    assert frontier.levels[0].passed is None
    assert frontier.levels[0].reasons == ("insufficient_data",)
