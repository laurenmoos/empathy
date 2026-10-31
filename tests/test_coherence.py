import numpy as np
import pytest

from empathy.reasoning.coherence import compute_temporal_coherence


def test_coherence_reports_perfect_agreement_and_independent_coverage() -> None:
    true = np.asarray([[0, 1], [1, 0]], dtype=bool)
    predicted = true.copy()

    report = compute_temporal_coherence(
        true,
        predicted,
        np.asarray([True, True]),
        np.asarray([True, False]),
        ("event-0", "event-1"),
        ("probe-0", "probe-1"),
    )

    assert report.status == "ok"
    assert report.aggregate_coherence == 1.0
    assert report.outcome_coverage == 0.5
    assert report.per_event_agreement.tolist() == [1.0, 1.0]
    assert report.valid_event_count == 2
    assert report.valid_judgment_count == 4


def test_coherence_counts_one_disagreement_among_eight_judgments() -> None:
    true = np.zeros((2, 4), dtype=bool)
    predicted = true.copy()
    predicted[1, 3] = True

    report = compute_temporal_coherence(
        true,
        predicted,
        np.asarray([True, True]),
        np.asarray([False, False]),
        ("a", "b"),
        ("p0", "p1", "p2", "p3"),
    )

    assert report.aggregate_coherence == 0.875
    assert report.per_event_agreement.tolist() == [1.0, 0.75]


def test_coherence_excludes_invalid_events_and_measures_prediction_flips() -> None:
    true = np.asarray([[0, 0], [1, 1], [0, 0]], dtype=bool)
    predicted = np.asarray([[0, 0], [0, 0], [1, 0]], dtype=bool)

    report = compute_temporal_coherence(
        true,
        predicted,
        np.asarray([True, False, True]),
        np.asarray([True, True, False]),
        ("a", "ignored", "c"),
        ("left", "right"),
    )

    assert report.aggregate_coherence == 0.75
    assert np.isnan(report.per_event_agreement[1])
    assert report.outcome_coverage == 0.5
    assert report.prediction_stability == 0.5


def test_coherence_returns_insufficient_data_for_empty_valid_mask() -> None:
    report = compute_temporal_coherence(
        np.zeros((2, 1), dtype=bool),
        np.zeros((2, 1), dtype=bool),
        np.asarray([False, False]),
        np.asarray([False, False]),
        ("a", "b"),
        ("p",),
    )

    assert report.status == "insufficient_data"
    assert report.aggregate_coherence is None
    assert report.outcome_coverage is None
    assert report.prediction_stability is None
    assert report.valid_event_count == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("true", np.zeros((1, 1), dtype=float)),
        ("predicted", np.zeros((1, 1), dtype=np.int64)),
        ("valid", np.asarray([1], dtype=np.int64)),
        ("covered", np.asarray([0.0])),
    ],
)
def test_coherence_rejects_non_boolean_inputs(field: str, value: np.ndarray) -> None:
    values = {
        "true": np.zeros((1, 1), dtype=bool),
        "predicted": np.zeros((1, 1), dtype=bool),
        "valid": np.asarray([True]),
        "covered": np.asarray([False]),
    }
    values[field] = value

    with pytest.raises(ValueError, match="boolean"):
        compute_temporal_coherence(
            values["true"],
            values["predicted"],
            values["valid"],
            values["covered"],
            ("event",),
            ("probe",),
        )


def test_coherence_rejects_mismatched_identifiers() -> None:
    with pytest.raises(ValueError, match="identifiers"):
        compute_temporal_coherence(
            np.zeros((1, 2), dtype=bool),
            np.zeros((1, 2), dtype=bool),
            np.asarray([True]),
            np.asarray([True]),
            ("event",),
            ("only-one-probe",),
        )


def test_coherence_rejects_empty_probe_axis_instead_of_returning_nan() -> None:
    with pytest.raises(ValueError, match="at least one probe"):
        compute_temporal_coherence(
            np.zeros((1, 0), dtype=bool),
            np.zeros((1, 0), dtype=bool),
            np.asarray([True]),
            np.asarray([True]),
            ("event",),
            (),
        )
