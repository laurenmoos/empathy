import numpy as np
import pytest

from empathy.model2.coherence import evaluate_constraint_recovery
from empathy.physics.world import WorldSchema


def _flat(**operators) -> np.ndarray:
    return WorldSchema.v1().world(**operators).to_flat(WorldSchema.v1())


def test_constraint_recovery_counts_inactive_agreement_and_active_tolerance() -> None:
    actual = _flat(
        vertical_acceleration={"a_y": -0.20},
        damping={"lambda": 0.30},
    )
    predicted = _flat(
        vertical_acceleration={"a_y": -0.06},
        damping={"lambda": 0.45},
    )

    recovery = evaluate_constraint_recovery(actual, predicted, tolerance=0.10)

    assert recovery.recovered == 2
    assert recovery.total == 3
    assert recovery.score == pytest.approx((0.93 + 1.0 + 0.85) / 3)
    assert [item.name for item in recovery.constraints] == ["gravity", "wind", "damping"]
    assert [item.recovered for item in recovery.constraints] == [True, True, False]
    assert recovery.constraints[0].parameter_error == pytest.approx(0.07)
    assert recovery.constraints[1].parameter_error is None
    assert [item.closeness for item in recovery.constraints] == pytest.approx(
        [0.93, 1.0, 0.85]
    )


def test_constraint_recovery_marks_activity_mismatch_as_missed() -> None:
    actual = _flat(horizontal_acceleration={"a_x": 0.20})
    predicted = _flat()

    recovery = evaluate_constraint_recovery(actual, predicted)

    assert recovery.recovered == 2
    assert not recovery.constraints[1].activity_match
    assert not recovery.constraints[1].recovered


def test_constraint_recovery_rejects_invalid_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        evaluate_constraint_recovery(_flat(), _flat(), tolerance=-0.1)
