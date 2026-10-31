"""Constraint recovery diagnostics evaluated only after belief-only inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from empathy.physics.world import WorldSchema


@dataclass(frozen=True)
class ConstraintStatus:
    name: str
    recovered: bool
    activity_match: bool
    parameter_error: float | None
    closeness: float


@dataclass(frozen=True)
class ConstraintRecovery:
    recovered: int
    total: int
    score: float
    constraints: tuple[ConstraintStatus, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "recovered": self.recovered,
            "total": self.total,
            "score": self.score,
            "constraints": [asdict(item) for item in self.constraints],
        }


def evaluate_constraint_recovery(
    actual_flat: np.ndarray,
    predicted_flat: np.ndarray,
    *,
    tolerance: float = 0.10,
) -> ConstraintRecovery:
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("constraint recovery tolerance must lie in [0, 1]")
    schema = WorldSchema.v1()
    actual = schema.from_flat(np.asarray(actual_flat, dtype=np.float64))
    predicted = schema.from_flat(np.asarray(predicted_flat, dtype=np.float64))
    names = (
        ("gravity", "vertical_acceleration"),
        ("wind", "horizontal_acceleration"),
        ("damping", "damping"),
    )
    statuses: list[ConstraintStatus] = []
    for label, slot_name in names:
        actual_slot = actual.slot(slot_name)
        predicted_slot = predicted.slot(slot_name)
        activity_match = actual_slot.active == predicted_slot.active
        spec = schema.spec(slot_name)
        parameter = spec.parameters[0]
        actual_value = actual_slot.parameters[0] if actual_slot.active else 0.0
        predicted_value = predicted_slot.parameters[0] if predicted_slot.active else 0.0
        axis_error = abs(
            parameter.normalize(actual_value) - parameter.normalize(predicted_value)
        )
        closeness = float(np.clip(1.0 - axis_error, 0.0, 1.0))
        parameter_error: float | None = (
            axis_error if actual_slot.active or predicted_slot.active else None
        )
        recovered = activity_match
        if activity_match and actual_slot.active:
            recovered = axis_error <= tolerance
        statuses.append(
            ConstraintStatus(label, recovered, activity_match, parameter_error, closeness)
        )
    recovered_count = sum(item.recovered for item in statuses)
    return ConstraintRecovery(
        recovered=recovered_count,
        total=len(statuses),
        score=float(np.mean([item.closeness for item in statuses])),
        constraints=tuple(statuses),
    )
