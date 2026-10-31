"""Constraint recovery diagnostics evaluated only after belief-only inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from perception_env.world import WorldSchema


@dataclass(frozen=True)
class ConstraintStatus:
    name: str
    recovered: bool
    activity_match: bool
    parameter_error: float | None


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
        parameter_error: float | None = None
        recovered = activity_match
        if activity_match and actual_slot.active:
            spec = schema.spec(slot_name)
            errors = [
                abs(parameter.normalize(actual_value) - parameter.normalize(predicted_value))
                for parameter, actual_value, predicted_value in zip(
                    spec.parameters,
                    actual_slot.parameters,
                    predicted_slot.parameters,
                    strict=True,
                )
            ]
            parameter_error = max(errors)
            recovered = parameter_error <= tolerance
        statuses.append(
            ConstraintStatus(label, recovered, activity_match, parameter_error)
        )
    recovered_count = sum(item.recovered for item in statuses)
    return ConstraintRecovery(
        recovered=recovered_count,
        total=len(statuses),
        score=recovered_count / len(statuses),
        constraints=tuple(statuses),
    )
