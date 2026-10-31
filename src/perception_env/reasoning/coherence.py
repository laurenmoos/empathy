"""Temporal possibility coherence and operator-capacity frontiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .shattering import ComplexityRecoveryEstimate


@dataclass(frozen=True, eq=False)
class TemporalCoherenceReport:
    event_ids: tuple[str, ...]
    probe_ids: tuple[str, ...]
    per_event_agreement: np.ndarray
    aggregate_coherence: float | None
    outcome_coverage: float | None
    prediction_stability: float | None
    valid_event_count: int
    valid_judgment_count: int
    status: str

    def __post_init__(self) -> None:
        values = np.asarray(self.per_event_agreement, dtype=np.float64)
        if values.shape != (len(self.event_ids),):
            raise ValueError("per-event agreement must match event identifiers")
        object.__setattr__(self, "per_event_agreement", values.copy())


def _require_boolean(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array")
    return array


def compute_temporal_coherence(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    valid_events: np.ndarray,
    outcome_covered: np.ndarray,
    event_ids: Sequence[str],
    probe_ids: Sequence[str],
) -> TemporalCoherenceReport:
    """Compare possible/impossible judgments across valid stream events."""
    truth = _require_boolean("true_labels", true_labels)
    predicted = _require_boolean("predicted_labels", predicted_labels)
    valid = _require_boolean("valid_events", valid_events)
    covered = _require_boolean("outcome_covered", outcome_covered)
    if truth.ndim != 2 or predicted.shape != truth.shape:
        raise ValueError("true and predicted labels must share shape [event, probe]")
    if valid.shape != (truth.shape[0],) or covered.shape != (truth.shape[0],):
        raise ValueError("event masks must match the event dimension")
    if len(event_ids) != truth.shape[0] or len(probe_ids) != truth.shape[1]:
        raise ValueError("event and probe identifiers must match label dimensions")
    if truth.shape[1] == 0:
        raise ValueError("at least one probe is required")

    event_ids_tuple = tuple(str(value) for value in event_ids)
    probe_ids_tuple = tuple(str(value) for value in probe_ids)
    per_event = np.full(truth.shape[0], np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return TemporalCoherenceReport(
            event_ids_tuple,
            probe_ids_tuple,
            per_event,
            None,
            None,
            None,
            0,
            0,
            "insufficient_data",
        )

    agreement = truth[valid] == predicted[valid]
    per_event[valid] = agreement.mean(axis=1)
    if valid_indices.size == 1:
        stability = 1.0
    else:
        adjacent_equal = predicted[valid][1:] == predicted[valid][:-1]
        stability = float(adjacent_equal.mean())
    return TemporalCoherenceReport(
        event_ids=event_ids_tuple,
        probe_ids=probe_ids_tuple,
        per_event_agreement=per_event,
        aggregate_coherence=float(agreement.mean()),
        outcome_coverage=float(covered[valid].mean()),
        prediction_stability=stability,
        valid_event_count=int(valid.sum()),
        valid_judgment_count=int(valid.sum()) * truth.shape[1],
        status="ok",
    )


@dataclass(frozen=True)
class CapacityThresholds:
    min_coherence: float = 0.90
    min_shattering_ratio: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_coherence <= 1.0:
            raise ValueError("minimum coherence must lie in [0, 1]")
        if not 0.0 <= self.min_shattering_ratio <= 1.0:
            raise ValueError("minimum shattering ratio must lie in [0, 1]")


@dataclass(frozen=True)
class CapacityLevelResult:
    level: str
    passed: bool | None
    coherence: float | None
    shattering_ratio_lower: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CapacityFrontier:
    levels: tuple[CapacityLevelResult, ...]
    breakpoint: str | None
    warnings: tuple[str, ...]
    thresholds: CapacityThresholds


def find_capacity_breakpoint(
    levels: Sequence[str],
    coherence: Mapping[str, TemporalCoherenceReport],
    complexity: Mapping[str, ComplexityRecoveryEstimate],
    thresholds: CapacityThresholds,
) -> CapacityFrontier:
    """Return the first measured level below either registered threshold."""
    results: list[CapacityLevelResult] = []
    for level in levels:
        if level not in coherence or level not in complexity:
            raise ValueError(f"missing reports for level {level}")
        coherence_report = coherence[level]
        complexity_report = complexity[level]
        reasons: list[str] = []
        if (
            coherence_report.status != "ok"
            or coherence_report.aggregate_coherence is None
            or not np.isfinite(coherence_report.aggregate_coherence)
        ):
            reasons.append("insufficient_data")
            passed: bool | None = None
        else:
            if coherence_report.aggregate_coherence < thresholds.min_coherence:
                reasons.append("coherence_below_threshold")
            if complexity_report.ratio_status == "defined":
                ratio_lower = complexity_report.ratio_interval.lower
                if ratio_lower is None or not np.isfinite(ratio_lower):
                    reasons.append("insufficient_data")
                elif ratio_lower < thresholds.min_shattering_ratio:
                    reasons.append("shattering_ratio_below_threshold")
            elif complexity_report.ratio_status != "undefined_zero_true_dimension":
                reasons.append("insufficient_data")
            passed = None if "insufficient_data" in reasons else not reasons
        ratio_value = (
            complexity_report.ratio_interval.lower
            if complexity_report.ratio_status == "defined"
            else None
        )
        results.append(
            CapacityLevelResult(
                level=str(level),
                passed=passed,
                coherence=coherence_report.aggregate_coherence,
                shattering_ratio_lower=ratio_value,
                reasons=tuple(reasons),
            )
        )

    breakpoint = next((item.level for item in results if item.passed is False), None)
    warnings: list[str] = []
    if breakpoint is not None:
        breakpoint_index = next(
            index for index, item in enumerate(results) if item.level == breakpoint
        )
        if any(item.passed is True for item in results[breakpoint_index + 1 :]):
            warnings.append("non_monotonic_recovery_after_breakpoint")
    return CapacityFrontier(tuple(results), breakpoint, tuple(warnings), thresholds)
