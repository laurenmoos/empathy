"""Perceptual-equivalence and parameter-identifiability audits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .shattering import LabelMatrix
from .world import WorldCode, WorldSchema


@dataclass(frozen=True)
class EquivalenceClass:
    label_pattern: tuple[bool, ...]
    world_indices: tuple[int, ...]
    exact_code_groups: tuple[tuple[int, ...], ...]
    status: str

    @property
    def unique_code_count(self) -> int:
        return len(self.exact_code_groups)


@dataclass(frozen=True)
class RecoveryComparison:
    exact_code: bool
    possibility_equivalent: bool


@dataclass(frozen=True, eq=False)
class IdentifiabilityReport:
    worlds: tuple[WorldCode, ...]
    labels: LabelMatrix
    classes: tuple[EquivalenceClass, ...]
    tolerance: float
    raster_iou: np.ndarray | None
    possibility_rasters: np.ndarray | None

    def compare(self, *, true_index: int, predicted_index: int) -> RecoveryComparison:
        schema = WorldSchema.v1()
        true_code = self.worlds[true_index].to_flat(schema)
        predicted_code = self.worlds[predicted_index].to_flat(schema)
        exact = bool(np.allclose(true_code, predicted_code, atol=self.tolerance, rtol=0.0))
        equivalent = bool(
            np.array_equal(
                self.labels.values[true_index], self.labels.values[predicted_index]
            )
        )
        return RecoveryComparison(exact_code=exact, possibility_equivalent=equivalent)


def _code_groups(
    indices: tuple[int, ...], codes: np.ndarray, tolerance: float
) -> tuple[tuple[int, ...], ...]:
    groups: list[list[int]] = []
    for index in indices:
        for group in groups:
            if np.allclose(codes[index], codes[group[0]], atol=tolerance, rtol=0.0):
                group.append(index)
                break
        else:
            groups.append([index])
    return tuple(tuple(group) for group in groups)


def _raster_iou(rasters: np.ndarray) -> np.ndarray:
    count = len(rasters)
    result = np.eye(count, dtype=float)
    for left in range(count):
        for right in range(left + 1, count):
            intersection = np.logical_and(rasters[left], rasters[right]).sum()
            union = np.logical_or(rasters[left], rasters[right]).sum()
            value = 1.0 if union == 0 else float(intersection / union)
            result[left, right] = result[right, left] = value
    return result


def group_equivalent_worlds(
    worlds: Sequence[WorldCode],
    labels: LabelMatrix,
    tolerance: float,
    *,
    possibility_rasters: np.ndarray | None = None,
) -> IdentifiabilityReport:
    if len(worlds) != labels.values.shape[0]:
        raise ValueError("world count must match label-matrix rows")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    schema = WorldSchema.v1()
    codes = np.stack([world.to_flat(schema) for world in worlds])
    classes: list[EquivalenceClass] = []
    patterns, inverse = np.unique(labels.values, axis=0, return_inverse=True)
    search_complete = bool(labels.metadata.get("search_complete", True))
    for class_index, pattern in enumerate(patterns):
        indices = tuple(int(index) for index in np.flatnonzero(inverse == class_index))
        code_groups = _code_groups(indices, codes, tolerance)
        if not search_complete:
            status = "unresolved"
        elif len(code_groups) == 1:
            status = "identifiable"
        else:
            status = "perceptually_equivalent"
        classes.append(
            EquivalenceClass(
                label_pattern=tuple(bool(value) for value in pattern),
                world_indices=indices,
                exact_code_groups=code_groups,
                status=status,
            )
        )
    raster_values: np.ndarray | None = None
    raster_iou: np.ndarray | None = None
    if possibility_rasters is not None:
        raster_values = np.asarray(possibility_rasters, dtype=bool)
        if raster_values.ndim != 3 or raster_values.shape[0] != len(worlds):
            raise ValueError("possibility rasters must have shape [world, height, width]")
        raster_iou = _raster_iou(raster_values)
    return IdentifiabilityReport(
        worlds=tuple(worlds),
        labels=labels,
        classes=tuple(classes),
        tolerance=tolerance,
        raster_iou=raster_iou,
        possibility_rasters=raster_values,
    )


def parameter_scoring_mask(report: IdentifiabilityReport) -> np.ndarray:
    mask = np.zeros(len(report.worlds), dtype=bool)
    for equivalence_class in report.classes:
        if equivalence_class.status == "identifiable":
            mask[list(equivalence_class.world_indices)] = True
    return mask

