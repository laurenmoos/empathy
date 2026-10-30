"""Project network outputs to valid executable world codes without truth access."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception_env.world import WorldCode, WorldSchema

from .contracts import WorldPrediction


@dataclass(frozen=True)
class DecodedWorlds:
    worlds: tuple[WorldCode, ...]
    flat: np.ndarray
    projection_counts: np.ndarray


def decode_prediction(
    prediction: WorldPrediction, schema: WorldSchema, *, activity_threshold: float = 0.5
) -> DecodedWorlds:
    if not 0 < activity_threshold < 1:
        raise ValueError("activity threshold must lie strictly between 0 and 1")
    logits = prediction.activity_logits.detach().cpu().numpy()
    parameters = prediction.normalized_parameters.detach().cpu().numpy()
    if (
        logits.ndim != 2
        or logits.shape[1] != len(schema.specs)
        or parameters.shape != (len(logits), sum(len(s.parameters) for s in schema.specs))
        or not np.isfinite(logits).all()
        or not np.isfinite(parameters).all()
    ):
        raise ValueError("invalid prediction shape or nonfinite values")
    threshold = np.log(activity_threshold / (1 - activity_threshold))
    worlds, counts = [], []
    for row, parameter_row in zip(logits, parameters, strict=True):
        active, cursor, projected = {}, 0, 0
        for index, spec in enumerate(schema.specs):
            values = {}
            for parameter in spec.parameters:
                value = parameter.denormalize(float(np.clip(parameter_row[cursor], 0, 1)))
                cursor += 1
                if row[index] < threshold:
                    continue
                if parameter.neutral_interval is not None:
                    low, high = parameter.neutral_interval
                    if low <= value <= high:
                        choices = [
                            v
                            for v in (low - 1e-6, high + 1e-6)
                            if parameter.minimum <= v <= parameter.maximum
                        ]
                        value = min(choices, key=lambda v: (abs(v - value), v))
                        projected += 1
                values[parameter.name] = value
            if row[index] >= threshold:
                active[spec.name] = values
        worlds.append(schema.world(**active))
        counts.append(projected)
    flat = (
        np.stack([world.to_flat(schema) for world in worlds])
        if worlds
        else np.empty((0, len(schema.field_order)))
    )
    return DecodedWorlds(tuple(worlds), flat, np.asarray(counts, dtype=np.int64))
