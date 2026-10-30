"""Versioned nested world families and reproducible sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .reachability import EntryStateGrid
from .world import WorldCode, WorldSchema


@dataclass(frozen=True)
class WorldFamily:
    schema: WorldSchema
    family_version: str
    level: str
    allowed: tuple[str, ...]
    max_active: int
    max_walls: int
    operator_config: dict[str, Any]
    entry_grid: EntryStateGrid
    invalid_rollout_policy: str

    def contains(self, world: WorldCode) -> bool:
        try:
            self.schema.validate(world)
        except ValueError:
            return False
        active = [slot.name for slot in world.slots if slot.active]
        if not set(active).issubset(self.allowed) or len(active) > self.max_active:
            return False
        wall_count = sum(name in {"vertical_wall", "horizontal_wall"} for name in active)
        return wall_count <= self.max_walls


def load_family(path: Path, level: str) -> WorldFamily:
    payload = yaml.safe_load(Path(path).read_text())
    if payload["schema_version"] != "v1":
        raise ValueError(f"unsupported schema version {payload['schema_version']!r}")
    if level not in payload["levels"]:
        raise ValueError(f"unknown family level {level!r}")
    level_config = payload["levels"][level]
    uncertainty = payload["entry_uncertainty"]
    return WorldFamily(
        schema=WorldSchema.v1(),
        family_version=str(payload["family_version"]),
        level=level,
        allowed=tuple(level_config["allowed"]),
        max_active=int(level_config["max_active"]),
        max_walls=int(level_config["max_walls"]),
        operator_config=dict(payload["operators"]),
        entry_grid=EntryStateGrid(
            tuple(uncertainty["x_bounds"]),
            tuple(uncertainty["y_bounds"]),
            tuple(uncertainty["vx_bounds"]),
            tuple(uncertainty["vy_bounds"]),
            tuple(uncertainty["resolution"]),
        ),
        invalid_rollout_policy=str(payload["invalid_rollout_policy"]),
    )


def _sample_parameter(config: dict[str, Any], rng: np.random.Generator) -> float:
    low, high = (float(value) for value in config["range"])
    neutral = config.get("neutral_interval")
    for _ in range(100):
        value = float(rng.uniform(low, high))
        if neutral is None or not float(neutral[0]) <= value <= float(neutral[1]):
            return value
    raise RuntimeError("could not sample outside configured neutral interval")


def sample_worlds(family: WorldFamily, count: int, seed: int) -> list[WorldCode]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    rng = np.random.default_rng(seed)
    worlds: list[WorldCode] = []
    allowed = list(family.allowed)
    for _ in range(count):
        while True:
            active_count = int(rng.integers(0, family.max_active + 1))
            names = list(rng.choice(allowed, size=active_count, replace=False)) if active_count else []
            wall_count = sum(name in {"vertical_wall", "horizontal_wall"} for name in names)
            if wall_count <= family.max_walls:
                break
        slots: dict[str, dict[str, float]] = {}
        for name in names:
            slots[str(name)] = {
                parameter_name: _sample_parameter(parameter_config, rng)
                for parameter_name, parameter_config in family.operator_config[str(name)].items()
            }
        world = family.schema.world(**slots)
        if not family.contains(world):
            raise RuntimeError("sampled world does not belong to requested family")
        worlds.append(world)
    return worlds

