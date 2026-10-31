"""Typed, explicit configuration for offline Model 2 experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


def _positive(**values) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _integer(**values) -> None:
    for name, value in values.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class FixtureConfig:
    seed: int = 7
    family_path: str = "configs/world_families/v1.yaml"
    levels: tuple[str, ...] = ("W1", "W2", "W3", "W4", "W5")
    world_count: int = 120
    episodes_per_world: int = 4
    history_length: int = 4
    size: int = 64
    split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    dt: float = 0.05
    steps: int = 80
    max_context_attempts: int = 200
    world_attempt_multiplier: int = 20
    entry_resolution: tuple[int, int, int, int] = (3, 3, 3, 3)
    tolerance: float = 0.025
    audit_bank_size: int = 64
    audit_probe_count: int = 64
    heldout_probe_count: int = 128
    oracle_fraction: float = 0.25
    x_bounds: tuple[float, float] = (0.10, 0.25)
    y_bounds: tuple[float, float] = (0.25, 0.75)
    vx_bounds: tuple[float, float] = (0.15, 0.35)
    vy_bounds: tuple[float, float] = (-0.10, 0.10)
    occluder: tuple[float, float, float, float] = (0.35, 0.65, 0.05, 0.95)

    def __post_init__(self) -> None:
        _integer(
            seed=self.seed,
            world_count=self.world_count,
            episodes=self.episodes_per_world,
            history_length=self.history_length,
            size=self.size,
            steps=self.steps,
            attempts=self.max_context_attempts,
            multiplier=self.world_attempt_multiplier,
            audit_bank=self.audit_bank_size,
            audit_probes=self.audit_probe_count,
            heldout_probes=self.heldout_probe_count,
        )
        _positive(
            world_count=self.world_count,
            episodes=self.episodes_per_world,
            dt=self.dt,
            steps=self.steps,
            attempts=self.max_context_attempts,
            multiplier=self.world_attempt_multiplier,
            tolerance=self.tolerance,
            audit_bank=self.audit_bank_size,
            audit_probes=self.audit_probe_count,
            heldout_probes=self.heldout_probe_count,
        )
        if self.history_length != 4 or self.size < 8 or not self.levels:
            raise ValueError("four events, size>=8, and nonempty levels required")
        if (
            len(self.split_fractions) != 3
            or min(self.split_fractions) <= 0
            or not math.isclose(sum(self.split_fractions), 1.0)
        ):
            raise ValueError("split fractions must be three positive values summing to one")
        if len(self.entry_resolution) != 4 or any(
            type(v) is not int or v < 1 or v % 2 == 0 for v in self.entry_resolution
        ):
            raise ValueError(
                "entry_resolution needs four positive odd integers (retain the true state)"
            )
        if not 0 <= self.oracle_fraction <= 1:
            raise ValueError("oracle_fraction must lie in [0,1]")
        for bounds in (self.x_bounds, self.y_bounds, self.vx_bounds, self.vy_bounds):
            if len(bounds) != 2 or not all(map(math.isfinite, bounds)) or bounds[0] > bounds[1]:
                raise ValueError("invalid initial-state bounds")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    activity_weight: float = 1.0
    parameter_weight: float = 1.0
    device: str = "cpu"
    num_workers: int = 0
    input_mode: str = "full"

    def __post_init__(self) -> None:
        _integer(
            seed=self.seed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        _positive(epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate)
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.weight_decay, self.activity_weight, self.parameter_weight)
        ):
            raise ValueError("loss weights and weight decay must be finite and nonnegative")
        if self.input_mode not in {"full", "single_belief", "shuffled_history"}:
            raise ValueError("unknown input control mode")
        if self.num_workers != 0:
            raise ValueError("num_workers=0 is required for deterministic reference training")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be cpu, cuda, or mps")


@dataclass(frozen=True)
class EvaluationConfig:
    search_seeds: tuple[int, ...] = (7, 19, 31)
    max_n: int = 3
    budget: int = 100

    def __post_init__(self) -> None:
        _integer(max_n=self.max_n, budget=self.budget)
        if self.budget < 1:
            raise ValueError("search budget must be positive")
        if not isinstance(self.search_seeds, (tuple, list)) or not self.search_seeds:
            raise ValueError("search_seeds must be a nonempty sequence")
        for seed in self.search_seeds:
            _integer(search_seed=seed)
        if len(set(self.search_seeds)) != len(self.search_seeds):
            raise ValueError("search_seeds must be distinct")
        object.__setattr__(self, "search_seeds", tuple(self.search_seeds))


def load_config(path: Path) -> tuple[FixtureConfig, TrainingConfig, EvaluationConfig]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or set(payload) != {"fixtures", "training", "evaluation"}:
        raise ValueError("config sections must be fixtures, training, evaluation")
    result = []
    for name, cls in (
        ("fixtures", FixtureConfig),
        ("training", TrainingConfig),
        ("evaluation", EvaluationConfig),
    ):
        supplied = payload[name]
        if not isinstance(supplied, dict) or set(supplied) - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown fields in {name}")
        defaults = cls()
        supplied = {
            k: tuple(v) if isinstance(getattr(defaults, k), tuple) else v
            for k, v in supplied.items()
        }
        result.append(cls(**supplied))
    return tuple(result)
