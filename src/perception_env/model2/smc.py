"""Belief-only sequential Monte Carlo search over simple physical worlds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception_env.config import SimulationConfig
from perception_env.simulate import BallState, rollout
from perception_env.world import WorldCode, WorldSchema

from .contracts import BeliefInputs


@dataclass(frozen=True)
class SMCConfig:
    particle_count: int = 32
    rounds: int = 2
    seed: int = 7
    grid_size: int = 8
    sinkhorn_iterations: int = 16
    sinkhorn_epsilon: float = 0.08
    signed_change_weight: float = 0.35
    score_temperature: float = 8.0
    jitter_scale: float = 0.04

    def __post_init__(self) -> None:
        integers = (
            self.particle_count,
            self.rounds,
            self.grid_size,
            self.sinkhorn_iterations,
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("SMC counts must be positive integers")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("SMC seed must be a nonnegative integer")
        if self.grid_size < 2:
            raise ValueError("SMC transport grid must be at least two")
        floats = (
            self.sinkhorn_epsilon,
            self.signed_change_weight,
            self.score_temperature,
            self.jitter_scale,
        )
        if not all(np.isfinite(value) and value > 0 for value in floats):
            raise ValueError("SMC scales must be finite and positive")


@dataclass(frozen=True, eq=False)
class SMCResult:
    ranked_worlds: np.ndarray
    weights: np.ndarray
    best_world: np.ndarray
    event_discrepancies: np.ndarray


def _world_from_particle(values: np.ndarray) -> WorldCode:
    gravity, wind, damping = (float(value) for value in values)
    active: dict[str, dict[str, float]] = {}
    if abs(gravity) > 0.05:
        active["vertical_acceleration"] = {"a_y": gravity}
    if abs(wind) > 0.05:
        active["horizontal_acceleration"] = {"a_x": wind}
    if damping > 0.01:
        active["damping"] = {"lambda": damping}
    return WorldSchema.v1().world(**active)


def _initial_particles(config: SMCConfig, rng: np.random.Generator) -> np.ndarray:
    anchors = np.asarray(
        [
            [-0.30, 0.0, 0.0],
            [0.30, 0.0, 0.0],
            [0.0, -0.30, 0.0],
            [0.0, 0.30, 0.0],
            [0.0, 0.0, 0.30],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    particles = np.zeros((config.particle_count, 3), dtype=np.float64)
    count = min(len(anchors), config.particle_count)
    particles[:count] = anchors[:count]
    if count < config.particle_count:
        remaining = config.particle_count - count
        values = np.column_stack(
            (
                rng.uniform(-0.30, 0.30, remaining),
                rng.uniform(-0.30, 0.30, remaining),
                rng.uniform(0.0, 0.60, remaining),
            )
        )
        active = rng.random((remaining, 3)) < 0.5
        particles[count:] = np.where(active, values, 0.0)
    return particles


def _entry_states(observed: np.ndarray) -> tuple[BallState, ...]:
    mass = np.clip(np.asarray(observed, dtype=np.float64), 0.0, None)
    rows, columns = np.nonzero(mass > 0)
    if len(columns):
        entry_column = int(columns.min())
        near_entry = columns <= entry_column + 1
        weights = mass[rows[near_entry], columns[near_entry]]
        entry_row = float(np.average(rows[near_entry], weights=weights))
        x = entry_column / max(observed.shape[1] - 1, 1)
        y = 1.0 - entry_row / max(observed.shape[0] - 1, 1)
    else:
        x, y = 0.34, 0.50
    return tuple(
        BallState(x, float(np.clip(y + y_offset, 0.02, 0.98)), vx, vy)
        for y_offset in (-0.015, 0.0, 0.015)
        for vx in (0.18, 0.24, 0.30)
        for vy in (-0.02, 0.0, 0.02)
    )


def _candidate_field(
    world: WorldCode, height: int, width: int, observed: np.ndarray
) -> np.ndarray:
    field = np.zeros((height, width), dtype=np.float64)
    simulation = SimulationConfig(dt=0.1, steps=12, seed=0)
    accepted = 0
    for initial in _entry_states(observed):
        trajectory = rollout(initial, world, simulation)
        if not trajectory.valid:
            continue
        accepted += 1
        for state in trajectory.states:
            column = int(round(state.x * (width - 1)))
            row = int(round((1.0 - state.y) * (height - 1)))
            if 0 <= row < height and 0 <= column < width:
                field[row, column] += 1.0
    if accepted:
        field /= accepted
    return field


def _downsample(values: np.ndarray, size: int) -> np.ndarray:
    height, width = values.shape
    rows = np.linspace(0, height, size + 1, dtype=int)
    columns = np.linspace(0, width, size + 1, dtype=int)
    output = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        for column in range(size):
            output[row, column] = values[
                rows[row] : rows[row + 1], columns[column] : columns[column + 1]
            ].sum()
    return output


def _distribution(values: np.ndarray) -> np.ndarray:
    flat = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), 0.0, None)
    total = float(flat.sum())
    if total <= 1.0e-12:
        return np.full(len(flat), 1.0 / len(flat), dtype=np.float64)
    return flat / total


def _transport_cost(
    left: np.ndarray, right: np.ndarray, config: SMCConfig
) -> float:
    left_dist = _distribution(_downsample(left, config.grid_size))
    right_dist = _distribution(_downsample(right, config.grid_size))
    axis = np.linspace(0.0, 1.0, config.grid_size)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    cost = ((points[:, None] - points[None, :]) ** 2).sum(axis=2)
    kernel = np.exp(-cost / config.sinkhorn_epsilon)
    u = np.ones_like(left_dist)
    v = np.ones_like(right_dist)
    for _ in range(config.sinkhorn_iterations):
        u = left_dist / np.maximum(kernel @ v, 1.0e-12)
        v = right_dist / np.maximum(kernel.T @ u, 1.0e-12)
    transport = u[:, None] * kernel * v[None, :]
    return float((transport * cost).sum())


def _event_discrepancy(
    candidate: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    difference: np.ndarray,
    config: SMCConfig,
) -> float:
    spatial = _transport_cost(candidate, post, config)
    candidate_scale = candidate / max(float(candidate.max()), 1.0e-12)
    signed = np.mean(np.abs((candidate_scale - pre) - difference))
    return float(spatial + config.signed_change_weight * signed)


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    return weights / total


class BeliefWorldSMC:
    def __init__(self, config: SMCConfig | None = None) -> None:
        self.config = config or SMCConfig()

    def infer(self, inputs: BeliefInputs) -> SMCResult:
        if len(inputs.beliefs) != 1:
            raise ValueError("live SMC inference expects one belief history")
        rng = np.random.default_rng(self.config.seed)
        particles = _initial_particles(self.config, rng)
        valid = np.flatnonzero(inputs.history_mask[0])
        discrepancies = np.empty((len(particles), len(valid)), dtype=np.float64)
        weights = np.full(len(particles), 1.0 / len(particles), dtype=np.float64)
        for round_index in range(self.config.rounds):
            for particle_index, particle in enumerate(particles):
                world = _world_from_particle(particle)
                for event_index, belief_index in enumerate(valid):
                    pre, post, difference = inputs.beliefs[0, belief_index]
                    field = _candidate_field(
                        world,
                        inputs.beliefs.shape[-2],
                        inputs.beliefs.shape[-1],
                        np.maximum(pre, post),
                    )
                    discrepancies[particle_index, event_index] = _event_discrepancy(
                        field, pre, post, difference, self.config
                    )
            log_weights = -self.config.score_temperature * discrepancies.sum(axis=1)
            weights = _normalize_log_weights(log_weights)
            if round_index + 1 == self.config.rounds:
                break
            selected = rng.choice(len(particles), len(particles), replace=True, p=weights)
            particles = particles[selected].copy()
            particles += rng.normal(0.0, self.config.jitter_scale, particles.shape)
            particles[:, :2] = np.clip(particles[:, :2], -0.30, 0.30)
            particles[:, 2] = np.clip(particles[:, 2], 0.0, 0.60)
            particles[:, :2][np.abs(particles[:, :2]) <= 0.05] = 0.0
            particles[:, 2][particles[:, 2] <= 0.01] = 0.0
        worlds = np.stack(
            [_world_from_particle(particle).to_flat(WorldSchema.v1()) for particle in particles]
        ).astype(np.float32)
        order = np.argsort(-weights, kind="stable")
        ranked_worlds = worlds[order]
        ranked_weights = weights[order]
        ranked_discrepancies = discrepancies[order]
        return SMCResult(
            ranked_worlds=ranked_worlds,
            weights=ranked_weights,
            best_world=ranked_worlds[0].copy(),
            event_discrepancies=ranked_discrepancies,
        )
