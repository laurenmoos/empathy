"""Belief-only sequential Monte Carlo search over simple physical worlds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from empathy.physics.config import SimulationConfig
from empathy.physics.simulate import BallState, rollout
from empathy.physics.world import WorldCode, WorldSchema

from .activation_adapter import ActivationBeliefAdapter, Model1ActivationPacket
from .adapters import adapt_model1_activations
from .contracts import BeliefInputs
from .imagination import ActivationImaginationLoop, CounterfactualUnavailable


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
    progress: tuple[ActivationSearchProgress, ...] = ()
    parameter_particles: np.ndarray | None = None
    update_count: int = 0


@dataclass(frozen=True, eq=False)
class ActivationSearchProgress:
    round_index: int
    best_discrepancy: float
    mean_discrepancy: float
    effective_sample_size: float
    accepted_revisions: int
    best_parameters: np.ndarray
    posterior_mean: np.ndarray


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
            [-0.06, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.0, -0.06, 0.0],
            [0.0, 0.06, 0.0],
            [0.0, 0.0, 0.03],
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
    neutral: np.ndarray,
    post: np.ndarray,
    difference: np.ndarray,
    config: SMCConfig,
) -> float:
    spatial = _transport_cost(candidate, post, config)
    candidate_scale = candidate / max(float(candidate.max()), 1.0e-12)
    neutral_scale = neutral / max(float(neutral.max()), 1.0e-12)
    signed = np.mean(np.abs((candidate_scale - neutral_scale) - difference))
    return float(spatial + config.signed_change_weight * signed)


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    return weights / total


def weighted_parameter_summary(
    particles: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted mean and central 80% marginal intervals."""
    values = np.asarray(particles, dtype=np.float64)
    probabilities = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or probabilities.shape != (len(values),):
        raise ValueError("posterior particles must be [N,3] with one weight per row")
    if not np.isfinite(values).all() or not np.isfinite(probabilities).all():
        raise ValueError("posterior particles and weights must be finite")
    total = float(probabilities.sum())
    if total <= 0 or np.any(probabilities < 0):
        raise ValueError("posterior weights must be nonnegative with positive mass")
    probabilities = probabilities / total
    mean = np.average(values, axis=0, weights=probabilities)
    intervals = np.empty((3, 2), dtype=np.float64)
    for parameter in range(3):
        order = np.argsort(values[:, parameter], kind="stable")
        ordered = values[order, parameter]
        cumulative = np.cumsum(probabilities[order])
        for bound, quantile in enumerate((0.1, 0.9)):
            index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(ordered) - 1)
            intervals[parameter, bound] = ordered[index]
    return mean, intervals


def weighted_hypothesis_clusters(
    particles: np.ndarray,
    weights: np.ndarray,
    *,
    max_clusters: int = 3,
    radius: float = 0.22,
) -> tuple[tuple[np.ndarray, float], ...]:
    """Group the joint posterior into a few separated physical-world hypotheses."""
    values = np.asarray(particles, dtype=np.float64)
    probabilities = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or probabilities.shape != (len(values),):
        raise ValueError("hypothesis particles must be [N,3] with matching weights")
    if max_clusters < 1 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("hypothesis clustering requires positive limits")
    total = float(probabilities.sum())
    if total <= 0 or np.any(probabilities < 0):
        raise ValueError("hypothesis weights must be nonnegative with positive mass")
    probabilities = probabilities / total
    positive = probabilities > 0
    values = values[positive]
    probabilities = probabilities[positive]
    probabilities = probabilities / probabilities.sum()
    normalized = (values - np.array([-0.30, -0.30, 0.0])) / 0.60
    groups: list[list[int]] = []
    for index in np.argsort(-probabilities, kind="stable"):
        if not groups:
            groups.append([int(index)])
            continue
        centers = np.stack(
            [
                np.average(normalized[group], axis=0, weights=probabilities[group])
                for group in groups
            ]
        )
        distances = np.linalg.norm(centers - normalized[index], axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] > radius and len(groups) < max_clusters:
            groups.append([int(index)])
        else:
            groups[nearest].append(int(index))
    clusters = [
        (
            np.average(values[group], axis=0, weights=probabilities[group]),
            min(float(probabilities[group].sum()), 1.0),
        )
        for group in groups
    ]
    clusters.sort(key=lambda cluster: cluster[1], reverse=True)
    return tuple(clusters)


def _jitter_particles(
    particles: np.ndarray, config: SMCConfig, rng: np.random.Generator
) -> np.ndarray:
    proposals = particles + rng.normal(0.0, config.jitter_scale, particles.shape)
    proposals[:, :2] = np.clip(proposals[:, :2], -0.30, 0.30)
    proposals[:, 2] = np.clip(proposals[:, 2], 0.0, 0.60)
    proposals[:, :2][np.abs(proposals[:, :2]) <= 0.05] = 0.0
    proposals[:, 2][proposals[:, 2] <= 0.01] = 0.0
    return proposals


def _score_particles(
    particles: np.ndarray,
    inputs: BeliefInputs,
    belief_indices: np.ndarray,
    config: SMCConfig,
) -> np.ndarray:
    scores = np.empty((len(particles), len(belief_indices)), dtype=np.float64)
    neutral_world = _world_from_particle(np.zeros(3, dtype=np.float64))
    for event_index, belief_index in enumerate(belief_indices):
        pre, post, difference = inputs.beliefs[0, belief_index]
        observed = np.maximum(pre, post)
        neutral = _candidate_field(
            neutral_world,
            inputs.beliefs.shape[-2],
            inputs.beliefs.shape[-1],
            observed,
        )
        for particle_index, particle in enumerate(particles):
            candidate = _candidate_field(
                _world_from_particle(particle),
                inputs.beliefs.shape[-2],
                inputs.beliefs.shape[-1],
                observed,
            )
            scores[particle_index, event_index] = _event_discrepancy(
                candidate, neutral, post, difference, config
            )
    return scores


class BeliefWorldSMC:
    def __init__(self, config: SMCConfig | None = None) -> None:
        self.config = config or SMCConfig()

    def infer(self, inputs: BeliefInputs) -> SMCResult:
        if len(inputs.beliefs) != 1:
            raise ValueError("live SMC inference expects one belief history")
        rng = np.random.default_rng(self.config.seed)
        particles = _initial_particles(self.config, rng)
        valid = np.flatnonzero(inputs.history_mask[0])
        weights = np.full(len(particles), 1.0 / len(particles), dtype=np.float64)
        log_weights = np.zeros(len(particles), dtype=np.float64)
        for event_position in range(len(valid)):
            event_score = _score_particles(
                particles, inputs, valid[event_position : event_position + 1], self.config
            )[:, 0]
            log_weights += -self.config.score_temperature * event_score
            weights = _normalize_log_weights(log_weights)
            if event_position + 1 == len(valid):
                break
            selected = rng.choice(len(particles), len(particles), replace=True, p=weights)
            particles = particles[selected].copy()
            log_weights.fill(0.0)
            processed = valid[: event_position + 1]
            for _ in range(self.config.rounds - 1):
                proposals = _jitter_particles(particles, self.config, rng)
                current_score = _score_particles(
                    particles, inputs, processed, self.config
                ).sum(axis=1)
                proposal_score = _score_particles(
                    proposals, inputs, processed, self.config
                ).sum(axis=1)
                log_acceptance = -self.config.score_temperature * (
                    proposal_score - current_score
                )
                accepted = np.log(rng.random(len(particles))) < np.minimum(
                    0.0, log_acceptance
                )
                particles[accepted] = proposals[accepted]
        discrepancies = _score_particles(particles, inputs, valid, self.config)
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

    def infer_activations(
        self,
        events: Sequence[Model1ActivationPacket],
        adapter: ActivationBeliefAdapter,
    ) -> SMCResult:
        return self.infer(adapt_model1_activations(events, adapter))


class ContinualBeliefWorldSMC:
    """Sequential belief-only posterior that persists across Model 1 versions."""

    def __init__(self, config: SMCConfig | None = None) -> None:
        self.config = config or SMCConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._particles: np.ndarray | None = None
        self._log_weights: np.ndarray | None = None
        self.update_count = 0
        self.initialization_count = 0

    def update(self, inputs: BeliefInputs) -> SMCResult:
        if len(inputs.beliefs) != 1:
            raise ValueError("live SMC inference expects one belief history")
        if self._particles is None:
            self._particles = _initial_particles(self.config, self._rng)
            self._log_weights = np.zeros(len(self._particles), dtype=np.float64)
            self.initialization_count += 1
        assert self._log_weights is not None
        valid = np.flatnonzero(inputs.history_mask[0])
        latest = valid[-1:]
        discrepancies = _score_particles(
            self._particles, inputs, latest, self.config
        )
        self._log_weights += -self.config.score_temperature * discrepancies[:, 0]
        weights = _normalize_log_weights(self._log_weights)
        self.update_count += 1

        worlds = np.stack(
            [
                _world_from_particle(particle).to_flat(WorldSchema.v1())
                for particle in self._particles
            ]
        ).astype(np.float32)
        order = np.argsort(-weights, kind="stable")
        result = SMCResult(
            ranked_worlds=worlds[order],
            weights=weights[order],
            best_world=worlds[order][0].copy(),
            event_discrepancies=discrepancies[order],
            parameter_particles=self._particles[order].copy(),
            update_count=self.update_count,
        )

        effective_sample_size = 1.0 / float(np.square(weights).sum())
        if effective_sample_size < 0.4 * len(self._particles):
            selected = self._rng.choice(
                len(self._particles), len(self._particles), replace=True, p=weights
            )
            elite = self._particles[int(np.argmax(weights))].copy()
            self._particles = _jitter_particles(
                self._particles[selected].copy(), self.config, self._rng
            )
            self._particles[0] = elite
            self._log_weights = np.zeros(len(self._particles), dtype=np.float64)
        return result


class ContinualActivationWorldSMC:
    """Persistent particle posterior scored only by imagined Model 1 activations."""

    def __init__(self, config: SMCConfig | None = None) -> None:
        self.config = config or SMCConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._particles: np.ndarray | None = None
        self._log_weights: np.ndarray | None = None
        self.update_count = 0
        self.initialization_count = 0

    def update(self, imagination: ActivationImaginationLoop) -> SMCResult:
        if type(imagination) is not ActivationImaginationLoop:
            raise TypeError("activation search requires an ActivationImaginationLoop")
        if self._particles is None:
            self._particles = _initial_particles(self.config, self._rng)
            self._log_weights = np.zeros(len(self._particles), dtype=np.float64)
            self.initialization_count += 1
        assert self._log_weights is not None
        worlds = tuple(_world_from_particle(particle) for particle in self._particles)
        discrepancies = imagination.discrepancies(worlds)
        totals = discrepancies.sum(axis=1)
        if not np.isfinite(totals).any():
            raise CounterfactualUnavailable(
                "no candidate world produced comparable Model 1 activations"
            )
        self._log_weights += -self.config.score_temperature * totals
        weights = _normalize_log_weights(self._log_weights)
        self.update_count += 1
        flat_worlds = np.stack(
            [world.to_flat(WorldSchema.v1()) for world in worlds]
        ).astype(np.float32)
        order = np.argsort(-weights, kind="stable")
        result = SMCResult(
            ranked_worlds=flat_worlds[order],
            weights=weights[order],
            best_world=flat_worlds[order][0].copy(),
            event_discrepancies=discrepancies[order],
            parameter_particles=self._particles[order].copy(),
            update_count=self.update_count,
        )
        effective_sample_size = 1.0 / float(np.square(weights).sum())
        if effective_sample_size < 0.4 * len(self._particles):
            selected = self._rng.choice(
                len(self._particles), len(self._particles), replace=True, p=weights
            )
            elite = self._particles[int(np.argmax(weights))].copy()
            self._particles = _jitter_particles(
                self._particles[selected].copy(), self.config, self._rng
            )
            self._particles[0] = elite
            self._log_weights = np.zeros(len(self._particles), dtype=np.float64)
        return result


class ActivationWorldSMC:
    """Particle search scored only through counterfactual Model 1 activations."""

    def __init__(self, config: SMCConfig | None = None) -> None:
        self.config = config or SMCConfig()

    def infer(self, imagination: ActivationImaginationLoop) -> SMCResult:
        rng = np.random.default_rng(self.config.seed)
        particles = _initial_particles(self.config, rng)
        worlds = tuple(_world_from_particle(particle) for particle in particles)
        discrepancies = imagination.discrepancies(worlds)
        total_discrepancy = discrepancies.sum(axis=1)
        if not np.isfinite(total_discrepancy).any():
            raise CounterfactualUnavailable(
                "no candidate world produced comparable Model 1 activations"
            )
        weights = _normalize_log_weights(
            -self.config.score_temperature * total_discrepancy
        )
        progress = [
            self._progress(0, particles, total_discrepancy, weights, accepted=0)
        ]
        for round_index in range(1, self.config.rounds):
            proposals = _jitter_particles(particles, self.config, rng)
            proposal_worlds = tuple(
                _world_from_particle(particle) for particle in proposals
            )
            proposal_discrepancies = imagination.discrepancies(proposal_worlds)
            proposal_total = proposal_discrepancies.sum(axis=1)
            current_finite = np.isfinite(total_discrepancy)
            proposal_finite = np.isfinite(proposal_total)
            log_acceptance = np.full(len(particles), -np.inf, dtype=np.float64)
            newly_valid = proposal_finite & ~current_finite
            both_valid = proposal_finite & current_finite
            log_acceptance[newly_valid] = np.inf
            log_acceptance[both_valid] = -self.config.score_temperature * (
                proposal_total[both_valid] - total_discrepancy[both_valid]
            )
            accepted = proposal_finite & (
                np.log(rng.random(len(particles)))
                < np.minimum(0.0, log_acceptance)
            )
            elite = int(np.nanargmin(total_discrepancy))
            accepted[elite] = proposal_total[elite] < total_discrepancy[elite]
            particles[accepted] = proposals[accepted]
            discrepancies[accepted] = proposal_discrepancies[accepted]
            total_discrepancy[accepted] = proposal_total[accepted]
            weights = _normalize_log_weights(
                -self.config.score_temperature * total_discrepancy
            )
            progress.append(
                self._progress(
                    round_index,
                    particles,
                    total_discrepancy,
                    weights,
                    accepted=int(accepted.sum()),
                )
            )
        worlds = np.stack(
            [
                _world_from_particle(particle).to_flat(WorldSchema.v1())
                for particle in particles
            ]
        ).astype(np.float32)
        order = np.argsort(-weights, kind="stable")
        ranked_worlds = worlds[order]
        return SMCResult(
            ranked_worlds=ranked_worlds,
            weights=weights[order],
            best_world=ranked_worlds[0].copy(),
            event_discrepancies=discrepancies[order],
            progress=tuple(progress),
        )

    @staticmethod
    def _progress(
        round_index: int,
        particles: np.ndarray,
        discrepancies: np.ndarray,
        weights: np.ndarray,
        *,
        accepted: int,
    ) -> ActivationSearchProgress:
        finite = discrepancies[np.isfinite(discrepancies)]
        best_index = int(np.nanargmin(discrepancies))
        return ActivationSearchProgress(
            round_index=round_index,
            best_discrepancy=float(discrepancies[best_index]),
            mean_discrepancy=float(finite.mean()),
            effective_sample_size=float(1.0 / np.square(weights).sum()),
            accepted_revisions=accepted,
            best_parameters=particles[best_index].copy(),
            posterior_mean=np.average(particles, axis=0, weights=weights),
        )
