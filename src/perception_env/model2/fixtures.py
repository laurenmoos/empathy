"""Offline oracle bootstrap; privileged state never enters the input archive."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from perception_env.beliefs import BeliefEvent, build_belief_history, superpose
from perception_env.config import RenderConfig, SimulationConfig
from perception_env.events import extract_reappearance
from perception_env.families import load_family, sample_worlds
from perception_env.reachability import (
    EntryStateGrid,
    Probe,
    enumerate_reachable,
    rasterize_endpoints,
)
from perception_env.render import Occluder, render_sequence
from perception_env.simulate import BallState, rollout
from perception_env.world import WorldCode, WorldSchema

from .config import FixtureConfig
from .contracts import BeliefInputs, SchemaLayout, WorldTargets
from .data import array_hash, save_dataset


@dataclass(frozen=True)
class FixtureManifest:
    root: Path
    dataset_hash: str
    accepted_worlds: int
    rejection_counts: dict[str, int]


def _world_id(world: WorldCode) -> str:
    code = np.round(world.to_flat(WorldSchema.v1()), 10)
    return hashlib.sha256(code.tobytes()).hexdigest()


def split_world_groups(
    world_ids: Sequence[str], fractions: tuple[float, float, float], seed: int
) -> np.ndarray:
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("invalid split fractions")
    unique = sorted(set(world_ids))
    if len(unique) < 3:
        raise ValueError("at least three unique worlds needed for disjoint splits")
    order = np.random.default_rng(seed).permutation(len(unique))
    train_count = max(1, min(len(unique) - 2, int(len(unique) * fractions[0])))
    val_count = max(1, min(len(unique) - train_count - 1, int(len(unique) * fractions[1])))
    assignment = {}
    for index, position in enumerate(order):
        assignment[unique[position]] = (
            0 if index < train_count else (1 if index < train_count + val_count else 2)
        )
    return np.array([assignment[world] for world in world_ids], dtype=np.int64)


def _grid(state: BallState, config: FixtureConfig) -> EntryStateGrid:
    # Preserve the exact state as the midpoint; shrink position uncertainty at edges.
    dx, dy = min(0.005, state.x, 1 - state.x), min(0.005, state.y, 1 - state.y)
    return EntryStateGrid(
        (state.x - dx, state.x + dx),
        (state.y - dy, state.y + dy),
        (state.vx - 0.02, state.vx + 0.02),
        (state.vy - 0.02, state.vy + 0.02),
        config.entry_resolution,
    )


def build_fixture_event(
    true_world: WorldCode,
    prior_world: WorldCode,
    initial: BallState,
    config: FixtureConfig,
    ordinal: int,
    seed: int,
) -> tuple[BeliefEvent, dict[str, object]]:
    sim = SimulationConfig(dt=config.dt, steps=config.steps, seed=seed)
    render = RenderConfig(
        width=config.size,
        height=config.size,
        ball_radius=1 if config.size < 32 else 2,
        min_hidden_steps=2,
    )
    trajectory = rollout(initial, true_world, sim)
    if not trajectory.valid:
        raise ValueError("out_of_bounds")
    sequence = render_sequence(trajectory, (Occluder(*config.occluder),), render)
    observed = extract_reappearance(sequence)
    if observed is None:
        raise ValueError(sequence.rejection_reason or "no_reappearance")
    if observed.hidden_start == 0 or not sequence.visible[: observed.hidden_start].any():
        raise ValueError("missing_visible_context")
    entry = trajectory.states[observed.hidden_start]
    grid = _grid(entry, config)
    relative_step = observed.step - observed.hidden_start
    query_config = SimulationConfig(dt=config.dt, steps=relative_step, seed=seed)
    context = f"event-{seed}-{ordinal}"
    truth = enumerate_reachable(
        grid, true_world, (relative_step,), query_config, context_id=context
    )
    prior = enumerate_reachable(
        grid, prior_world, (relative_step,), query_config, context_id=context
    )
    pre_paths = [path for path in prior.trajectories if path.valid]
    post_paths = [
        path
        for path in truth.trajectories
        if path.valid
        and np.hypot(path.states[-1].x - observed.x, path.states[-1].y - observed.y)
        <= config.tolerance
    ]
    if not post_paths:
        raise ValueError("empty_oracle_posterior")
    pre = superpose(pre_paths, np.ones(len(pre_paths)), render)
    post = superpose(post_paths, np.ones(len(post_paths)), render)
    provenance = {
        "ordinal": ordinal,
        "context_id": context,
        "seed": seed,
        "true_world": _world_id(true_world),
        "prior_world": _world_id(prior_world),
        "prior_world_flat": prior_world.to_flat(WorldSchema.v1()).tolist(),
        "entry_state": [entry.x, entry.y, entry.vx, entry.vy],
        "entry_grid": grid.to_metadata(),
        "query_step": relative_step,
        "simulation_config": asdict(query_config),
        "hidden_start": observed.hidden_start,
        "reappearance_step": observed.step,
        "reappearance_visible": bool(sequence.visible[observed.step]),
        "reappearance": [observed.x, observed.y],
        "source": "oracle" if true_world == prior_world else "perturbed_prior_oracle_post",
    }
    return BeliefEvent(ordinal, pre, post), provenance


def _reachability(world: WorldCode, event: dict):
    raw = event["entry_grid"]
    grid = EntryStateGrid(
        *(
            tuple(raw[key])
            for key in ("x_bounds", "y_bounds", "vx_bounds", "vy_bounds", "resolution")
        )
    )
    return enumerate_reachable(
        grid,
        world,
        (event["query_step"],),
        SimulationConfig(**event["simulation_config"]),
        context_id=event["context_id"],
    )


def _probes(result, count: int, tolerance: float, seed: int) -> tuple[Probe, ...]:
    rng = np.random.default_rng(seed)
    step = result.query_steps[0]
    endpoints = result.endpoints[step]
    points = rng.uniform(0, 1, (count, 2))
    if endpoints:
        for index in range(count // 2):
            endpoint = endpoints[int(rng.integers(len(endpoints)))]
            angle, radius = rng.uniform(0, 2 * np.pi), 0.05 * np.sqrt(rng.uniform())
            points[index] = np.clip(
                [endpoint.x + radius * np.cos(angle), endpoint.y + radius * np.sin(angle)], 0, 1
            )
    return tuple(Probe(result.context_id, float(x), float(y), step, tolerance) for x, y in points)


def _perturb(world: WorldCode, rng: np.random.Generator) -> WorldCode:
    schema = WorldSchema.v1()
    active = {
        slot.name: slot.parameter_dict(spec)
        for slot, spec in zip(world.slots, schema.specs, strict=True)
        if slot.active
    }
    # A single-slot toggle is a transparent, guaranteed distinct valid perturbation.
    slot = schema.specs[int(rng.integers(len(schema.specs)))]
    if slot.name in active:
        del active[slot.name]
    else:
        values = {}
        for parameter in slot.parameters:
            for _ in range(1000):
                value = float(rng.uniform(parameter.minimum, parameter.maximum))
                try:
                    parameter.validate(value, active=True)
                except ValueError:
                    continue
                values[parameter.name] = value
                break
            else:
                raise ValueError("unable to sample active perturbation")
        active[slot.name] = values
    return schema.world(**active)


def generate_fixtures(config: FixtureConfig, output: Path) -> FixtureManifest:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    schema = WorldSchema.v1()
    families = [load_family(Path(config.family_path), level) for level in config.levels]
    rng_world, rng_context, rng_prior = [
        np.random.default_rng(seed) for seed in np.random.SeedSequence(config.seed).spawn(3)
    ]
    accepted, seen, rows, histories, codes = [], set(), [], [], []
    audits, heldouts, audit_labels, heldout_labels, rasters, masks = [], [], [], [], [], []
    rejections = Counter()
    attempts = 0
    for attempts in range(1, config.world_count * config.world_attempt_multiplier + 1):
        family = families[(attempts - 1) % len(families)]
        world = sample_worlds(family, 1, int(rng_world.integers(2**31)))[0]
        identity = _world_id(world)
        if identity in seen:
            rejections["duplicate_world"] += 1
            continue
        episodes = []
        try:
            for _episode in range(config.episodes_per_world):
                events, provenance = [], []
                for ordinal in range(4):
                    for _ in range(config.max_context_attempts):
                        initial = BallState(
                            *(
                                float(rng_context.uniform(*bounds))
                                for bounds in (
                                    config.x_bounds,
                                    config.y_bounds,
                                    config.vx_bounds,
                                    config.vy_bounds,
                                )
                            )
                        )
                        prior = (
                            world
                            if rng_prior.random() < config.oracle_fraction
                            else (_perturb(world, rng_prior))
                        )
                        try:
                            event, meta = build_fixture_event(
                                world,
                                prior,
                                initial,
                                config,
                                ordinal,
                                int(rng_context.integers(2**31)),
                            )
                        except ValueError as error:
                            rejections[str(error)] += 1
                            continue
                        events.append(event)
                        provenance.append(meta)
                        break
                    else:
                        raise ValueError("context_attempt_limit")
                episodes.append((build_belief_history(events), provenance))
        except ValueError as error:
            rejections[str(error)] += 1
            continue
        seen.add(identity)
        accepted.append(world)
        bank = sample_worlds(family, config.audit_bank_size, config.seed + attempts + 10000)
        for episode, (history, provenance) in enumerate(episodes):
            row_audits, row_heldouts, row_alabels, row_hlabels, row_rasters = [], [], [], [], []
            candidate_matches = np.ones(len(bank), bool)
            for meta in provenance:
                truth = _reachability(world, meta)
                audit = _probes(
                    truth, config.audit_probe_count, config.tolerance, meta["seed"] + 20000
                )
                heldout = _probes(
                    truth, config.heldout_probe_count, config.tolerance, meta["seed"] + 30000
                )
                true_labels = np.array([truth.label(probe) for probe in audit])
                for index, candidate in enumerate(bank):
                    result = _reachability(candidate, meta)
                    candidate_matches[index] &= np.array_equal(
                        true_labels, [result.label(probe) for probe in audit]
                    )
                row_audits.append([asdict(probe) for probe in audit])
                row_heldouts.append([asdict(probe) for probe in heldout])
                row_alabels.append(true_labels)
                row_hlabels.append([truth.label(probe) for probe in heldout])
                row_rasters.append(
                    rasterize_endpoints(truth, RenderConfig(width=config.size, height=config.size))[
                        0
                    ]
                )
            identifiable = not any(
                match and _world_id(candidate) != identity
                for match, candidate in zip(candidate_matches, bank, strict=True)
            )
            layout = SchemaLayout.from_schema(schema)
            flat = world.to_flat(schema)
            active = flat[list(layout.activity_indices)].astype(bool)
            masks.append(active[list(layout.parameter_to_slot)] & identifiable)
            histories.append(history)
            codes.append(flat)
            rows.append(
                {
                    "world_id": identity,
                    "episode_id": f"{identity}-{episode}",
                    "family_level": family.level,
                    "events": provenance,
                    "audit_status": "identifiable" if identifiable else "perceptually_equivalent",
                    "comparison_bank": [item.to_flat(schema).tolist() for item in bank],
                }
            )
            audits.append(row_audits)
            heldouts.append(row_heldouts)
            audit_labels.append(row_alabels)
            heldout_labels.append(row_hlabels)
            rasters.append(row_rasters)
        if len(accepted) == config.world_count:
            break
    if len(accepted) != config.world_count:
        raise ValueError(
            f"incomplete fixture generation: {len(accepted)}/{config.world_count}; "
            f"rejections={dict(rejections)}"
        )
    world_ids = [row["world_id"] for row in rows]
    split = split_world_groups(world_ids, config.split_fractions, config.seed + 40000)
    evaluation_arrays = {
        "audit_labels": np.asarray(audit_labels, bool),
        "heldout_labels": np.asarray(heldout_labels, bool),
        "truth_rasters": np.asarray(rasters, bool),
    }
    evaluation_metadata = {
        "audit_probes": audits,
        "heldout_probes": heldouts,
        "contexts": [row["events"] for row in rows],
        "heldout_scope": "new_probes_same_contexts",
        "tolerance": config.tolerance,
    }
    eval_json = json.dumps(evaluation_metadata, sort_keys=True, separators=(",", ":"))
    input_rows = [
        {
            **row,
            "events": [
                {
                    name: event[name]
                    for name in ("ordinal", "true_world", "prior_world", "source", "seed")
                }
                for event in row["events"]
            ],
        }
        for row in rows
    ]
    manifest = save_dataset(
        output,
        BeliefInputs(
            np.stack([h.tensor for h in histories]), np.stack([h.mask for h in histories])
        ),
        WorldTargets(np.stack(codes), np.stack(masks)),
        split,
        {
            "world_ids": world_ids,
            "rows": input_rows,
            "fixture_config": asdict(config),
            "family_config": Path(config.family_path).read_text(),
            "rejection_counts": dict(rejections),
            "world_attempts": attempts,
            "source": "privileged_oracle_bootstrap",
            "evaluation_hash": array_hash(evaluation_arrays),
            "evaluation_metadata_hash": hashlib.sha256(eval_json.encode()).hexdigest(),
        },
    )
    np.savez_compressed(output / "evaluation.npz", **evaluation_arrays)
    (output / "evaluation.json").write_text(eval_json)
    return FixtureManifest(output, manifest["dataset_hash"], len(accepted), dict(rejections))
