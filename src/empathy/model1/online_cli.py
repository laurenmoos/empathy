"""Headless online Seeing training over repeated throws in one world."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from empathy.physics.config import RenderConfig, SimulationConfig
from empathy.physics.events import extract_reappearance
from empathy.physics.render import Occluder, render_sequence
from empathy.physics.simulate import BallState, rollout

from ..live.inference import LiveParameters, world_from_parameters
from .online import OnlineSeeingConfig, OnlineSeeingModel
from .online_train import (
    ObservedSeeingEpisode,
    OnlineSeeingRuntime,
    observed_states_from_path,
)


def _clip(frames: np.ndarray, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.from_numpy(frames[start:end].astype(np.float32) / 255.0)
    values = values[None, :, None]
    return values, torch.ones((1, values.shape[1]), dtype=torch.bool)


def _observed_episode(index: int, rng: np.random.Generator) -> ObservedSeeingEpisode:
    world = world_from_parameters(LiveParameters(gravity=-0.06, wind=0.06, damping=0.04))
    initial = BallState(
        0.1,
        float(rng.uniform(0.58, 0.72)),
        float(rng.uniform(0.28, 0.32)),
        float(rng.uniform(-0.01, 0.02)),
    )
    simulation = SimulationConfig(dt=0.1, steps=48, seed=index)
    trajectory = rollout(initial, world, simulation)
    sequence = render_sequence(
        trajectory,
        (Occluder(0.35, 0.55, 0.05, 0.95),),
        RenderConfig(width=48, height=48, ball_radius=2, min_hidden_steps=2),
    )
    event = extract_reappearance(sequence)
    if event is None:
        raise RuntimeError("generated online throw has no visible reappearance")
    pre_start = max(0, event.hidden_start - 12)
    post_end = min(len(sequence.frames), event.step + 8)
    pre_frames, pre_mask = _clip(sequence.frames, pre_start, event.hidden_start)
    post_frames, post_mask = _clip(sequence.frames, event.step, post_end)
    path = torch.tensor([(state.x, state.y) for state in trajectory.states])
    entry_state, exit_state, departure = observed_states_from_path(
        path,
        hidden_start=event.hidden_start,
        reappearance_step=event.step,
        dt=simulation.dt,
        left_edge=0.35,
    )
    return ObservedSeeingEpisode(
        pre_frames=pre_frames,
        pre_mask=pre_mask,
        post_frames=post_frames,
        post_mask=post_mask,
        entry_state=entry_state,
        exit_state=exit_state,
        departure_positions=departure,
        dt=simulation.dt,
    )


def run_online_training(
    *, episodes: int, log_dir: Path, seed: int = 7
) -> list[dict[str, int | float]]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    config = OnlineSeeingConfig()
    runtime = OnlineSeeingRuntime(
        OnlineSeeingModel(config), seed=seed, occluder=(0.35, 0.55)
    )
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=1)
    rows: list[dict[str, int | float]] = []
    metrics_path = log_dir / "metrics.jsonl"
    with metrics_path.open("w") as metrics_file:
        for episode_index in range(episodes):
            episode = _observed_episode(episode_index, rng)
            result = runtime.complete(runtime.begin(episode), episode, train=True)
            metrics = result.metrics
            row: dict[str, int | float] = {
                "episode": episode_index + 1,
                "loss": round(metrics.loss, 6),
                "geometric_loss": round(metrics.geometric_loss, 6),
                "contrastive_loss": round(metrics.contrastive_loss, 6),
                "exit_error": round(metrics.exit_error, 6),
                "belief_spread": round(metrics.belief_spread, 6),
                "effective_samples": round(metrics.effective_samples, 3),
                "updates": metrics.updates,
            }
            rows.append(row)
            encoded = json.dumps(row, sort_keys=True)
            print(encoded, flush=True)
            metrics_file.write(encoded + "\n")
            metrics_file.flush()
            for name, value in row.items():
                if name != "episode":
                    writer.add_scalar(f"seeing/{name}", value, episode_index + 1)
            writer.flush()
    writer.close()
    torch.save(
        {
            "model": runtime.model.state_dict(),
            "world_latent": runtime.world_latent.detach(),
            "config": config.__dict__,
            "updates": runtime.updates,
        },
        log_dir / "online-seeing.pt",
    )
    return rows
