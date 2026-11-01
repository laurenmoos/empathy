from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from empathy.model1.online import OnlineSeeingConfig, StepwiseWorldModel
from empathy.model1.online_train import (
    ObservedSeeingEpisode,
    OnlineSeeingRuntime,
    observed_states_from_path,
)


def _episode(y: float = 0.55) -> ObservedSeeingEpisode:
    generator = torch.Generator().manual_seed(int(y * 1000))
    pre = torch.rand(1, 5, 1, 20, 20, generator=generator)
    post = pre.clone()
    mask = torch.ones(1, 5, dtype=torch.bool)
    entry = torch.tensor([[0.38, y, 0.25, -0.01]])
    exit_state = torch.tensor([[0.58, y - 0.03, 0.25, -0.03]])
    departure = torch.tensor([[[0.60, y - 0.033], [0.63, y - 0.037]]])
    return ObservedSeeingEpisode(pre, mask, post, mask, entry, exit_state, departure)


def _runtime(seed: int = 3) -> OnlineSeeingRuntime:
    config = OnlineSeeingConfig(
        channels=(4, 8), representation_size=12, world_size=5, sample_count=32
    )
    return OnlineSeeingRuntime(StepwiseWorldModel(config, hidden_size=16), seed=seed, fast_lr=0.05)


def test_observed_episode_has_no_hidden_trajectory_field() -> None:
    assert "hidden_path" not in {field.name for field in fields(ObservedSeeingEpisode)}


def test_exit_evidence_reweights_closer_paths_without_mutating_prior() -> None:
    runtime = _runtime()
    episode = _episode()
    frozen = runtime.begin(episode)
    mean_before = frozen.belief.mean.clone()
    result = runtime.complete(frozen, episode, train=False)
    endpoint_error = (result.paths[:, :, -1] - episode.exit_state[:, None, :2]).square().sum(-1)
    best = endpoint_error.argmin(dim=1)
    worst = endpoint_error.argmax(dim=1)
    assert result.posterior_weights[0, best] > result.posterior_weights[0, worst]
    assert torch.equal(frozen.belief.mean, mean_before)


def test_repeated_observations_update_world_state_and_preserve_replay() -> None:
    runtime = _runtime()
    episode = _episode()
    initial = runtime.world_latent.detach().clone()
    errors = []
    for _ in range(8):
        result = runtime.complete(runtime.begin(episode), episode, train=True)
        errors.append(result.metrics.prediction_loss)
    assert runtime.updates == 8
    assert len(runtime.replay) == 8
    assert not torch.equal(runtime.world_latent.detach(), initial)
    assert min(errors[4:]) <= errors[0]


def test_only_one_loss_is_reported_and_it_scores_the_frozen_prediction() -> None:
    runtime = _runtime(8)
    episode = _episode(0.48)
    frozen = runtime.begin(episode)
    expected = runtime._evidence_loss(runtime._sample_error(frozen.paths, episode))
    result = runtime.complete(frozen, episode, train=True)
    assert result.metrics.prediction_loss == pytest.approx(float(expected))
    assert set(result.metrics.__dict__) == {"prediction_loss", "belief_spread", "updates"}


def test_observed_state_extraction_ignores_every_hidden_point() -> None:
    path = torch.tensor(
        [[0.10, 0.50], [0.20, 0.51], [0.30, 0.52], [0.40, 0.20], [0.50, 0.80],
         [0.60, 0.54], [0.70, 0.55], [0.80, 0.56]]
    )
    changed = path.clone()
    changed[3:5] = torch.tensor([[0.40, -50.0], [0.50, 50.0]])
    first = observed_states_from_path(path, hidden_start=3, reappearance_step=5, dt=0.1)
    second = observed_states_from_path(changed, hidden_start=3, reappearance_step=5, dt=0.1)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_replayed_geometric_constraints_change_the_next_update() -> None:
    first = _runtime(12)
    second = _runtime(12)
    second.model.load_state_dict(first.model.state_dict())
    current = _episode(0.55)
    old = _episode(0.48)
    shifted = replace(
        old,
        exit_state=old.exit_state + torch.tensor([[0.0, 0.3, 0.0, 0.0]]),
        departure_positions=old.departure_positions + torch.tensor([[[0.0, 0.3]]]),
    )
    first.replay.append(old)
    second.replay.append(shifted)
    first.complete(first.begin(current), current, train=True)
    second.complete(second.begin(current), current, train=True)
    assert not torch.allclose(first.world_latent, second.world_latent)
