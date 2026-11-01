from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from empathy.model1.config import load_model1_config
from empathy.model1.model import Model1
from empathy.model1.online import OnlineSeeingConfig, StepwiseWorldModel
from empathy.model1.online_train import ObservedSeeingEpisode
from empathy.model2.imagination import CounterfactualUnavailable
from empathy.physics.simulate import BallState
from empathy.physics.world import WorldSchema


def test_frozen_model1_runner_returns_activations_without_perceptions() -> None:
    assert importlib.util.find_spec("empathy.model1.counterfactual") is not None
    counterfactual = importlib.import_module("empathy.model1.counterfactual")
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    model = Model1(config.model, prediction_steps=config.dataset.prediction_steps)
    episode = counterfactual.CounterfactualEpisode(
        "episode-0",
        0,
        BallState(0.1, 0.60, 0.18, 0.0),
    )
    runner = counterfactual.FrozenModel1CounterfactualRunner(
        model,
        config,
        (episode,),
        checkpoint_id="model1-test",
    )

    packets = runner.activations_for(WorldSchema.v1().world())

    assert len(packets) == 1
    assert packets[0].checkpoint_id == "model1-test"
    assert packets[0].pre_activation.shape == (config.model.hidden_size,)
    assert packets[0].post_activation.shape == (config.model.hidden_size,)
    assert not packets[0].pre_activation.flags.writeable
    assert not hasattr(packets[0], "frames")
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_frozen_online_runner_imagines_with_the_same_model1_checkpoint() -> None:
    counterfactual = importlib.import_module("empathy.model1.counterfactual")
    model_config = OnlineSeeingConfig(
        channels=(4, 8), representation_size=12, world_size=6, sample_count=8
    )
    model = StepwiseWorldModel(model_config, hidden_size=10)
    frame_config = load_model1_config(Path("configs/model1/v1.yaml"))
    frames = torch.zeros((1, 4, 1, 48, 48), dtype=torch.float32)
    mask = torch.ones((1, 4), dtype=torch.bool)
    observed_episode = ObservedSeeingEpisode(
        pre_frames=frames,
        pre_mask=mask,
        post_frames=frames.clone(),
        post_mask=mask.clone(),
        entry_state=torch.tensor([[0.35, 0.60, 0.18, 0.0]]),
        exit_state=torch.tensor([[0.55, 0.58, 0.18, -0.02]]),
        departure_positions=torch.tensor([[[0.57, 0.57], [0.59, 0.56]]]),
    )
    episode = counterfactual.CounterfactualEpisode(
        "episode-0", 0, BallState(0.1, 0.60, 0.18, 0.0)
    )

    runner, observed = counterfactual.FrozenOnlineSeeingCounterfactualRunner.from_observed(
        model,
        torch.zeros((1, model_config.world_size)),
        frame_config,
        ((episode, observed_episode),),
        checkpoint_id="online-1",
    )
    imagined = runner.activations_for(WorldSchema.v1().world())

    assert len(observed) == len(imagined) == 1
    assert observed[0].checkpoint_id == imagined[0].checkpoint_id == "online-1"
    assert observed[0].pre_activation.shape == imagined[0].pre_activation.shape == (20,)
    assert observed[0].episode_id == imagined[0].episode_id == "episode-0"
    assert not hasattr(observed[0], "frames")
    assert not hasattr(runner, "observed_episodes")
    assert all(not parameter.requires_grad for parameter in runner.model.parameters())
    assert np.isfinite(imagined[0].activation_difference).all()


def test_online_runner_marks_noncomparable_candidate_events_unavailable(
    monkeypatch,
) -> None:
    counterfactual = importlib.import_module("empathy.model1.counterfactual")
    model_config = OnlineSeeingConfig(
        channels=(4, 8), representation_size=12, world_size=6, sample_count=8
    )
    runner = counterfactual.FrozenOnlineSeeingCounterfactualRunner(
        StepwiseWorldModel(model_config, hidden_size=10),
        torch.zeros((1, model_config.world_size)),
        load_model1_config(Path("configs/model1/v1.yaml")),
        (
            counterfactual.CounterfactualEpisode(
                "episode-0", 0, BallState(0.1, 0.60, 0.18, 0.0)
            ),
        ),
        checkpoint_id="online-1",
    )

    def reject_path(*args, **kwargs):
        del args, kwargs
        raise ValueError("path lacks visible points around the occlusion")

    monkeypatch.setattr(counterfactual, "observed_states_from_path", reject_path)

    with pytest.raises(CounterfactualUnavailable, match="not comparable"):
        runner.activations_for(WorldSchema.v1().world())

