from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from empathy.model1.config import load_model1_config
from empathy.model1.model import Model1
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

