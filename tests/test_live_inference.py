from pathlib import Path

import numpy as np
import pytest
import torch

from perception_env.live_inference import (
    LiveInferenceEngine,
    LiveInferenceError,
    LiveParameters,
    world_from_parameters,
)
from perception_env.model1.config import load_model1_config
from perception_env.model1.types import TrajectoryMixture
from perception_env.model2.contracts import BeliefInputs, WorldPrediction
from perception_env.world import WorldSchema

ROOT = Path(__file__).parents[1]


class FakeModel1:
    def __call__(self, frames, times, mask) -> TrajectoryMixture:
        steps = 12
        x = torch.linspace(0.35, 0.7, steps)
        y = torch.full((steps,), 0.5)
        path = torch.stack((x, y), dim=-1)
        positions = torch.stack((path, path + torch.tensor([0.0, 0.03])))[None]
        return TrajectoryMixture(
            positions=positions,
            logits=torch.zeros((1, 2)),
            entry_state=torch.full((1, 4), 0.5),
        )


def test_world_from_parameters_canonicalizes_neutral_values() -> None:
    schema = WorldSchema.v1()

    world = world_from_parameters(LiveParameters(gravity=-0.4, wind=0.02, damping=0.0))

    assert world.slot("vertical_acceleration").active
    assert world.slot("vertical_acceleration").parameters == (-0.4,)
    assert not world.slot("horizontal_acceleration").active
    assert not world.slot("damping").active
    assert np.allclose(world.to_flat(schema)[:6], [1, 0.3, 0, 0, 0, 0])


def test_live_inference_passes_only_beliefs_to_model2_and_returns_both_worlds() -> None:
    config = load_model1_config(ROOT / "configs" / "model1" / "v1.yaml")
    captured: list[BeliefInputs] = []

    def model2_predict(inputs: BeliefInputs) -> WorldPrediction:
        captured.append(inputs)
        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]])
        return WorldPrediction(logits, torch.full((1, 7), 0.5))

    engine = LiveInferenceEngine(FakeModel1(), config, model2_predict)

    result = engine.infer(LiveParameters(gravity=-0.1, wind=0.0, damping=0.0))

    assert len(captured) == 1
    assert captured[0].beliefs.shape == (1, 4, 3, 64, 64)
    assert captured[0].history_mask.tolist() == [[True, True, True, True]]
    assert result.ground_truth_flat.shape == (12,)
    assert result.guessed_flat.shape == (12,)
    assert result.ground_truth_flat[0] == 1
    assert result.frames.shape[0] > result.reappearance_step
    assert result.pre_belief.shape == result.post_belief.shape == (64, 64)
    payload = result.to_payload()
    actual_gravity = payload["ground_truth"]["operators"]["vertical_acceleration"][
        "parameters"
    ]["a_y"]
    assert actual_gravity == pytest.approx(-0.1)
    assert "frames" not in captured[0].__dict__


def test_live_inference_rejects_world_without_valid_reappearance() -> None:
    config = load_model1_config(ROOT / "configs" / "model1" / "v1.yaml")
    engine = LiveInferenceEngine(
        FakeModel1(),
        config,
        lambda _: WorldPrediction(torch.zeros((1, 5)), torch.zeros((1, 7))),
    )

    with pytest.raises(LiveInferenceError, match="reappearance|bounds"):
        engine.infer(LiveParameters(gravity=0.0, wind=-1.0, damping=0.0))
