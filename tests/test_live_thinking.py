from pathlib import Path

import numpy as np
import pytest
import torch

from perception_env.live_inference import LiveParameters
from perception_env.live_thinking import LiveThinkingSession, OnlineModel2Runtime
from perception_env.model1.config import load_model1_config
from perception_env.model1.types import TrajectoryMixture
from perception_env.model2.contracts import BeliefInputs, WorldPrediction
from perception_env.model2.network import InverseWorldCNN
from perception_env.simulate import BallState
from perception_env.world import WorldSchema

ROOT = Path(__file__).parents[1]


class FourPathModel1:
    def __call__(self, frames, times, mask) -> TrajectoryMixture:
        del frames, times, mask
        steps = 12
        x = torch.linspace(0.34, 0.68, steps)
        paths = [
            torch.stack((x, torch.full_like(x, 0.43 + 0.04 * index)), dim=-1)
            for index in range(4)
        ]
        return TrajectoryMixture(
            positions=torch.stack(paths)[None],
            logits=torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
            entry_state=torch.tensor([[0.34, 0.5, 0.18, 0.0]]),
        )


class RecordingWorldRuntime:
    def __init__(self) -> None:
        self.inputs: list[BeliefInputs] = []
        self.order: list[str] = []
        self.weight_update_count = 0

    def predict(self, inputs: BeliefInputs) -> WorldPrediction:
        self.order.append("predict")
        self.inputs.append(inputs)
        call = len(self.inputs)
        logits = torch.tensor([[float(call), -10.0, -10.0, -10.0, -10.0]])
        return WorldPrediction(logits, torch.full((1, 7), 0.5))

    def train_step(self, inputs: BeliefInputs, target_world_flat: np.ndarray) -> dict[str, float]:
        self.order.append("train")
        assert inputs is self.inputs[-1]
        assert target_world_flat.shape == (12,)
        self.weight_update_count += 1
        return {"loss": 0.25, "weight_update_count": self.weight_update_count}


def _session(runtime=None, **kwargs) -> LiveThinkingSession:
    return LiveThinkingSession(
        FourPathModel1(),
        load_model1_config(ROOT / "configs" / "model1" / "v1.yaml"),
        runtime or RecordingWorldRuntime(),
        LiveParameters(gravity=-0.06, wind=0.0, damping=0.0),
        **kwargs,
    )


def test_episode_thinks_in_four_belief_only_steps_before_training() -> None:
    runtime = RecordingWorldRuntime()
    session = _session(runtime)

    episode = session.next_episode()

    assert episode.initial_state != BallState(0.1, 0.5, 0.18, 0.0)
    assert [step.label for step in episode.thought_steps] == [
        "first possibility",
        "more possibilities",
        "full belief",
        "reappearance revision",
    ]
    assert runtime.order == ["predict", "predict", "predict", "predict", "train"]
    assert all(isinstance(inputs, BeliefInputs) for inputs in runtime.inputs)
    assert all(set(inputs.__dict__) == {"beliefs", "history_mask"} for inputs in runtime.inputs)
    assert episode.thought_steps[-1].guessed_flat[0] == 1
    assert episode.training == {"loss": 0.25, "weight_update_count": 1}
    assert episode.latent_update_count == 4
    assert episode.weight_update_count == 1
    payload = episode.to_payload()
    path = payload["ground_truth"]["path"]
    assert len(path) > 1
    assert all(len(point) == 2 for point in path)
    assert all(0.0 <= coordinate <= 1.0 for point in path for coordinate in point)
    assert payload["thought_steps"][0]["predicted_parameters"] == [0.5] * 7


def test_successive_episodes_vary_offsets_and_keep_completed_event_history() -> None:
    runtime = RecordingWorldRuntime()
    session = _session(runtime)

    first = session.next_episode()
    second = session.next_episode()
    third = session.next_episode()

    assert len({first.initial_state, second.initial_state, third.initial_state}) == 3
    assert first.episode_index == 0
    assert second.episode_index == 1
    assert third.episode_index == 2
    assert runtime.inputs[3].history_mask.tolist() == [[False, False, False, True]]
    assert runtime.inputs[7].history_mask.tolist() == [[False, False, True, True]]
    assert runtime.inputs[11].history_mask.tolist() == [[False, True, True, True]]
    assert session.completed_event_count == 3


def test_invalid_offsets_are_skipped_with_a_fixed_attempt_bound() -> None:
    candidates = (
        BallState(0.1, 1.2, 0.18, 0.0),
        BallState(0.1, 0.5, 0.18, 0.0),
    )
    session = _session(initial_states=candidates, max_offset_attempts=2)

    episode = session.next_episode()

    assert episode.initial_state == candidates[1]

    failing = _session(initial_states=candidates[:1], max_offset_attempts=1)
    with pytest.raises(ValueError, match="valid offset"):
        failing.next_episode()


def test_online_runtime_performs_one_small_weight_update_without_decay() -> None:
    torch.manual_seed(3)
    model = InverseWorldCNN()
    runtime = OnlineModel2Runtime(model)
    beliefs = np.zeros((1, 4, 3, 16, 16), dtype=np.float32)
    beliefs[0, 3, 1, 8, 8] = 1.0
    beliefs[0, 3, 2, 8, 8] = 1.0
    inputs = BeliefInputs(beliefs, np.array([[False, False, False, True]]))
    target = WorldSchema.v1().world().to_flat(WorldSchema.v1()).astype(np.float32)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    metrics = runtime.train_step(inputs, target)

    assert runtime.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-4)
    assert runtime.optimizer.param_groups[0]["weight_decay"] == 0
    assert metrics["weight_update_count"] == 1
    assert metrics["loss"] > 0
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
