from pathlib import Path

import numpy as np
import pytest
import torch

from empathy.live.inference import LiveParameters
from empathy.live.observer import Model2CheckpointGuess
from empathy.live.thinking import (
    DEFAULT_INITIAL_STATES,
    LiveThinkingSession,
    OnlineModel2Runtime,
)
from empathy.model1.config import load_model1_config
from empathy.model1.types import TrajectoryMixture
from empathy.model2.contracts import BeliefInputs, WorldPrediction
from empathy.model2.network import InverseWorldCNN
from empathy.physics.simulate import BallState
from empathy.physics.world import WorldSchema

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


class RecordingObserver:
    def __init__(self) -> None:
        self.checkpoints = []
        self.latest = None
        self.closed = False

    def submit(self, checkpoint) -> None:
        self.checkpoints.append(checkpoint)

    def close(self) -> None:
        self.closed = True

    def complete_event(self, inputs: BeliefInputs) -> dict[str, float]:
        self.order.append("complete")
        assert inputs is self.inputs[-1]
        return {"loss": 0.25, "weight_update_count": self.weight_update_count}


def _session(runtime=None, observer=None, **kwargs) -> LiveThinkingSession:
    return LiveThinkingSession(
        FourPathModel1(),
        load_model1_config(ROOT / "configs" / "model1" / "v1.yaml"),
        runtime or RecordingWorldRuntime(),
        LiveParameters(gravity=-0.06, wind=0.0, damping=0.0),
        model2_observer=observer,
        **kwargs,
    )


def test_default_throws_form_a_controlled_vertical_sweep() -> None:
    heights = [state.y for state in DEFAULT_INITIAL_STATES]

    assert len(heights) >= 12
    assert heights[:5] == pytest.approx([0.50, 0.08, 0.92, 0.15, 0.85])
    assert min(heights) <= 0.10
    assert max(heights) >= 0.90
    assert max(heights) - min(heights) >= 0.80
    assert {(state.x, state.vx, state.vy) for state in DEFAULT_INITIAL_STATES} == {
        (0.1, 0.18, 0.0)
    }


def test_episode_publishes_one_checkpoint_without_running_model2_synchronously() -> None:
    runtime = RecordingWorldRuntime()
    observer = RecordingObserver()
    session = _session(runtime, observer)

    episode = session.next_episode()

    assert episode.initial_state.y in {state.y for state in DEFAULT_INITIAL_STATES}
    assert episode.initial_state.x == pytest.approx(0.1)
    assert episode.initial_state.vx == pytest.approx(0.18)
    assert [step.label for step in episode.thought_steps] == [
        "first possibility",
        "more possibilities",
        "full belief",
        "reappearance revision",
    ]
    assert runtime.order == []
    assert len(observer.checkpoints) == 1
    assert isinstance(observer.checkpoints[0].inputs, BeliefInputs)
    assert observer.checkpoints[0].version == 1
    assert episode.thought_steps[-1].guessed_flat[0] == 0
    assert episode.training["loss"] == 0.0
    assert episode.training["weight_update_count"] == 0
    assert episode.training["seeing_updates"] == 1
    assert episode.latent_update_count == 1
    assert episode.weight_update_count == 0
    payload = episode.to_payload()
    assert payload["model1_checkpoint_version"] == 1
    assert payload["model2_checkpoint_version"] is None
    ground_truth = payload["ground_truth"]
    path = ground_truth["path"]
    assert "frames_base64" not in ground_truth
    assert ground_truth["occluder"] == pytest.approx([0.35, 0.55, 0.05, 0.95])
    assert len(path) > 1
    assert len(path) == ground_truth["frame_count"]
    assert path[0] == pytest.approx(payload["initial_state"][:2])
    assert all(len(point) == 2 for point in path)
    assert all(0.0 <= coordinate <= 1.0 for point in path for coordinate in point)
    assert payload["thought_steps"][0]["predicted_parameters"] == [0.0] * 7
    recovery = payload["thought_steps"][0]["constraint_recovery"]
    assert recovery["total"] == 3
    assert recovery["recovered"] == 2
    assert 0.8 < recovery["score"] < 1.0
    assert len(payload["thought_steps"][0]["paths"]) == 4
    assert len(payload["thought_steps"][2]["paths"]) == 24
    entry = payload["thought_steps"][0]["paths"][0][0]
    assert entry[0] == pytest.approx(ground_truth["occluder"][0])
    assert all(
        path_points[0] == pytest.approx(entry)
        for step in payload["thought_steps"]
        for path_points in step["paths"]
    )
    assert sum(payload["thought_steps"][2]["path_weights"]) == pytest.approx(1.0)
    assert all(
        len(point) == 2
        for path_points in payload["thought_steps"][2]["paths"]
        for point in path_points
    )


def test_successive_episodes_vary_offsets_and_keep_completed_event_history() -> None:
    runtime = RecordingWorldRuntime()
    observer = RecordingObserver()
    session = _session(runtime, observer)

    first = session.next_episode()
    second = session.next_episode()
    third = session.next_episode()

    assert len({first.initial_state, second.initial_state, third.initial_state}) == 3
    assert first.episode_index == 0
    assert second.episode_index == 1
    assert third.episode_index == 2
    assert observer.checkpoints[0].inputs.history_mask.tolist() == [[False, False, False, True]]
    assert observer.checkpoints[1].inputs.history_mask.tolist() == [[False, False, True, True]]
    assert observer.checkpoints[2].inputs.history_mask.tolist() == [[False, True, True, True]]
    assert session.completed_event_count == 3


def test_default_stream_realizes_probes_across_most_of_the_vertical_range() -> None:
    session = _session()

    episodes = [session.next_episode() for _ in DEFAULT_INITIAL_STATES]
    heights = [episode.initial_state.y for episode in episodes]

    assert min(heights) <= 0.10
    assert max(heights) >= 0.90
    assert max(heights) - min(heights) >= 0.80


def test_next_rollout_uses_latest_completed_model2_checkpoint_guess() -> None:
    observer = RecordingObserver()
    session = _session(observer=observer)

    first = session.next_episode()
    observer.latest = Model2CheckpointGuess(
        1,
        WorldPrediction(
            torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]]),
            torch.full((1, 7), 0.5),
        ),
    )
    second = session.next_episode()

    assert first.model2_checkpoint_version is None
    assert second.model1_checkpoint_version == 2
    assert second.model2_checkpoint_version == 1
    assert all(step.guessed_parameters.tolist() == [0.5] * 7 for step in second.thought_steps)
    assert len(observer.checkpoints) == 2


def test_close_stops_the_model2_observer() -> None:
    observer = RecordingObserver()
    session = _session(observer=observer)

    session.close()

    assert observer.closed


def test_default_observer_is_not_started_before_seeing_runtime_initializes(
    monkeypatch,
) -> None:
    created = []

    class TrackingObserver:
        def __init__(self, predict):
            created.append(predict)

    def fail_seeing_runtime(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("seeing initialization failed")

    monkeypatch.setattr("empathy.live.thinking.AsyncModel2Observer", TrackingObserver)
    monkeypatch.setattr("empathy.live.thinking.OnlineSeeingRuntime", fail_seeing_runtime)

    with pytest.raises(RuntimeError, match="seeing initialization failed"):
        _session()

    assert created == []


def test_live_session_persists_one_online_seeing_learner_across_throws() -> None:
    session = _session()
    learner = session.seeing_runtime

    first = session.next_episode()
    second = session.next_episode()

    assert session.seeing_runtime is learner
    assert learner.updates == 2
    assert first.training["seeing_updates"] == 1
    assert second.training["seeing_updates"] == 2
    assert len(second.thought_steps[-1].paths) == learner.model.config.sample_count


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
