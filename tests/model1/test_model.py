from pathlib import Path

import pytest
import torch

from perception_env.model1.config import load_model1_config
from perception_env.model1.model import Model1
from perception_env.model1.types import TrajectoryMixture


def _inputs(batch: int = 2):
    torch.manual_seed(3)
    frames = torch.rand(batch, 6, 1, 64, 64)
    times = torch.linspace(-0.5, 0.0, 6).repeat(batch, 1)
    mask = torch.tensor([[True, True, True, True, False, False]] * batch)
    return frames, times, mask


def test_model_emits_weighted_multimodal_trajectories_with_gradients() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    model = Model1(config.model, prediction_steps=12)
    frames, times, mask = _inputs()

    prediction = model(frames, times, mask)

    assert prediction.positions.shape == (2, 8, 12, 2)
    assert prediction.logits.shape == (2, 8)
    assert prediction.entry_state.shape == (2, 4)
    assert torch.all((prediction.positions >= 0) & (prediction.positions <= 1))
    assert torch.all((prediction.entry_state >= 0) & (prediction.entry_state <= 1))
    assert torch.allclose(prediction.weights.sum(dim=1), torch.ones(2))
    prediction.positions.sum().backward()
    assert model.frame_encoder.blocks[0].weight.grad is not None


def test_masked_timesteps_do_not_change_prediction() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    torch.manual_seed(11)
    model = Model1(config.model, prediction_steps=12).eval()
    frames, times, mask = _inputs()
    changed_frames = frames.clone()
    changed_times = times.clone()
    changed_frames[:, 4:] = 999.0
    changed_times[:, 4:] = 999.0

    first = model(frames, times, mask)
    second = model(changed_frames, changed_times, mask)

    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.entry_state, second.entry_state)


def test_traced_forward_exposes_the_late_temporal_activation() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    torch.manual_seed(13)
    model = Model1(config.model, prediction_steps=12).eval()
    frames, times, mask = _inputs()

    trace = model.forward_with_trace(frames, times, mask)
    prediction = model(frames, times, mask)

    assert trace.activation.shape == (2, config.model.hidden_size)
    assert trace.activation.requires_grad
    assert torch.equal(trace.prediction.positions, prediction.positions)
    assert torch.equal(trace.prediction.logits, prediction.logits)
    assert torch.equal(trace.prediction.entry_state, prediction.entry_state)


def test_masked_timesteps_do_not_change_traced_activation() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    torch.manual_seed(19)
    model = Model1(config.model, prediction_steps=12).eval()
    frames, times, mask = _inputs()
    changed_frames = frames.clone()
    changed_times = times.clone()
    changed_frames[:, 4:] = 999.0
    changed_times[:, 4:] = 999.0

    first = model.forward_with_trace(frames, times, mask)
    second = model.forward_with_trace(changed_frames, changed_times, mask)

    assert torch.equal(first.activation, second.activation)


def test_nan_padded_frames_do_not_poison_forward_or_gradients() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    model = Model1(config.model, prediction_steps=12)
    frames, times, mask = _inputs()
    frames[:, 4:] = float("nan")
    times[:, 4:] = float("nan")

    prediction = model(frames, times, mask)
    prediction.positions.sum().backward()

    assert torch.isfinite(prediction.positions).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_initialization_and_forward_replay_under_seed() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    frames, times, mask = _inputs()
    torch.manual_seed(17)
    first = Model1(config.model, prediction_steps=12)(frames, times, mask)
    torch.manual_seed(17)
    second = Model1(config.model, prediction_steps=12)(frames, times, mask)

    assert torch.equal(first.positions, second.positions)
    assert torch.equal(first.logits, second.logits)


def test_prediction_rejects_nonfinite_logits() -> None:
    prediction = TrajectoryMixture(
        positions=torch.zeros(1, 2, 3, 2),
        logits=torch.tensor([[0.0, float("nan")]]),
        entry_state=torch.zeros(1, 4),
    )

    with pytest.raises(ValueError, match="finite"):
        prediction.validate()
