from __future__ import annotations

import torch

from empathy.model1.online import (
    OnlineSeeingConfig,
    SharedVideoEncoder3D,
    StepwiseWorldModel,
)


def test_video_encoder_ignores_masked_frames() -> None:
    torch.manual_seed(1)
    encoder = SharedVideoEncoder3D((4, 8), output_size=12)
    frames = torch.rand(2, 5, 1, 24, 24)
    mask = torch.tensor([[True, True, True, False, False]] * 2)
    changed = frames.clone()
    changed[:, 3:] = 100.0
    assert torch.allclose(encoder(frames, mask), encoder(changed, mask))


def test_stepwise_world_model_rolls_forward_one_physical_step_at_a_time() -> None:
    torch.manual_seed(4)
    config = OnlineSeeingConfig(
        channels=(4, 8), representation_size=12, world_size=5, sample_count=3
    )
    model = StepwiseWorldModel(config, hidden_size=16)
    frames = torch.rand(1, 4, 1, 20, 20)
    mask = torch.ones(1, 4, dtype=torch.bool)
    entry = torch.tensor([[0.35, 0.5, 0.2, 0.0]])
    world = torch.zeros(1, 5)
    belief = model.initial_belief(frames, mask, entry, world)
    samples = belief.sample(3, generator=torch.Generator().manual_seed(9))
    paths = model.rollout(samples, entry, world, steps=6, dt=0.1)

    assert paths.shape == (1, 3, 6, 2)
    assert torch.isfinite(paths).all()
    assert torch.all(paths[..., 1:, 0] > paths[..., :-1, 0])
    paths.mean().backward()
    assert any(parameter.grad is not None for parameter in model.transition.parameters())


def test_stepwise_rollout_is_constant_velocity_when_acceleration_is_zero() -> None:
    config = OnlineSeeingConfig(channels=(4,), representation_size=8, world_size=3)
    model = StepwiseWorldModel(config, hidden_size=8)
    torch.nn.init.zeros_(model.acceleration.weight)
    torch.nn.init.zeros_(model.acceleration.bias)
    samples = torch.zeros(1, 1, 8)
    entry = torch.tensor([[0.35, 0.4, 0.2, -0.1]])
    paths = model.rollout(samples, entry, torch.zeros(1, 3), steps=3, dt=0.1)
    expected = torch.tensor([[[[0.37, 0.39], [0.39, 0.38], [0.41, 0.37]]]])
    assert torch.allclose(paths, expected, atol=1e-6)
