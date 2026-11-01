from __future__ import annotations

import torch

from empathy.model1.online import (
    HiddenPathBelief,
    OnlineSeeingConfig,
    OnlineSeeingModel,
    SharedVideoEncoder3D,
)


def test_video_encoder_ignores_masked_frames() -> None:
    torch.manual_seed(1)
    encoder = SharedVideoEncoder3D((4, 8), output_size=12)
    frames = torch.rand(2, 5, 1, 24, 24)
    mask = torch.tensor([[True, True, True, False, False]] * 2)
    changed = frames.clone()
    changed[:, 3:] = 100.0
    assert torch.allclose(encoder(frames, mask), encoder(changed, mask))


def test_hidden_path_sampling_is_seeded_and_decodes_finite_beziers() -> None:
    belief = HiddenPathBelief(torch.zeros(1, 6), torch.full((1, 6), -1.0))
    one = belief.sample(7, generator=torch.Generator().manual_seed(4))
    two = belief.sample(7, generator=torch.Generator().manual_seed(4))
    assert torch.equal(one, two)

    entry = torch.tensor([[0.38, 0.55, 0.25, 0.0]])
    paths = belief.decode(one, entry, occluder=(0.38, 0.58), steps=9)
    assert paths.shape == (1, 7, 9, 2)
    assert torch.isfinite(paths).all()
    assert torch.allclose(paths[:, :, 0], entry[:, None, :2], atol=1e-6)
    assert torch.allclose(paths[:, :, -1, 0], torch.full((1, 7), 0.58), atol=1e-6)


def test_pre_belief_is_differentiable() -> None:
    torch.manual_seed(2)
    config = OnlineSeeingConfig(
        channels=(4, 8), representation_size=12, world_size=5, sample_count=6
    )
    model = OnlineSeeingModel(config)
    frames = torch.rand(2, 5, 1, 24, 24)
    mask = torch.tensor([[True, True, True, True, False]] * 2)
    entry = torch.tensor([[0.38, 0.45, 0.3, 0.0], [0.38, 0.6, 0.3, 0.0]])
    world = torch.zeros(2, 5, requires_grad=True)

    belief = model.pre_belief(frames, mask, entry, world)
    samples = belief.sample(6, generator=torch.Generator().manual_seed(3))
    paths = belief.decode(samples, entry, occluder=(0.38, 0.58), steps=8)
    (paths.mean() + model.encode_post(frames, mask).mean()).backward()

    assert belief.mean.shape == (2, 6)
    assert world.grad is not None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_duration_controls_exit_velocity_without_changing_draw_resolution() -> None:
    belief = HiddenPathBelief(torch.zeros(1, 6), torch.zeros(1, 6))
    samples = torch.zeros(1, 1, 6, requires_grad=True)
    entry = torch.tensor([[0.38, 0.55, 0.25, 0.0]])
    exit_state = belief.implied_exit_state(samples, entry, occluder=(0.38, 0.58))
    exit_state[..., 2:].sum().backward()
    assert samples.grad is not None
    assert samples.grad[..., 5].abs().item() > 0

    short = belief.decode(samples.detach(), entry, occluder=(0.38, 0.58), steps=8)
    long = belief.decode(samples.detach(), entry, occluder=(0.38, 0.58), steps=64)
    assert torch.allclose(short[:, :, -1], long[:, :, -1])
    assert torch.allclose(
        belief.implied_exit_state(samples.detach(), entry, occluder=(0.38, 0.58)),
        exit_state.detach(),
    )
