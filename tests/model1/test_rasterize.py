import torch

from empathy.model1.rasterize import rasterize_actual_path, rasterize_mixture
from empathy.model1.types import TrajectoryMixture


def test_rasterizer_mixes_modes_and_preserves_interior_mass() -> None:
    positions = torch.tensor(
        [[[[0.25, 0.25], [0.5, 0.5]], [[0.75, 0.75], [0.5, 0.5]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    logits = torch.tensor([[-1.3862944, -0.2876821]], requires_grad=True)
    prediction = TrajectoryMixture(positions, logits, torch.zeros(1, 4))

    belief = rasterize_mixture(prediction, height=5, width=5)

    assert belief.time_belief.shape == (1, 2, 5, 5)
    assert belief.collapsed.shape == (1, 5, 5)
    assert torch.allclose(belief.time_belief.sum((-1, -2)), torch.ones(1, 2))
    assert torch.isclose(belief.time_belief[0, 0, 3, 1], torch.tensor(0.25))
    assert torch.isclose(belief.time_belief[0, 0, 1, 3], torch.tensor(0.75))
    assert torch.all((belief.collapsed >= 0) & (belief.collapsed <= 1))
    belief.collapsed.sum().backward()
    assert positions.grad is not None
    assert logits.grad is not None


def test_rasterizer_handles_edges_and_slightly_outside_coordinates() -> None:
    prediction = TrajectoryMixture(
        positions=torch.tensor(
            [[[[0.0, 0.0], [1.0, 1.0], [-0.01, 0.5], [1.01, 0.5]]]],
            requires_grad=True,
        ),
        logits=torch.zeros(1, 1, requires_grad=True),
        entry_state=torch.zeros(1, 4),
    )

    belief = rasterize_mixture(prediction, height=8, width=8)

    assert torch.isfinite(belief.time_belief).all()
    assert torch.all((belief.collapsed >= 0) & (belief.collapsed <= 1))
    assert torch.allclose(belief.time_belief.sum((-1, -2)), torch.ones(1, 4))


def test_actual_path_rasterizer_ignores_masked_steps() -> None:
    positions = torch.tensor([[[0.5, 0.5], [0.9, 0.9]]])
    mask = torch.tensor([[True, False]])

    belief = rasterize_actual_path(positions, mask, height=5, width=5)

    assert belief[0, 2, 2] == 1
    assert belief.sum() == 1


def test_mixture_step_mask_removes_tail_mass_and_gradients() -> None:
    positions = torch.tensor(
        [[[[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]]]], requires_grad=True
    )
    prediction = TrajectoryMixture(
        positions=positions,
        logits=torch.zeros(1, 1, requires_grad=True),
        entry_state=torch.zeros(1, 4),
    )
    mask = torch.tensor([[True, True, False]])

    belief = rasterize_mixture(prediction, height=5, width=5, step_mask=mask)
    belief.collapsed.sum().backward()

    assert belief.time_belief[0, 2].sum() == 0
    assert torch.count_nonzero(positions.grad[0, 0, 2]) == 0
    assert torch.isfinite(positions.grad).all()
