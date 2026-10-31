import math

import pytest
import torch

from empathy.model2.contracts import WorldPrediction
from empathy.model2.losses import world_code_loss


def test_analytic_activity_and_masked_parameter_loss():
    logits = torch.zeros(1, 5, requires_grad=True)
    parameters = torch.full((1, 7), 0.5, requires_grad=True)
    targets = torch.zeros(1, 12)
    targets[0, 0], targets[0, 1] = 1, 0.8
    targets[0, 2], targets[0, 3] = 1, 0.2
    identified = torch.zeros(1, 7, dtype=torch.bool)
    identified[0, 0] = True
    result = world_code_loss(WorldPrediction(logits, parameters), targets, identified)
    assert result.activity.item() == pytest.approx(math.log(2))
    assert result.parameters.item() == pytest.approx(0.09)
    assert result.parameter_count == 1
    result.total.backward()
    assert parameters.grad[0, 0].item() == pytest.approx(-0.6)
    assert torch.count_nonzero(parameters.grad[0, 1:]) == 0
    assert torch.count_nonzero(logits.grad) == 5


@pytest.mark.parametrize("active", [False, True])
def test_zero_selected_parameters_have_zero_finite_gradients(active):
    parameters = torch.full((2, 7), 0.5, requires_grad=True)
    targets = torch.zeros(2, 12)
    if active:
        targets[:, 0], targets[:, 1] = 1, 0.8
    result = world_code_loss(
        WorldPrediction(torch.zeros(2, 5, requires_grad=True), parameters),
        targets,
        torch.zeros(2, 7, dtype=torch.bool),
    )
    assert result.parameters.item() == 0
    assert result.parameter_count == 0
    assert torch.isfinite(result.total)
    result.total.backward()
    assert torch.count_nonzero(parameters.grad) == 0


def test_mask_uses_true_activity_not_predicted_activity():
    targets = torch.zeros(1, 12)
    targets[0, 0], targets[0, 1] = 1, 0.8
    prediction = WorldPrediction(torch.full((1, 5), -100.0), torch.full((1, 7), 0.5))
    result = world_code_loss(prediction, targets, torch.ones(1, 7, dtype=torch.bool))
    assert result.parameter_count == 1
    assert result.parameters.item() > 0


@pytest.mark.parametrize("kind", ["shape", "nan", "mask"])
def test_loss_rejects_invalid_targets(kind):
    targets = torch.zeros(1, 12)
    mask = torch.ones(1, 7, dtype=torch.bool)
    if kind == "shape":
        targets = targets[:, :11]
    elif kind == "nan":
        targets[0, 1] = float("nan")
    else:
        mask = mask.float()
    with pytest.raises(ValueError):
        world_code_loss(WorldPrediction(torch.zeros(1, 5), torch.zeros(1, 7)), targets, mask)
