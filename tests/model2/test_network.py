import pytest
import torch

from empathy.model2.network import InverseWorldCNN


def test_cnn_shapes_gradients_and_padding_invariance():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = InverseWorldCNN()
    assert any(isinstance(layer, torch.nn.Conv2d) for layer in model.modules())
    beliefs = torch.zeros(2, 4, 3, 16, 16)
    beliefs[:, -1, 1, 4:8, 3:6] = 1
    beliefs[:, -1, 2] = beliefs[:, -1, 1]
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[:, -1] = True
    prediction = model(beliefs, mask)
    assert prediction.activity_logits.shape == (2, 5)
    assert prediction.normalized_parameters.shape == (2, 7)
    assert ((prediction.normalized_parameters >= 0) & (prediction.normalized_parameters <= 1)).all()
    prediction.activity_logits.sum().backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    model.eval()
    changed = beliefs.clone()
    changed[:, :3] = 999
    assert torch.equal(model(changed, mask).activity_logits, model(beliefs, mask).activity_logits)
    assert model(
        torch.zeros(1, 4, 3, 64, 64), torch.ones(1, 4, dtype=torch.bool)
    ).activity_logits.shape == (1, 5)
    with pytest.raises(TypeError):
        model(beliefs, mask, positions=torch.ones(1))


@pytest.mark.parametrize("kind", ["shape", "empty", "nan", "mask", "difference"])
def test_cnn_rejects_invalid_actual_inputs(kind):
    beliefs = torch.zeros(1, 4, 3, 16, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    if kind == "shape":
        beliefs = beliefs[:, :3]
    elif kind == "empty":
        mask[:] = False
    elif kind == "nan":
        beliefs[0, 0, 0, 0, 0] = float("nan")
    elif kind == "mask":
        mask = mask.float()
    elif kind == "difference":
        beliefs[0, 0, 2] = 1
    with pytest.raises(ValueError):
        InverseWorldCNN()(beliefs, mask)
