from dataclasses import replace
from pathlib import Path

import pytest
import torch

from perception_env.model1.config import load_model1_config
from perception_env.model1.losses import model1_loss
from perception_env.model1.types import (
    BeliefPrediction,
    Model1Batch,
    TrajectoryMixture,
)


def _prediction(*, nan: bool = False) -> TrajectoryMixture:
    positions = torch.tensor(
        [[[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], [[0.9, 0.9], [0.8, 0.8], [0.7, 0.7]]]],
        requires_grad=True,
    )
    if nan:
        positions = positions.clone()
        positions[0, 0, 0, 0] = float("nan")
    return TrajectoryMixture(
        positions=positions,
        logits=torch.zeros(1, 2, requires_grad=True),
        entry_state=torch.zeros(1, 4, requires_grad=True),
    )


def _belief(value: float) -> BeliefPrediction:
    return BeliefPrediction(
        time_belief=torch.full((1, 3, 4, 4), value, requires_grad=True),
        collapsed=torch.full((1, 4, 4), value, requires_grad=True),
    )


def _batch(mask=None) -> Model1Batch:
    if mask is None:
        mask = torch.tensor([[True, True, True]])
    return Model1Batch(
        pre_frames=torch.zeros(1, 2, 1, 4, 4),
        pre_times=torch.zeros(1, 2),
        pre_mask=torch.ones(1, 2, dtype=torch.bool),
        post_frames=torch.zeros(1, 2, 1, 4, 4),
        post_times=torch.zeros(1, 2),
        post_mask=torch.ones(1, 2, dtype=torch.bool),
        entry_state=torch.ones(1, 4),
        target_positions=torch.tensor([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]]),
        target_mask=mask,
        oracle_pre_belief=torch.ones(1, 4, 4),
        actual_path_belief=torch.zeros(1, 4, 4),
        example_ids=("example",),
        family_levels=("W0",),
    )


def test_loss_selects_best_mode_and_backpropagates_to_logits() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    pre = _prediction()
    post = _prediction()

    terms = model1_loss(pre, post, _belief(0.5), _belief(0.5), _batch(), config.loss)

    assert terms.trajectory == 0
    terms.total.backward()
    assert pre.logits.grad is not None
    assert torch.count_nonzero(pre.logits.grad) == 2


def test_zero_state_weight_removes_state_term() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    loss_config = replace(config.loss, state_weight=0.0)

    terms = model1_loss(
        _prediction(), _prediction(), _belief(0.5), _belief(0.5), _batch(), loss_config
    )

    assert terms.state == 0


def test_masked_target_values_do_not_change_loss() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    mask = torch.tensor([[True, True, False]])
    first = _batch(mask)
    changed = replace(first, target_positions=first.target_positions.clone())
    changed.target_positions[0, 2] = torch.tensor([999.0, 999.0])

    loss_a = model1_loss(
        _prediction(), _prediction(), _belief(0.5), _belief(0.5), first, config.loss
    )
    loss_b = model1_loss(
        _prediction(), _prediction(), _belief(0.5), _belief(0.5), changed, config.loss
    )

    assert torch.equal(loss_a.total, loss_b.total)


def test_loss_rejects_empty_targets_and_nonfinite_predictions() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    with pytest.raises(ValueError, match="valid target"):
        model1_loss(
            _prediction(),
            _prediction(),
            _belief(0.5),
            _belief(0.5),
            _batch(torch.zeros(1, 3, dtype=torch.bool)),
            config.loss,
        )
    with pytest.raises(ValueError, match="finite"):
        model1_loss(
            _prediction(nan=True),
            _prediction(),
            _belief(0.5),
            _belief(0.5),
            _batch(),
            config.loss,
        )


def test_pre_and_post_beliefs_use_different_targets() -> None:
    config = load_model1_config(Path("configs/model1/v1.yaml"))
    terms = model1_loss(
        _prediction(), _prediction(), _belief(0.99), _belief(0.01), _batch(), config.loss
    )

    assert terms.pre_belief < 0.01
    assert terms.post_belief < 0.01
