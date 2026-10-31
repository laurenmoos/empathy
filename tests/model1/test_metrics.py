import math

import pytest
import torch

from empathy.model1.metrics import (
    belief_metrics,
    calibration_report,
    displacement_metrics,
    mixture_nll,
)
from empathy.model1.types import TrajectoryMixture


def _prediction() -> TrajectoryMixture:
    return TrajectoryMixture(
        positions=torch.tensor(
            [
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[1.0, 0.0], [1.0, 0.0]],
                ]
            ]
        ),
        logits=torch.log(torch.tensor([[0.8, 0.2]])),
        entry_state=torch.zeros(1, 4),
    )


def test_displacement_metrics_respect_mask_and_top_weight() -> None:
    targets = torch.tensor([[[1.0, 0.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, False]])

    metrics = displacement_metrics(_prediction(), targets, mask)

    assert metrics.min_ade.item() == 0
    assert metrics.min_fde.item() == 0
    assert metrics.top1_ade.item() == 1


def test_mixture_nll_matches_fixed_gaussian_mixture() -> None:
    targets = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, False]])
    sigma = 0.1
    expected_density = (0.8 + 0.2 * math.exp(-50.0)) / (2 * math.pi * sigma**2)

    nll = mixture_nll(_prediction(), targets, mask, sigma=sigma)

    assert nll.item() == pytest.approx(-math.log(expected_density), rel=1e-5)


def test_calibration_reports_top_mode_ece_and_credible_coverage() -> None:
    targets = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, True]])
    correct = calibration_report(_prediction(), targets, mask, radius=0.05, bins=10)
    wrong = calibration_report(
        _prediction(), torch.ones_like(targets), mask, radius=0.05, bins=10
    )
    certain = _prediction()
    certain = TrajectoryMixture(
        certain.positions,
        torch.tensor([[100.0, -100.0]]),
        certain.entry_state,
    )
    perfectly_calibrated = calibration_report(
        certain, targets, mask, radius=0.05, bins=10
    )

    assert correct.ece.item() == pytest.approx(0.2)
    assert wrong.ece.item() == pytest.approx(0.8)
    assert perfectly_calibrated.ece.item() == 0
    assert correct.credible_coverage == {"0.5": 1.0, "0.8": 1.0, "0.9": 1.0}


def test_belief_metrics_handle_identical_disjoint_and_empty() -> None:
    one = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    other = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    empty = torch.zeros_like(one)

    identical = belief_metrics(one, one)
    disjoint = belief_metrics(one, other)
    both_empty = belief_metrics(empty, empty)

    assert identical["soft_iou"].item() == 1
    assert identical["bce"].item() < 1e-5
    assert disjoint["soft_iou"].item() == 0
    assert both_empty["soft_iou"].item() == 1
