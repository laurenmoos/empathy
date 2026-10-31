import numpy as np
import pytest

from empathy.physics.config import RenderConfig
from empathy.physics.simulate import BallState, TerminationReason, Trajectory
from empathy.reasoning.beliefs import BeliefEvent, build_belief_history, superpose


def trajectory(points: list[tuple[float, float]], seed: int = 0) -> Trajectory:
    states = tuple(BallState(x, y, 0.0, 0.0) for x, y in points)
    return Trajectory(states, (), True, seed, TerminationReason.COMPLETED)


def test_superpose_normalizes_weighted_binary_path_masks() -> None:
    paths = [
        trajectory([(0.0, 0.0), (0.5, 0.5)]),
        trajectory([(0.0, 0.0), (1.0, 1.0)]),
    ]

    density = superpose(paths, np.array([1.0, 3.0]), RenderConfig(width=3, height=3))

    assert density.shape == (3, 3)
    assert density[2, 0] == pytest.approx(1.0)
    assert density[1, 1] == pytest.approx(0.25)
    assert density[0, 2] == pytest.approx(0.75)


def test_superpose_empty_paths_returns_zero_image() -> None:
    density = superpose([], np.array([]), RenderConfig(width=4, height=3))

    assert np.array_equal(density, np.zeros((3, 4), dtype=np.float32))


def test_belief_history_pads_chronologically_and_computes_exact_difference() -> None:
    event_2 = BeliefEvent(
        step=2,
        pre=np.array([[0.2, 0.4]], dtype=np.float32),
        post=np.array([[0.3, 0.1]], dtype=np.float32),
    )
    event_5 = BeliefEvent(
        step=5,
        pre=np.array([[0.6, 0.2]], dtype=np.float32),
        post=np.array([[0.1, 0.9]], dtype=np.float32),
    )

    history = build_belief_history([event_5, event_2], length=4)

    assert history.tensor.shape == (4, 3, 1, 2)
    assert history.mask.tolist() == [False, False, True, True]
    assert history.steps.tolist() == [-1, -1, 2, 5]
    np.testing.assert_allclose(history.tensor[2, 2], [[0.1, -0.3]], atol=1e-7)
    np.testing.assert_allclose(history.tensor[3, 2], [[-0.5, 0.7]], atol=1e-7)
    assert history.channel_names == ("pre_belief", "post_belief", "belief_difference")


def test_belief_event_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        BeliefEvent(step=1, pre=np.zeros((2, 2)), post=np.zeros((3, 3)))

