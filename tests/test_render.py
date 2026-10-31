import numpy as np

from perception_env.physics.config import RenderConfig
from perception_env.physics.render import Occluder, render_sequence
from perception_env.physics.simulate import BallState, TerminationReason, Trajectory


def make_trajectory(xs: list[float], y: float = 0.5) -> Trajectory:
    states = tuple(BallState(x, y, 0.1, 0.0) for x in xs)
    return Trajectory(states, (), True, 11, TerminationReason.COMPLETED)


def test_render_sequence_shapes_visibility_and_latent_invariance() -> None:
    trajectory = make_trajectory([0.2, 0.4, 0.5, 0.6, 0.8])
    original_states = trajectory.states
    occluder = Occluder(0.35, 0.65, 0.35, 0.65)

    rendered = render_sequence(
        trajectory, (occluder,), RenderConfig(width=32, height=32, ball_radius=1)
    )

    assert rendered.frames.shape == (5, 32, 32)
    assert rendered.frames.dtype == np.uint8
    assert rendered.visible.tolist() == [True, False, False, False, True]
    assert rendered.fully_hidden.tolist() == [False, True, True, True, False]
    assert rendered.partially_occluded.tolist() == [False] * 5
    assert np.count_nonzero(rendered.visible_ball_masks[2]) == 0
    assert np.count_nonzero(rendered.ball_masks[2]) > 0
    assert trajectory.states == original_states
    assert rendered.latent_states == original_states


def test_render_sequence_marks_partial_only_occlusion() -> None:
    trajectory = make_trajectory([0.2, 0.33, 0.2])

    rendered = render_sequence(
        trajectory,
        (Occluder(0.35, 0.65, 0.35, 0.65),),
        RenderConfig(width=64, height=64, ball_radius=2),
    )

    assert rendered.partially_occluded.any()
    assert not rendered.fully_hidden.any()
    assert rendered.rejection_reason == "partial_only_occlusion"


def test_render_sequence_rejects_no_occlusion() -> None:
    rendered = render_sequence(
        make_trajectory([0.1, 0.2, 0.3]),
        (Occluder(0.7, 0.9, 0.2, 0.8),),
        RenderConfig(width=32, height=32, ball_radius=1),
    )

    assert rendered.rejection_reason == "no_occlusion"

