from empathy.physics.config import RenderConfig
from empathy.physics.events import extract_reappearance
from empathy.physics.render import Occluder, render_sequence
from empathy.physics.simulate import BallState, TerminationReason, Trajectory


def sequence_for(xs: list[float], min_hidden_steps: int = 2):
    states = tuple(BallState(x, 0.5, 0.1, 0.0) for x in xs)
    trajectory = Trajectory(states, (), True, 5, TerminationReason.COMPLETED)
    return render_sequence(
        trajectory,
        (Occluder(0.35, 0.65, 0.35, 0.65),),
        RenderConfig(width=64, height=64, ball_radius=2, min_hidden_steps=min_hidden_steps),
    )


def test_extracts_first_visible_reappearance_after_hidden_run() -> None:
    sequence = sequence_for([0.2, 0.4, 0.5, 0.6, 0.8])

    event = extract_reappearance(sequence)

    assert event is not None
    assert event.step == 4
    assert event.x == 0.8
    assert event.y == 0.5
    assert event.hidden_start == 1
    assert event.hidden_end == 3
    assert sequence.rejection_reason is None


def test_rejects_hidden_run_shorter_than_configured_minimum() -> None:
    sequence = sequence_for([0.2, 0.5, 0.8], min_hidden_steps=2)

    assert extract_reappearance(sequence) is None
    assert sequence.rejection_reason == "insufficient_hidden_run"


def test_rejects_sequence_without_visible_reappearance() -> None:
    sequence = sequence_for([0.2, 0.4, 0.5, 0.6])

    assert extract_reappearance(sequence) is None
    assert sequence.rejection_reason == "no_visible_reappearance"

