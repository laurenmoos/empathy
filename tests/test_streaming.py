from dataclasses import asdict

import numpy as np
import pytest

from perception_env.beliefs import BeliefEvent
from perception_env.streaming import (
    Model2Features,
    OcclusionEpisode,
    StreamingSession,
    StreamingSessionState,
    StreamWorldPrediction,
)
from perception_env.world import WorldSchema


def _event(step: int, x: int) -> BeliefEvent:
    pre = np.zeros((8, 8), np.float32)
    post = pre.copy()
    post[4, x] = 1
    return BeliefEvent(step, pre, post)


def _episode() -> OcclusionEpisode:
    return OcclusionEpisode(
        "event-1",
        "context-1",
        np.zeros((4, 8, 8), np.float32),
        np.asarray([True, False, False, True]),
        3,
        (0.75, 0.5),
        {"true_world_flat": WorldSchema.v1().world().to_flat(WorldSchema.v1()).tolist()},
    )


def test_model2_features_are_right_aligned_and_reject_first_order_fields() -> None:
    features = Model2Features.from_events([_event(2, 2), _event(4, 4)])
    assert features.pre_belief.shape == (4, 8, 8)
    assert features.history_mask.tolist() == [False, False, True, True]
    assert not features.pre_belief[:2].any()
    values = asdict(features)
    for forbidden in ("frames", "position", "visible", "outcome", "target_world"):
        with pytest.raises(ValueError, match="exactly"):
            Model2Features.from_mapping({**values, forbidden: np.zeros(1)})


def test_session_state_round_trips_without_model_objects() -> None:
    state = StreamingSessionState(
        world_flat=WorldSchema.v1().world().to_flat(WorldSchema.v1()),
        events=(_event(2, 2),),
        processed_event_ids=("event-1",),
        latent_update_count=1,
        weight_update_count=0,
    )
    restored = StreamingSessionState.from_mapping(state.to_mapping())
    assert np.array_equal(restored.world_flat, state.world_flat)
    assert restored.processed_event_ids == state.processed_event_ids


class FakeBeliefs:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def propose(self, episode, world_prior):
        self.calls.append("propose")
        image = np.zeros((8, 8), np.float32)
        image[4, 2 if world_prior is None else 6] = 1
        return image

    def assimilate(self, episode, pre_belief):
        self.calls.append("assimilate")
        image = pre_belief.copy()
        image[4, 6] = 1
        return image


class FakeWorlds:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        schema = WorldSchema.v1()
        flat = schema.world(horizontal_acceleration={"a_x": 0.2}).to_flat(schema)
        return StreamWorldPrediction(flat, np.ones(5), np.zeros(7), 0.8)


def test_uncovered_outcome_triggers_bounded_feedback_and_updates_history() -> None:
    beliefs = FakeBeliefs()
    worlds = FakeWorlds()

    def possibility(world, episode):
        image = np.zeros((8, 8), bool)
        if worlds.calls >= 2:
            image[4, 6] = True
        return image

    session = StreamingSession(
        beliefs,
        worlds,
        possibility,
        initial_world_flat=WorldSchema.v1().world().to_flat(WorldSchema.v1()),
        max_feedback_iterations=3,
        outcome_tolerance_pixels=1,
    )
    update = session.process(_episode())
    assert update.status == "covered"
    assert len(update.iterations) == 2
    assert beliefs.calls == ["propose", "assimilate", "propose", "assimilate"]
    assert session.snapshot().latent_update_count == 2
    assert len(session.snapshot().events) == 1


def test_unresolved_feedback_stops_at_limit() -> None:
    session = StreamingSession(
        FakeBeliefs(),
        FakeWorlds(),
        lambda world, episode: np.zeros((8, 8), bool),
        initial_world_flat=WorldSchema.v1().world().to_flat(WorldSchema.v1()),
        max_feedback_iterations=2,
        outcome_tolerance_pixels=0,
    )
    update = session.process(_episode())
    assert update.status == "unresolved"
    assert len(update.iterations) == 2

