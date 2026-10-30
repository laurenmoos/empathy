import numpy as np
import pytest

from perception_env.model2.adapters import (
    Model1BeliefPacket,
    adapt_model1_events,
    parse_model1_packet,
)


def packet(ordinal=0, episode="a"):
    return Model1BeliefPacket(episode, ordinal, np.zeros((8, 8)), np.full((8, 8), 0.5))


def test_adapter_only_passes_beliefs_and_derives_difference():
    inputs = adapt_model1_events([packet(i) for i in range(6)])
    assert inputs.beliefs.shape == (1, 4, 3, 8, 8)
    assert inputs.history_mask.all()
    np.testing.assert_array_equal(inputs.beliefs[0, :, 2], 0.5)
    padded = adapt_model1_events([packet()])
    assert padded.history_mask.tolist() == [[False, False, False, True]]
    assert not padded.beliefs[:, :3].any()


@pytest.mark.parametrize(
    "field",
    [
        "frames",
        "coordinates",
        "world_flat",
        "targets",
        "markers",
        "entry_state",
        "belief_difference",
    ],
)
def test_packet_parser_rejects_first_order_or_target_fields(field):
    with pytest.raises(ValueError, match="fields"):
        parse_model1_packet(
            {
                "episode_id": "a",
                "event_ordinal": 0,
                "pre_belief": np.zeros((8, 8)),
                "post_belief": np.zeros((8, 8)),
                field: None,
            }
        )


@pytest.mark.parametrize(
    "events", [[], [packet(1), packet(0)], [packet(), packet()], [packet(), packet(1, "b")]]
)
def test_adapter_rejects_inconsistent_histories(events):
    with pytest.raises(ValueError):
        adapt_model1_events(events)


@pytest.mark.parametrize(
    "value", [np.full((8, 8), np.nan), np.ones((8, 8)) * 2, np.zeros((3, 8, 8)), np.zeros((7, 7))]
)
def test_packet_validates_beliefs_at_construction(value):
    with pytest.raises(ValueError):
        Model1BeliefPacket("a", 0, value, value)
