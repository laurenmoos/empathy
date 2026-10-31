import numpy as np
import pytest
import torch

from perception_env.model2.activation_adapter import (
    ActivationAdapterConfig,
    ActivationBeliefAdapter,
    Model1ActivationPacket,
)
from perception_env.model2.adapters import (
    Model1BeliefPacket,
    adapt_model1_activations,
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


def activation_packet(ordinal=0, checkpoint="snapshot-a"):
    return Model1ActivationPacket(
        "activation-episode",
        ordinal,
        checkpoint,
        np.full(4, ordinal / 10, np.float32),
        np.full(4, (ordinal + 1) / 10, np.float32),
    )


def test_activation_adapter_builds_the_same_padded_belief_contract() -> None:
    torch.manual_seed(3)
    decoder = ActivationBeliefAdapter(
        ActivationAdapterConfig(input_size=4, height=8, width=8, hidden_size=6)
    ).eval()

    inputs = adapt_model1_activations([activation_packet(0)], decoder)

    assert inputs.beliefs.shape == (1, 4, 3, 8, 8)
    assert inputs.history_mask.tolist() == [[False, False, False, True]]
    assert not inputs.beliefs[0, :3].any()
    np.testing.assert_array_equal(
        inputs.beliefs[0, -1, 2],
        inputs.beliefs[0, -1, 1] - inputs.beliefs[0, -1, 0],
    )
    repeated = adapt_model1_activations([activation_packet(0)], decoder)
    np.testing.assert_array_equal(inputs.beliefs, repeated.beliefs)


def test_activation_history_rejects_mixed_model1_checkpoints() -> None:
    decoder = ActivationBeliefAdapter(
        ActivationAdapterConfig(input_size=4, height=8, width=8)
    )

    with pytest.raises(ValueError, match="checkpoint"):
        adapt_model1_activations(
            [activation_packet(0, "a"), activation_packet(1, "b")], decoder
        )


def test_shuffled_activation_control_preserves_shape_but_changes_evidence() -> None:
    torch.manual_seed(31)
    decoder = ActivationBeliefAdapter(
        ActivationAdapterConfig(input_size=4, height=8, width=8, hidden_size=6)
    ).eval()
    events = [
        Model1ActivationPacket(
            "activation-episode",
            0,
            "snapshot-a",
            np.arange(4, dtype=np.float32),
            np.arange(4, dtype=np.float32)[::-1],
        )
    ]

    full = adapt_model1_activations(events, decoder)
    shuffled = adapt_model1_activations(
        events, decoder, control_mode="shuffled", seed=7
    )

    assert shuffled.beliefs.shape == full.beliefs.shape
    assert shuffled.history_mask.tolist() == full.history_mask.tolist()
    assert not np.array_equal(shuffled.beliefs, full.beliefs)
