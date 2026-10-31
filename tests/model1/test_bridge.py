import numpy as np
import pytest
import torch

from perception_env.model1.rasterize import belief_event_from_predictions
from perception_env.model1.types import BeliefPrediction
from perception_env.reasoning.beliefs import build_belief_history
from perception_env.reasoning.records import Model2InputRecord, belief_entropy


def test_bridge_exports_only_collapsed_beliefs() -> None:
    pre = BeliefPrediction(torch.zeros(1, 2, 4, 4), torch.zeros(1, 4, 4))
    post = BeliefPrediction(torch.ones(1, 2, 4, 4), torch.ones(1, 4, 4))
    events = [belief_event_from_predictions(pre, post, event_step=index) for index in range(4)]
    history = build_belief_history(events)
    mapping = {
        "pre_belief": history.tensor[:, 0],
        "post_belief": history.tensor[:, 1],
        "belief_difference": history.tensor[:, 2],
        "entropy": belief_entropy(history.tensor[:, 0]),
        "history_mask": history.mask,
        "target_world": np.zeros(12, dtype=np.float32),
    }

    record = Model2InputRecord.from_mapping(mapping)

    assert set(mapping) == {
        "pre_belief",
        "post_belief",
        "belief_difference",
        "entropy",
        "history_mask",
        "target_world",
    }
    assert np.array_equal(record.belief_difference, record.post_belief - record.pre_belief)


@pytest.mark.parametrize(
    "field",
    ["frames", "entry_state", "latent_positions", "family_level", "modes", "weights"],
)
def test_model2_record_rejects_model1_first_order_and_latent_fields(field) -> None:
    values = {
        "pre_belief": np.zeros((4, 4, 4), dtype=np.float32),
        "post_belief": np.zeros((4, 4, 4), dtype=np.float32),
        "belief_difference": np.zeros((4, 4, 4), dtype=np.float32),
        "entropy": belief_entropy(np.zeros((4, 4, 4), dtype=np.float32)),
        "history_mask": np.ones(4, dtype=bool),
        "target_world": np.zeros(12, dtype=np.float32),
        field: np.zeros(1),
    }

    with pytest.raises(ValueError, match="unsupported|first-order"):
        Model2InputRecord.from_mapping(values)
