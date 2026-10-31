import numpy as np
import pytest
import torch

from perception_env.model2.contracts import WorldPrediction
from perception_env.model2.decode import decode_prediction
from perception_env.physics.world import WorldSchema


def test_decoder_neutral_ties_one_sided_and_inactive_zeros():
    schema = WorldSchema.v1()
    prediction = WorldPrediction(
        torch.tensor([[100.0] * 5, [-100.0] * 5]),
        torch.tensor([[0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0]] * 2),
    )
    decoded = decode_prediction(prediction, schema)
    assert decoded.projection_counts.tolist() == [3, 0]
    assert np.all(decoded.flat[1] == 0)
    assert decoded.worlds[0].slot("vertical_acceleration").parameters[0] == pytest.approx(-0.050001)
    assert decoded.worlds[0].slot("damping").parameters[0] == pytest.approx(0.010001)
    for world in decoded.worlds:
        schema.validate(world)
    assert decoded.worlds[0].slot("vertical_wall").parameters == (0.05, 1.0)
    assert np.array_equal(decoded.flat, decode_prediction(prediction, schema).flat)


def test_decoder_rejects_nonfinite_predictions():
    with pytest.raises(ValueError):
        decode_prediction(
            WorldPrediction(torch.zeros(1, 5), torch.full((1, 7), float("nan"))), WorldSchema.v1()
        )
