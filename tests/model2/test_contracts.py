import numpy as np
import pytest

from perception_env.model2.contracts import BeliefInputs, SchemaLayout, WorldTargets
from perception_env.world import WorldSchema


def inputs(n=2, size=16):
    values = np.zeros((n, 4, 3, size, size), dtype=np.float32)
    values[:, -1, 1, 2:4, 3:5] = 1
    values[:, -1, 2] = values[:, -1, 1]
    mask = np.zeros((n, 4), dtype=bool)
    mask[:, -1] = True
    return values, mask


def test_layout_and_input_copy():
    layout = SchemaLayout.from_schema(WorldSchema.v1())
    assert layout.activity_indices == (0, 2, 4, 6, 9)
    assert layout.parameter_indices == (1, 3, 5, 7, 8, 10, 11)
    assert layout.parameter_to_slot == (0, 1, 2, 3, 3, 4, 4)
    values, mask = inputs()
    record = BeliefInputs(values, mask)
    values[:] = 5
    assert record.beliefs.max() == 1
    assert not record.beliefs.flags.writeable


@pytest.mark.parametrize(
    "name",
    [
        "frames",
        "positions",
        "features",
        "event_marker",
        "target_world",
        "entropy",
        "metadata",
        "visibility_masks",
    ],
)
def test_rejects_nonbelief_fields(name):
    values, mask = inputs()
    with pytest.raises(ValueError, match="fields"):
        BeliefInputs.from_mapping({"beliefs": values, "history_mask": mask, name: 0})


@pytest.mark.parametrize(
    "kind",
    ["shape", "nan", "difference", "padding", "mask", "empty", "range", "resolution", "suffix"],
)
def test_invalid_inputs(kind):
    values, mask = inputs()
    if kind == "shape":
        values = values[:, :3]
    elif kind == "nan":
        values[0, -1, 0, 0, 0] = np.nan
    elif kind == "difference":
        values[0, -1, 2] = 0
    elif kind == "padding":
        values[0, 0] = 1
    elif kind == "mask":
        mask = mask.astype(int)
    elif kind == "empty":
        mask[:] = False
    elif kind == "range":
        values[0, -1, 0] = -1
    elif kind == "resolution":
        values = values[..., :4, :4]
    elif kind == "suffix":
        mask[0] = [True, False, False, True]
    with pytest.raises(ValueError):
        BeliefInputs(values, mask)


def test_zero_valid_beliefs_and_target_validation():
    BeliefInputs(np.zeros((1, 4, 3, 8, 8)), np.ones((1, 4), dtype=bool))
    world = WorldSchema.v1().world().to_flat(WorldSchema.v1())[None]
    target = WorldTargets(world, np.zeros((1, 7), dtype=bool))
    assert target.world_flat.shape == (1, 12)
    for invalid in [np.ones((1, 11)), world + 0.1, np.full((1, 12), np.nan)]:
        with pytest.raises(ValueError):
            WorldTargets(invalid, np.zeros((1, 7), dtype=bool))
