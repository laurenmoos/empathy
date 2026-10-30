from pathlib import Path

import numpy as np

from perception_env.families import load_family, sample_worlds
from perception_env.world import WorldSchema

CONFIG = Path("configs/world_families/v1.yaml")


def test_world_family_levels_are_nested_by_membership() -> None:
    schema = WorldSchema.v1()
    examples = {
        "W0": schema.world(),
        "W1": schema.world(vertical_acceleration={"a_y": -0.4}),
        "W2": schema.world(horizontal_acceleration={"a_x": 0.3}),
        "W3": schema.world(
            vertical_acceleration={"a_y": -0.4}, damping={"lambda": 0.2}
        ),
        "W4": schema.world(
            vertical_acceleration={"a_y": -0.4},
            damping={"lambda": 0.2},
            vertical_wall={"position": 0.8, "restitution": 0.7},
        ),
        "W5": schema.world(
            vertical_acceleration={"a_y": -0.4},
            horizontal_acceleration={"a_x": 0.3},
            damping={"lambda": 0.2},
            vertical_wall={"position": 0.8, "restitution": 0.7},
            horizontal_wall={"position": 0.2, "restitution": 0.7},
        ),
    }

    for level_index in range(6):
        family = load_family(CONFIG, f"W{level_index}")
        for example_index in range(6):
            assert family.contains(examples[f"W{example_index}"]) is (
                example_index <= level_index
            )


def test_seeded_world_sampling_is_stable_and_excludes_neutral_active_values() -> None:
    family = load_family(CONFIG, "W5")

    first = sample_worlds(family, count=20, seed=123)
    second = sample_worlds(family, count=20, seed=123)

    assert first == second
    assert any(any(slot.active for slot in world.slots) for world in first)
    for world in first:
        family.schema.validate(world)
        assert family.contains(world)
        assert np.isfinite(world.to_flat(family.schema)).all()


def test_family_exposes_versioned_uncertainty_and_invalid_policy() -> None:
    family = load_family(CONFIG, "W3")

    assert family.schema.version == "v1"
    assert family.family_version == "v1"
    assert family.invalid_rollout_policy == "terminate_and_reject"
    assert family.entry_grid.resolution == (3, 3, 3, 3)

