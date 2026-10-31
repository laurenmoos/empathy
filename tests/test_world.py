import numpy as np
import pytest

from perception_env.physics.world import WorldSchema


def test_v1_world_round_trips_flat_representation() -> None:
    schema = WorldSchema.v1()
    world = schema.world(
        vertical_acceleration={"a_y": -0.6},
        damping={"lambda": 0.4},
        vertical_wall={"position": 0.82, "restitution": 0.75},
    )

    encoded = world.to_flat(schema)
    restored = schema.from_flat(encoded)

    assert encoded.shape == (12,)
    assert restored == world
    assert schema.field_order == (
        "vertical_acceleration.active",
        "vertical_acceleration.a_y",
        "horizontal_acceleration.active",
        "horizontal_acceleration.a_x",
        "damping.active",
        "damping.lambda",
        "vertical_wall.active",
        "vertical_wall.position",
        "vertical_wall.restitution",
        "horizontal_wall.active",
        "horizontal_wall.position",
        "horizontal_wall.restitution",
    )


def test_inactive_parameters_are_canonicalized() -> None:
    schema = WorldSchema.v1()

    world = schema.world(inactive_parameters={"vertical_acceleration": {"a_y": 0.8}})

    assert world.slot("vertical_acceleration").parameters == (0.0,)
    assert np.array_equal(world.to_flat(schema)[:2], np.array([0.0, 0.0]))


@pytest.mark.parametrize(
    ("slot", "parameters"),
    [
        ("vertical_acceleration", {"a_y": 0.0}),
        ("horizontal_acceleration", {"a_x": 0.01}),
        ("damping", {"lambda": 0.0}),
    ],
)
def test_active_continuous_operator_rejects_neutral_value(slot: str, parameters: dict) -> None:
    schema = WorldSchema.v1()

    with pytest.raises(ValueError, match="neutral interval"):
        schema.world(**{slot: parameters})


def test_from_flat_rejects_wrong_width() -> None:
    schema = WorldSchema.v1()

    with pytest.raises(ValueError, match="expected 12 values"):
        schema.from_flat(np.zeros(11))

