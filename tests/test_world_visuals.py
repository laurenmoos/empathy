import json

from empathy.physics.world import WorldSchema
from empathy.reasoning.visualize import render_world_card


def test_world_card_is_reproducible_and_names_every_operator(tmp_path) -> None:
    schema = WorldSchema.v1()
    world = schema.world(
        vertical_acceleration={"a_y": -0.5},
        horizontal_acceleration={"a_x": 0.2},
    )

    first = render_world_card(world, schema, tmp_path / "first.png")
    second = render_world_card(world, schema, tmp_path / "second.png")

    assert first.path.stat().st_size > 0
    assert first.metadata == second.metadata
    assert first.metadata["schema_version"] == "v1"
    assert first.metadata["operators"] == list(schema.slot_names)
    assert json.loads(first.metadata_path.read_text()) == first.metadata
