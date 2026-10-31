from test_records import make_record

from perception_env.physics.world import WorldSchema
from perception_env.reasoning.records import load_record, save_record
from perception_env.reasoning.visualize import render_dataset_contact_sheet


def test_contact_sheet_renders_from_reloaded_records(tmp_path) -> None:
    saved = [save_record(make_record(), tmp_path / f"source-{index}") for index in range(2)]
    records = [load_record(path) for path in saved]

    artifact = render_dataset_contact_sheet(
        records, WorldSchema.v1(), tmp_path / "contact-sheet.png"
    )

    assert artifact.path.stat().st_size > 0
    assert artifact.metadata["source"] == "reloaded_records"
    assert artifact.metadata["record_ids"] == ["example-19", "example-19"]
    assert artifact.metadata["panels_per_record"] == [
        "world code",
        "observed frame",
        "pre belief",
    ]
