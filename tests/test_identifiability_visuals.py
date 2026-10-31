import numpy as np
from test_identifiability import fixture

from perception_env.reasoning.identifiability import group_equivalent_worlds
from perception_env.reasoning.visualize import render_equivalence_gallery


def test_equivalence_gallery_shows_world_codes_rasters_and_agreement(tmp_path) -> None:
    worlds, labels = fixture()
    rasters = np.zeros((5, 8, 8), dtype=bool)
    for index in range(5):
        rasters[index, 3, index + 1] = True
    report = group_equivalent_worlds(
        worlds, labels, tolerance=1e-8, possibility_rasters=rasters
    )

    artifact = render_equivalence_gallery(report, tmp_path / "equivalence.png")

    assert artifact.path.stat().st_size > 0
    assert artifact.metadata["panels"] == ["world code", "possibility raster"]
    assert artifact.metadata["class_sizes"] == [2, 2, 1]
    assert artifact.metadata["statuses"] == [
        "identifiable",
        "perceptually_equivalent",
        "identifiable",
    ]
    assert artifact.metadata["held_out_label_agreement"] is True
