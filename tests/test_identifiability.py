import numpy as np

from perception_env.physics.world import WorldSchema
from perception_env.reasoning.identifiability import group_equivalent_worlds, parameter_scoring_mask
from perception_env.reasoning.reachability import Probe
from perception_env.reasoning.shattering import LabelMatrix


def fixture(search_complete: bool = True):
    schema = WorldSchema.v1()
    worlds = [
        schema.world(),
        schema.world(),
        schema.world(vertical_acceleration={"a_y": -0.2}),
        schema.world(vertical_acceleration={"a_y": -0.8}),
        schema.world(horizontal_acceleration={"a_x": 0.3}),
    ]
    labels = LabelMatrix(
        values=np.asarray([[0, 0], [0, 0], [0, 1], [0, 1], [1, 1]], dtype=bool),
        world_ids=tuple(f"world-{index}" for index in range(5)),
        probes=(Probe("held-out", 0.2, 0.5, 4, 0.03), Probe("held-out", 0.8, 0.5, 4, 0.03)),
        metadata={"search_complete": search_complete},
    )
    return worlds, labels


def test_groups_identical_codes_and_distinct_perceptual_equivalents() -> None:
    worlds, labels = fixture()

    report = group_equivalent_worlds(worlds, labels, tolerance=1e-8)

    assert [item.world_indices for item in report.classes] == [(0, 1), (2, 3), (4,)]
    assert [item.status for item in report.classes] == [
        "identifiable",
        "perceptually_equivalent",
        "identifiable",
    ]
    assert report.classes[0].unique_code_count == 1
    assert report.classes[1].unique_code_count == 2
    assert parameter_scoring_mask(report).tolist() == [True, True, False, False, True]


def test_exact_code_and_possibility_equivalence_are_scored_separately() -> None:
    worlds, labels = fixture()
    report = group_equivalent_worlds(worlds, labels, tolerance=1e-8)

    identical = report.compare(true_index=0, predicted_index=1)
    equivalent = report.compare(true_index=2, predicted_index=3)
    different = report.compare(true_index=2, predicted_index=4)

    assert identical.exact_code is True and identical.possibility_equivalent is True
    assert equivalent.exact_code is False and equivalent.possibility_equivalent is True
    assert different.exact_code is False and different.possibility_equivalent is False


def test_incomplete_search_marks_groups_unresolved_and_disables_parameter_scoring() -> None:
    worlds, labels = fixture(search_complete=False)

    report = group_equivalent_worlds(worlds, labels, tolerance=1e-8)

    assert {item.status for item in report.classes} == {"unresolved"}
    assert not parameter_scoring_mask(report).any()


def test_raster_overlap_is_retained_as_secondary_diagnostic() -> None:
    worlds, labels = fixture()
    rasters = np.zeros((5, 4, 4), dtype=bool)
    rasters[0, 1, 1] = True
    rasters[1, 1, 1] = True
    rasters[2, 2, 1] = True
    rasters[3, 2, 1:3] = True
    rasters[4, 3, 3] = True

    report = group_equivalent_worlds(
        worlds, labels, tolerance=1e-8, possibility_rasters=rasters
    )

    assert report.raster_iou[0, 1] == 1.0
    assert report.raster_iou[2, 3] == 0.5
    assert report.raster_iou[2, 4] == 0.0

