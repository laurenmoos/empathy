from itertools import product

import numpy as np

from perception_env.reachability import Probe
from perception_env.shattering import LabelMatrix, search_shattered_set
from perception_env.visualize import render_shattering_matrix


def test_shattering_matrix_groups_rows_and_highlights_selected_probes(tmp_path) -> None:
    values = np.asarray(
        [list(bits) for bits in product([0, 1], repeat=3)] + [[0, 0, 0]],
        dtype=bool,
    )
    labels = LabelMatrix(
        values=values,
        world_ids=tuple(f"world-{index}" for index in range(len(values))),
        probes=tuple(Probe("clip-a", index * 0.1, 0.5, 4, 0.03) for index in range(3)),
        metadata={"operator_support": "fixture", "uncertainty_grid": "fixture", "tolerance": 0.03},
    )
    estimate = search_shattered_set(labels, max_n=3, budget=100, seed=4)

    artifact = render_shattering_matrix(estimate, tmp_path / "shattering.png")

    assert artifact.path.stat().st_size > 0
    assert artifact.metadata["dimension"] == 3
    assert artifact.metadata["selected_probe_indices"] == [0, 1, 2]
    assert artifact.metadata["distinct_labelings"] == 8
    assert artifact.metadata["world_count"] == 9
    assert artifact.metadata["budget"] == 100
