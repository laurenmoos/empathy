import warnings

import numpy as np

from empathy.physics.config import RenderConfig, SimulationConfig
from empathy.physics.world import WorldSchema
from empathy.reasoning.beliefs import BeliefEvent
from empathy.reasoning.reachability import EntryStateGrid, enumerate_reachable
from empathy.reasoning.visualize import render_belief_panel


def make_result(empty: bool):
    grid = EntryStateGrid(
        x_bounds=(0.99, 0.99) if empty else (0.1, 0.9),
        y_bounds=(0.5, 0.5),
        vx_bounds=(1.0, 1.0) if empty else (0.0, 0.0),
        vy_bounds=(0.0, 0.0),
        resolution=(1, 1, 1, 1) if empty else (9, 1, 1, 1),
    )
    return enumerate_reachable(
        grid,
        WorldSchema.v1().world(),
        query_steps=(1 if empty else 0,),
        config=SimulationConfig(dt=0.1, steps=1),
    )


def test_belief_panel_separates_model_channels_from_first_order_diagnostic(tmp_path) -> None:
    result = make_result(empty=False)
    pre = np.zeros((8, 8), dtype=np.float32)
    pre[4, 1:7] = np.linspace(0.1, 0.9, 6)
    post = np.zeros((8, 8), dtype=np.float32)
    post[4, 4:7] = 1.0
    event = BeliefEvent(step=3, pre=pre, post=post)

    artifact = render_belief_panel(
        result,
        event,
        RenderConfig(width=8, height=8),
        tmp_path / "belief.png",
        true_endpoint=(0.8, 0.5),
    )

    assert artifact.path.stat().st_size > 0
    assert artifact.metadata["model2_channels"] == [
        "pre_belief",
        "post_belief",
        "belief_difference",
    ]
    assert artifact.metadata["first_order_excluded"] is True
    assert artifact.metadata["panels"] == [
        "sampled paths",
        "endpoint scatter",
        "binary occupancy",
        "pre belief",
        "post belief",
        "belief difference",
        "first-order diagnostic",
    ]


def test_empty_and_saturated_belief_panels_render_without_warnings(tmp_path) -> None:
    zeros = np.zeros((8, 8), dtype=np.float32)
    ones = np.ones((8, 8), dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        empty = render_belief_panel(
            make_result(empty=True),
            BeliefEvent(1, zeros, zeros),
            RenderConfig(width=8, height=8),
            tmp_path / "empty.png",
        )
        saturated = render_belief_panel(
            make_result(empty=False),
            BeliefEvent(1, ones, ones),
            RenderConfig(width=8, height=8),
            tmp_path / "saturated.png",
        )

    assert empty.path.stat().st_size > 0
    assert saturated.path.stat().st_size > 0
    assert caught == []
