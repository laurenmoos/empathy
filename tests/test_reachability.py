import numpy as np

from perception_env.physics.config import RenderConfig, SimulationConfig
from perception_env.physics.world import WorldSchema
from perception_env.reasoning.reachability import (
    EntryStateGrid,
    Probe,
    enumerate_reachable,
    rasterize_endpoints,
)


def test_enumerates_hand_computable_endpoints_and_probe_labels() -> None:
    grid = EntryStateGrid(
        x_bounds=(0.2, 0.2),
        y_bounds=(0.5, 0.5),
        vx_bounds=(0.1, 0.2),
        vy_bounds=(0.0, 0.0),
        resolution=(1, 1, 2, 1),
    )
    result = enumerate_reachable(
        grid,
        WorldSchema.v1().world(),
        query_steps=(1, 2),
        config=SimulationConfig(dt=1.0, steps=2),
        context_id="clip-a",
    )

    assert [(state.x, state.y) for state in result.endpoints[1]] == [
        (0.3, 0.5),
        (0.4, 0.5),
    ]
    assert [(state.x, state.y) for state in result.endpoints[2]] == [
        (0.4, 0.5),
        (0.6, 0.5),
    ]
    assert result.label(Probe("clip-a", 0.39, 0.5, step=1, tolerance=0.02)) is True
    assert result.label(Probe("clip-a", 0.35, 0.5, step=1, tolerance=0.02)) is False


def test_entry_grid_records_explicit_bounds_and_resolution() -> None:
    grid = EntryStateGrid(
        x_bounds=(0.1, 0.3),
        y_bounds=(0.4, 0.4),
        vx_bounds=(-0.1, 0.1),
        vy_bounds=(0.0, 0.0),
        resolution=(3, 1, 3, 1),
    )

    assert len(grid.states()) == 9
    assert grid.to_metadata()["resolution"] == [3, 1, 3, 1]
    assert grid.to_metadata()["vx_bounds"] == [-0.1, 0.1]


def test_rasterizes_binary_endpoint_occupancy_per_query_step() -> None:
    grid = EntryStateGrid(
        x_bounds=(0.0, 1.0),
        y_bounds=(0.5, 0.5),
        vx_bounds=(0.0, 0.0),
        vy_bounds=(0.0, 0.0),
        resolution=(3, 1, 1, 1),
    )
    result = enumerate_reachable(
        grid,
        WorldSchema.v1().world(),
        query_steps=(0,),
        config=SimulationConfig(dt=0.1, steps=1),
    )

    occupancy = rasterize_endpoints(result, RenderConfig(width=5, height=5, ball_radius=1))

    assert occupancy.shape == (1, 5, 5)
    assert occupancy.dtype == np.bool_
    assert occupancy.sum() == 3
    assert occupancy[0, 2].tolist() == [True, False, True, False, True]


def test_empty_reachable_set_rasterizes_to_zeros() -> None:
    grid = EntryStateGrid(
        x_bounds=(0.99, 0.99),
        y_bounds=(0.5, 0.5),
        vx_bounds=(1.0, 1.0),
        vy_bounds=(0.0, 0.0),
        resolution=(1, 1, 1, 1),
    )
    result = enumerate_reachable(
        grid,
        WorldSchema.v1().world(),
        query_steps=(1,),
        config=SimulationConfig(dt=0.1, steps=1),
    )

    occupancy = rasterize_endpoints(result, RenderConfig(width=8, height=8))

    assert not occupancy.any()


def test_query_endpoint_does_not_depend_on_later_rollout_invalidity() -> None:
    grid = EntryStateGrid(
        x_bounds=(0.8, 0.8),
        y_bounds=(0.5, 0.5),
        vx_bounds=(0.1, 0.1),
        vy_bounds=(0.0, 0.0),
        resolution=(1, 1, 1, 1),
    )
    world = WorldSchema.v1().world()

    short = enumerate_reachable(
        grid, world, (1,), SimulationConfig(dt=1.0, steps=1)
    )
    long = enumerate_reachable(
        grid, world, (1,), SimulationConfig(dt=1.0, steps=3)
    )

    assert short.endpoints[1] == long.endpoints[1]
    assert short.endpoints[1][0].x == 0.9

