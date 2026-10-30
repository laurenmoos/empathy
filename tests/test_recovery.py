import numpy as np
import pytest

from perception_env.config import SimulationConfig
from perception_env.reachability import EntryStateGrid, Probe
from perception_env.recovery import EvaluationContext, build_matched_possibility_bank
from perception_env.world import WorldSchema


def _context() -> EvaluationContext:
    return EvaluationContext(
        context_id="shared",
        entry_grid=EntryStateGrid(
            (0.1, 0.1),
            (0.5, 0.5),
            (0.2, 0.2),
            (0.0, 0.0),
            (1, 1, 1, 1),
        ),
        simulation=SimulationConfig(dt=0.5, steps=2, seed=3),
        query_steps=(1, 2),
    )


def test_matched_bank_uses_identical_event_and_probe_axes() -> None:
    world = WorldSchema.v1().world()
    probes = (
        Probe("shared", 0.2, 0.5, 1, 1.0e-9),
        Probe("shared", 0.3, 0.5, 2, 1.0e-9),
    )

    bank = build_matched_possibility_bank(
        (world, world),
        (world, world),
        (_context(),),
        probes,
    )

    assert bank.true_labels.shape == (2, 2)
    assert bank.predicted_labels.shape == (2, 2)
    assert bank.event_ids == ("event-0", "event-1")
    assert bank.probe_ids == (
        "shared:1:0.2:0.5:1e-09",
        "shared:2:0.3:0.5:1e-09",
    )
    assert bank.valid_events.tolist() == [True, True]
    assert np.array_equal(bank.true_labels, bank.predicted_labels)
    assert bank.metadata["contexts"]["shared"]["query_steps"] == [1, 2]


def test_decode_failure_is_retained_as_an_invalid_event() -> None:
    world = WorldSchema.v1().world()
    probe = Probe("shared", 0.2, 0.5, 1, 1.0e-9)

    bank = build_matched_possibility_bank(
        (world, world),
        (world, None),
        (_context(),),
        (probe,),
    )

    assert bank.valid_events.tolist() == [True, False]
    assert bank.predicted_labels.tolist() == [[True], [False]]
    assert bank.recovered.valid.tolist() == [True, False]
    assert bank.recovered.worlds[1] is None


def test_possibility_equivalent_worlds_receive_identical_labels() -> None:
    schema = WorldSchema.v1()
    true_world = schema.world()
    different_but_equivalent = schema.world(
        vertical_acceleration={"a_y": 0.1}
    )
    probe = Probe("shared", 0.2, 0.5, 1, 0.1)

    bank = build_matched_possibility_bank(
        (true_world,),
        (different_but_equivalent,),
        (_context(),),
        (probe,),
    )

    assert not np.array_equal(
        true_world.to_flat(schema), different_but_equivalent.to_flat(schema)
    )
    assert bank.true_labels.tolist() == [[True]]
    assert bank.predicted_labels.tolist() == [[True]]


def test_evaluation_api_persists_no_first_order_arrays() -> None:
    world = WorldSchema.v1().world()
    bank = build_matched_possibility_bank(
        (world,),
        (world,),
        (_context(),),
        (Probe("shared", 0.2, 0.5, 1, 1.0e-9),),
    )

    forbidden = {"frames", "beliefs", "visible", "positions"}
    assert forbidden.isdisjoint(vars(bank))
    assert forbidden.isdisjoint(bank.metadata)


def test_matched_bank_rejects_unknown_probe_context() -> None:
    world = WorldSchema.v1().world()

    with pytest.raises(ValueError, match="context"):
        build_matched_possibility_bank(
            (world,),
            (world,),
            (_context(),),
            (Probe("missing", 0.2, 0.5, 1, 0.01),),
        )


def test_matched_bank_rejects_query_step_outside_context() -> None:
    world = WorldSchema.v1().world()

    with pytest.raises(ValueError, match="query step"):
        build_matched_possibility_bank(
            (world,),
            (world,),
            (_context(),),
            (Probe("shared", 0.4, 0.5, 3, 0.01),),
        )


def test_probe_ids_include_tolerance_and_metadata_preserves_definitions() -> None:
    world = WorldSchema.v1().world()
    probes = (
        Probe("shared", 0.2, 0.5, 1, 0.01),
        Probe("shared", 0.2, 0.5, 1, 0.02),
    )

    bank = build_matched_possibility_bank(
        (world,),
        (world,),
        (_context(),),
        probes,
    )

    assert bank.probe_ids == (
        "shared:1:0.2:0.5:0.01",
        "shared:1:0.2:0.5:0.02",
    )
    assert bank.metadata["probes"] == [
        {"context_id": "shared", "x": 0.2, "y": 0.5, "step": 1, "tolerance": 0.01},
        {"context_id": "shared", "x": 0.2, "y": 0.5, "step": 1, "tolerance": 0.02},
    ]


def test_matched_bank_rejects_duplicate_probe_definitions() -> None:
    world = WorldSchema.v1().world()
    probe = Probe("shared", 0.2, 0.5, 1, 0.01)

    with pytest.raises(ValueError, match="probe identifiers must be unique"):
        build_matched_possibility_bank(
            (world,),
            (world,),
            (_context(),),
            (probe, probe),
        )


def test_matched_bank_rejects_empty_probe_bank() -> None:
    world = WorldSchema.v1().world()

    with pytest.raises(ValueError, match="at least one probe"):
        build_matched_possibility_bank(
            (world,),
            (world,),
            (_context(),),
            (),
        )
