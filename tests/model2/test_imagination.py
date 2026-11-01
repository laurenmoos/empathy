from __future__ import annotations

import importlib
import importlib.util
import inspect

import numpy as np
import pytest

from empathy.model2 import smc
from empathy.model2.activation_adapter import Model1ActivationPacket
from empathy.model2.imagination import ActivationImaginationLoop, CounterfactualUnavailable
from empathy.model2.smc import SMCConfig
from empathy.physics.world import WorldCode, WorldSchema


class ParameterActivationSource:
    def __init__(self) -> None:
        self.calls = 0

    def activations_for(self, world: WorldCode) -> tuple[Model1ActivationPacket, ...]:
        self.calls += 1
        gravity = world.slot("vertical_acceleration")
        wind = world.slot("horizontal_acceleration")
        damping = world.slot("damping")
        activation = np.asarray(
            [
                gravity.parameters[0] if gravity.active else 0.0,
                wind.parameters[0] if wind.active else 0.0,
                damping.parameters[0] if damping.active else 0.0,
            ],
            dtype=np.float32,
        )
        return (
            Model1ActivationPacket(
                "episode-0",
                0,
                "frozen-model-1",
                activation,
                activation * 2.0,
            ),
        )


def test_activation_smc_boundary_accepts_only_the_imagination_loop() -> None:
    assert importlib.util.find_spec("empathy.model2.imagination") is not None
    assert hasattr(smc, "ActivationWorldSMC")
    assert tuple(inspect.signature(smc.ActivationWorldSMC.infer).parameters) == (
        "self",
        "imagination",
    )


def test_particles_are_weighted_by_counterfactual_model1_activations() -> None:
    imagination_module = importlib.import_module("empathy.model2.imagination")
    activation_world_smc = smc.ActivationWorldSMC
    observed = (
        Model1ActivationPacket(
            "episode-0",
            0,
            "frozen-model-1",
            np.asarray([-0.30, 0.0, 0.0], dtype=np.float32),
            np.asarray([-0.60, 0.0, 0.0], dtype=np.float32),
        ),
    )
    source = ParameterActivationSource()
    imagination = imagination_module.ActivationImaginationLoop(observed, source)
    result = activation_world_smc(
        SMCConfig(particle_count=6, rounds=1, seed=3)
    ).infer(imagination)

    best = WorldSchema.v1().from_flat(result.best_world)
    assert best.slot("vertical_acceleration").parameters[0] == pytest.approx(-0.3)
    assert np.isclose(result.weights.sum(), 1.0)
    assert np.isfinite(result.event_discrepancies).all()
    assert source.calls == 6


def test_imagination_rejects_activation_shaped_objects_with_raw_frames() -> None:
    class RawActivationCarrier:
        checkpoint_id = "frozen-model-1"
        pre_activation = np.zeros(3, dtype=np.float32)
        post_activation = np.zeros(3, dtype=np.float32)
        frames = np.zeros((2, 8, 8), dtype=np.uint8)

    with pytest.raises(TypeError, match="Model1ActivationPacket"):
        ActivationImaginationLoop((RawActivationCarrier(),), ParameterActivationSource())

    class PacketWithFrames(Model1ActivationPacket):
        frames = np.zeros((2, 8, 8), dtype=np.uint8)

    smuggled = PacketWithFrames(
        "episode-0",
        0,
        "frozen-model-1",
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    with pytest.raises(TypeError, match="Model1ActivationPacket"):
        ActivationImaginationLoop((smuggled,), ParameterActivationSource())


def test_counterfactual_events_must_match_observed_episode_identity() -> None:
    observed = (
        Model1ActivationPacket(
            "actual-episode",
            3,
            "frozen-model-1",
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        ),
    )
    imagination = ActivationImaginationLoop(observed, ParameterActivationSource())
    discrepancy = imagination.discrepancies((WorldSchema.v1().world(),))
    assert np.isinf(discrepancy[0, 0])


def test_all_unavailable_counterfactuals_do_not_return_a_posterior() -> None:
    observed = (
        Model1ActivationPacket(
            "actual-episode",
            3,
            "frozen-model-1",
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        ),
    )
    imagination = ActivationImaginationLoop(observed, ParameterActivationSource())
    with pytest.raises(CounterfactualUnavailable, match="no candidate"):
        smc.ActivationWorldSMC(
            SMCConfig(particle_count=6, rounds=1, seed=3)
        ).infer(imagination)


def test_rejuvenation_rounds_do_not_reapply_the_same_activation_evidence() -> None:
    observed = (
        Model1ActivationPacket(
            "episode-0",
            0,
            "frozen-model-1",
            np.asarray([-0.30, 0.0, 0.0], dtype=np.float32),
            np.asarray([-0.60, 0.0, 0.0], dtype=np.float32),
        ),
    )
    one = smc.ActivationWorldSMC(
        SMCConfig(particle_count=6, rounds=1, seed=3)
    ).infer(ActivationImaginationLoop(observed, ParameterActivationSource()))
    four = smc.ActivationWorldSMC(
        SMCConfig(particle_count=6, rounds=4, seed=3)
    ).infer(ActivationImaginationLoop(observed, ParameterActivationSource()))

    assert np.array_equal(one.ranked_worlds, four.ranked_worlds)
    assert np.array_equal(one.weights, four.weights)

