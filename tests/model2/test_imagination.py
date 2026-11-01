from __future__ import annotations

import importlib
import importlib.util
import inspect

import numpy as np
import pytest

from empathy.model2 import smc
from empathy.model2.activation_adapter import Model1ActivationPacket
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
                "counterfactual",
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
            "observed",
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

