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


class PartiallyUnavailableSource(ParameterActivationSource):
    def activations_for(self, world: WorldCode) -> tuple[Model1ActivationPacket, ...]:
        gravity = world.slot("vertical_acceleration")
        if not gravity.active or gravity.parameters[0] >= 0:
            raise CounterfactualUnavailable("candidate has no comparable event")
        return super().activations_for(world)


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


def test_imagination_uses_batched_counterfactual_model_calls_when_available() -> None:
    observed = (
        Model1ActivationPacket(
            "episode-0",
            0,
            "frozen-model-1",
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )

    class BatchSource(ParameterActivationSource):
        def activations_for(self, world):
            raise AssertionError("single-world path should not run")

        def activations_for_many(self, worlds):
            return tuple(
                ParameterActivationSource.activations_for(self, world)
                for world in worlds
            )

    source = BatchSource()
    worlds = (WorldSchema.v1().world(), WorldSchema.v1().world())

    discrepancies = ActivationImaginationLoop(observed, source).discrepancies(worlds)

    assert discrepancies.shape == (2, 1)
    assert source.calls == 2


def test_continual_activation_search_retains_its_posterior_across_checkpoints() -> None:
    observed = (
        Model1ActivationPacket(
            "episode-0",
            0,
            "frozen-model-1",
            np.asarray([-0.20, 0.10, 0.20], dtype=np.float32),
            np.asarray([-0.40, 0.20, 0.40], dtype=np.float32),
        ),
    )
    search = smc.ContinualActivationWorldSMC(
        SMCConfig(particle_count=8, rounds=1, seed=19)
    )

    first = search.update(ActivationImaginationLoop(observed, ParameterActivationSource()))
    second = search.update(ActivationImaginationLoop(observed, ParameterActivationSource()))

    assert first.update_count == 1
    assert second.update_count == 2
    assert search.initialization_count == 1
    assert second.parameter_particles.shape == (8, 3)
    assert np.isfinite(second.weights).all()
    assert second.weights.sum() == pytest.approx(1.0)


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


def test_search_rounds_report_activation_progress_and_parameter_estimates() -> None:
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
    four = smc.ActivationWorldSMC(
        SMCConfig(particle_count=6, rounds=4, seed=3)
    ).infer(ActivationImaginationLoop(observed, source))

    assert [step.round_index for step in four.progress] == [0, 1, 2, 3]
    assert all(step.best_parameters.shape == (3,) for step in four.progress)
    assert all(step.posterior_mean.shape == (3,) for step in four.progress)
    assert all(np.isfinite(step.best_discrepancy) for step in four.progress)
    assert all(np.isfinite(step.effective_sample_size) for step in four.progress)
    assert all(step.accepted_revisions >= 0 for step in four.progress)
    best = [step.best_discrepancy for step in four.progress]
    assert all(
        later <= earlier for earlier, later in zip(best, best[1:], strict=False)
    )
    assert source.calls == 6 * 4


def test_search_handles_mixed_valid_and_invalid_counterfactuals_without_nan() -> None:
    observed = (
        Model1ActivationPacket(
            "episode-0",
            0,
            "frozen-model-1",
            np.asarray([-0.30, 0.0, 0.0], dtype=np.float32),
            np.asarray([-0.60, 0.0, 0.0], dtype=np.float32),
        ),
    )
    with np.errstate(invalid="raise"):
        result = smc.ActivationWorldSMC(
            SMCConfig(particle_count=6, rounds=2, seed=3)
        ).infer(ActivationImaginationLoop(observed, PartiallyUnavailableSource()))
    assert np.isfinite(result.weights).all()

