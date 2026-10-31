import inspect

import numpy as np

from perception_env.model2.contracts import BeliefInputs
from perception_env.model2.smc import BeliefWorldSMC, SMCConfig
from perception_env.world import WorldSchema


def _inputs(post: np.ndarray | None = None, pre: np.ndarray | None = None) -> BeliefInputs:
    size = 32
    beliefs = np.zeros((1, 4, 3, size, size), dtype=np.float32)
    if post is not None:
        beliefs[0, -1, 1] = post
    if pre is not None:
        beliefs[0, -1, 0] = pre
    beliefs[0, -1, 2] = beliefs[0, -1, 1] - beliefs[0, -1, 0]
    return BeliefInputs(beliefs, np.array([[False, False, False, True]]))


def test_smc_inference_boundary_accepts_beliefs_only() -> None:
    assert tuple(inspect.signature(BeliefWorldSMC.infer).parameters) == ("self", "inputs")


def test_empty_beliefs_produce_finite_normalized_deterministic_posterior() -> None:
    config = SMCConfig(particle_count=18, rounds=2, seed=17)

    first = BeliefWorldSMC(config).infer(_inputs())
    second = BeliefWorldSMC(config).infer(_inputs())

    assert len(first.ranked_worlds) == 18
    assert np.isfinite(first.weights).all()
    assert np.isclose(first.weights.sum(), 1.0)
    assert np.array_equal(first.ranked_worlds, second.ranked_worlds)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.best_world, first.ranked_worlds[0])
    assert first.event_discrepancies.shape == (18, 1)


def test_curved_belief_ranks_downward_gravity_above_neutral_world() -> None:
    size = 32
    pre = np.zeros((size, size), dtype=np.float32)
    post = np.zeros_like(pre)
    for offset, column in enumerate(range(11, 20)):
        pre[12, column] = 1.0
        post[12 + round((offset / 8) ** 2 * 3), column] = 1.0

    result = BeliefWorldSMC(
        SMCConfig(particle_count=32, rounds=2, seed=5)
    ).infer(_inputs(post, pre))
    schema = WorldSchema.v1()
    worlds = [schema.from_flat(row) for row in result.ranked_worlds]
    gravity_weights = [
        weight
        for world, weight in zip(worlds, result.weights, strict=True)
        if world.slot("vertical_acceleration").active
        and world.slot("vertical_acceleration").parameters[0] < 0
    ]
    neutral_weights = [
        weight
        for world, weight in zip(worlds, result.weights, strict=True)
        if not world.slot("vertical_acceleration").active
    ]

    assert result.best_world[0] == 1.0
    assert schema.from_flat(result.best_world).slot("vertical_acceleration").parameters[0] < 0
    assert max(gravity_weights) > max(neutral_weights)
