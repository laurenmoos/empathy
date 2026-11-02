import inspect

import numpy as np
import pytest
import torch

from empathy.model2.activation_adapter import (
    ActivationAdapterConfig,
    ActivationBeliefAdapter,
    Model1ActivationPacket,
)
from empathy.model2.adapters import adapt_model1_activations
from empathy.model2.contracts import BeliefInputs
from empathy.model2.smc import (
    BeliefWorldSMC,
    ContinualBeliefWorldSMC,
    SMCConfig,
    weighted_hypothesis_clusters,
    weighted_parameter_summary,
)
from empathy.physics.world import WorldSchema


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


def test_signed_revision_changes_posterior_when_post_belief_is_identical() -> None:
    size = 32
    post = np.zeros((size, size), dtype=np.float32)
    post[12, 11:20] = 1.0

    newly_added = BeliefWorldSMC(SMCConfig(particle_count=18, rounds=2, seed=11)).infer(
        _inputs(post, np.zeros_like(post))
    )
    unchanged = BeliefWorldSMC(SMCConfig(particle_count=18, rounds=2, seed=11)).infer(
        _inputs(post, post)
    )

    assert sorted(row.tobytes() for row in newly_added.ranked_worlds) == sorted(
        row.tobytes() for row in unchanged.ranked_worlds
    )
    assert not np.array_equal(newly_added.weights, unchanged.weights)
    assert not np.array_equal(
        newly_added.event_discrepancies, unchanged.event_discrepancies
    )


def test_single_event_likelihood_is_not_reapplied_by_rejuvenation_rounds() -> None:
    post = np.zeros((32, 32), dtype=np.float32)
    post[12, 11:20] = 1.0

    one_round = BeliefWorldSMC(
        SMCConfig(particle_count=18, rounds=1, seed=13)
    ).infer(_inputs(post))
    four_rounds = BeliefWorldSMC(
        SMCConfig(particle_count=18, rounds=4, seed=13)
    ).infer(_inputs(post))

    assert np.array_equal(one_round.ranked_worlds, four_rounds.ranked_worlds)
    assert np.array_equal(one_round.weights, four_rounds.weights)
    assert np.array_equal(one_round.event_discrepancies, four_rounds.event_discrepancies)


def test_activation_entry_point_reuses_the_belief_smc_search() -> None:
    torch.manual_seed(23)
    adapter = ActivationBeliefAdapter(
        ActivationAdapterConfig(input_size=4, height=16, width=16, hidden_size=8)
    ).eval()
    events = [
        Model1ActivationPacket(
            "episode",
            0,
            "snapshot",
            np.zeros(4, np.float32),
            np.zeros(4, np.float32),
            np.ones(4, np.float32),
        )
    ]
    config = SMCConfig(particle_count=6, rounds=1, seed=29, grid_size=4)
    smc = BeliefWorldSMC(config)

    explicit = smc.infer(adapt_model1_activations(events, adapter))
    activation = smc.infer_activations(events, adapter)

    assert np.array_equal(activation.ranked_worlds, explicit.ranked_worlds)
    assert np.array_equal(activation.weights, explicit.weights)
    assert np.array_equal(activation.event_discrepancies, explicit.event_discrepancies)


def test_continual_smc_retains_one_particle_population_across_updates() -> None:
    post = np.zeros((32, 32), dtype=np.float32)
    post[12, 11:20] = 1.0
    smc = ContinualBeliefWorldSMC(
        SMCConfig(particle_count=18, rounds=1, seed=31)
    )

    first = smc.update(_inputs(post))
    initializations = smc.initialization_count
    second = smc.update(_inputs(post))

    assert first.update_count == 1
    assert second.update_count == 2
    assert initializations == smc.initialization_count == 1
    assert first.parameter_particles.shape == (18, 3)
    assert second.parameter_particles.shape == (18, 3)
    assert np.isfinite(second.weights).all()
    assert second.weights.sum() == pytest.approx(1.0)


def test_weighted_parameter_summary_reports_physical_means_and_intervals() -> None:
    particles = np.array(
        [
            [-0.30, -0.10, 0.00],
            [-0.10, 0.00, 0.20],
            [0.20, 0.30, 0.60],
        ],
        dtype=np.float64,
    )
    weights = np.array([0.15, 0.70, 0.15], dtype=np.float64)

    mean, intervals = weighted_parameter_summary(particles, weights)

    assert mean == pytest.approx([-0.085, 0.03, 0.23])
    assert intervals.shape == (3, 2)
    assert intervals[:, 0] == pytest.approx([-0.30, -0.10, 0.00])
    assert intervals[:, 1] == pytest.approx([0.20, 0.30, 0.60])


def test_weighted_hypothesis_clusters_preserve_separate_joint_worlds() -> None:
    particles = np.array(
        [
            [-0.22, 0.08, 0.10],
            [-0.18, 0.12, 0.14],
            [0.18, -0.12, 0.48],
            [0.22, -0.08, 0.52],
        ],
        dtype=np.float64,
    )
    weights = np.array([0.35, 0.25, 0.20, 0.20], dtype=np.float64)

    clusters = weighted_hypothesis_clusters(particles, weights)

    assert len(clusters) == 2
    assert clusters[0][1] == pytest.approx(0.60)
    assert clusters[0][0] == pytest.approx([-0.203333, 0.096667, 0.116667], abs=1e-5)
    assert clusters[1][1] == pytest.approx(0.40)
    assert clusters[1][0] == pytest.approx([0.20, -0.10, 0.50])


def test_weighted_hypothesis_clusters_ignore_zero_mass_particles() -> None:
    particles = np.array(
        [[-0.2, 0.0, 0.1], [0.2, 0.0, 0.1], [0.0, 0.3, 0.5]],
        dtype=np.float64,
    )
    weights = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    clusters = weighted_hypothesis_clusters(particles, weights)

    assert len(clusters) == 1
    assert clusters[0][0] == pytest.approx(particles[0])
    assert clusters[0][1] == pytest.approx(1.0)


def test_hypothesis_cluster_weight_never_exceeds_probability_bounds() -> None:
    particles = np.zeros((10, 3), dtype=np.float64)
    weights = np.exp(np.linspace(-20.0, 0.0, 10))
    weights /= weights.sum()

    clusters = weighted_hypothesis_clusters(particles, weights)

    assert all(0.0 < weight <= 1.0 for _, weight in clusters)
