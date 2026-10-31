from pathlib import Path

import numpy as np
import pytest
import torch

from perception_env.model2.activation_adapter import (
    ActivationAdapterConfig,
    ActivationAdapterTrainingConfig,
    ActivationBeliefAdapter,
    Model1ActivationPacket,
    activation_candidate_ranking_loss,
    activation_reconstruction_loss,
    fit_activation_adapter,
    load_activation_adapter,
    save_activation_adapter,
)


def _packet(**overrides) -> Model1ActivationPacket:
    values = {
        "episode_id": "episode-1",
        "event_ordinal": 0,
        "checkpoint_id": "checkpoint-a",
        "pre_activation": np.linspace(-1.0, 1.0, 6, dtype=np.float32),
        "post_activation": np.linspace(1.0, -1.0, 6, dtype=np.float32),
    }
    values.update(overrides)
    return Model1ActivationPacket(**values)


def test_activation_packet_is_detached_validated_and_immutable() -> None:
    source = np.arange(6, dtype=np.float32)
    packet = _packet(pre_activation=source)
    source[:] = 99

    assert packet.pre_activation.tolist() == list(range(6))
    assert not packet.pre_activation.flags.writeable
    assert np.array_equal(
        packet.activation_difference,
        packet.post_activation - packet.pre_activation,
    )

    with pytest.raises(ValueError, match="same shape"):
        _packet(post_activation=np.zeros(5, np.float32))
    with pytest.raises(ValueError, match="finite"):
        _packet(pre_activation=np.full(6, np.nan, np.float32))
    with pytest.raises(ValueError, match="checkpoint"):
        _packet(checkpoint_id="")


def test_adapter_reconstructs_valid_pre_post_and_difference_beliefs() -> None:
    config = ActivationAdapterConfig(input_size=6, height=8, width=10, hidden_size=12)
    adapter = ActivationBeliefAdapter(config)
    pre = torch.randn(3, 6, requires_grad=True)
    post = torch.randn(3, 6, requires_grad=True)

    beliefs = adapter(pre, post)

    assert beliefs.shape == (3, 3, 8, 10)
    assert torch.all((beliefs[:, :2] >= 0) & (beliefs[:, :2] <= 1))
    assert torch.equal(beliefs[:, 2], beliefs[:, 1] - beliefs[:, 0])
    beliefs.sum().backward()
    assert pre.grad is None
    assert post.grad is None
    assert adapter.decoder[0].weight.grad is not None


def test_reconstruction_loss_can_train_the_adapter() -> None:
    torch.manual_seed(5)
    config = ActivationAdapterConfig(input_size=4, height=8, width=8, hidden_size=16)
    adapter = ActivationBeliefAdapter(config)
    pre = torch.randn(4, 4)
    post = torch.randn(4, 4)
    pre_target = torch.rand(4, 8, 8)
    post_target = torch.rand(4, 8, 8)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.03)

    initial = activation_reconstruction_loss(
        adapter(pre, post), pre_target, post_target
    ).item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = activation_reconstruction_loss(
            adapter(pre, post), pre_target, post_target
        )
        loss.backward()
        optimizer.step()
    final = activation_reconstruction_loss(
        adapter(pre, post), pre_target, post_target
    ).item()

    assert final < initial * 0.5


def test_adapter_checkpoint_replays_exactly(tmp_path: Path) -> None:
    torch.manual_seed(11)
    config = ActivationAdapterConfig(input_size=6, height=8, width=8, hidden_size=10)
    adapter = ActivationBeliefAdapter(config).eval()
    pre = torch.randn(2, 6)
    post = torch.randn(2, 6)
    expected = adapter(pre, post)
    checkpoint = save_activation_adapter(adapter, tmp_path / "adapter.pt")

    loaded = load_activation_adapter(checkpoint)

    assert loaded.config == config
    assert torch.equal(loaded(pre, post), expected)


def test_candidate_ranking_loss_prefers_the_matching_simulated_world() -> None:
    evidence = torch.zeros(1, 3, 8, 8, requires_grad=True)
    positive = torch.zeros(1, 8, 8)
    negatives = torch.ones(1, 2, 8, 8)

    matching = activation_candidate_ranking_loss(
        evidence, positive, negatives, margin=0.1
    )
    mismatching = activation_candidate_ranking_loss(
        evidence, torch.ones_like(positive), torch.zeros_like(negatives), margin=0.1
    )

    assert matching.item() == pytest.approx(0.0)
    assert mismatching.item() > 0.1
    mismatching.backward()
    assert evidence.grad is not None


def test_recorded_stream_events_fit_deterministically(tmp_path: Path) -> None:
    paths = []
    for index in range(4):
        pre = np.full(4, index / 4, np.float32)
        post = np.full(4, (index + 1) / 4, np.float32)
        pre_belief = np.full((8, 8), index / 4, np.float32)
        post_belief = np.full((8, 8), (index + 1) / 4, np.float32)
        path = tmp_path / f"event-{index}.npz"
        np.savez_compressed(
            path,
            pre_activation=pre,
            post_activation=post,
            pre_belief=pre_belief,
            post_belief=post_belief,
        )
        paths.append(path)
    config = ActivationAdapterConfig(input_size=4, height=8, width=8, hidden_size=12)
    training = ActivationAdapterTrainingConfig(epochs=30, learning_rate=0.03, seed=17)

    first = fit_activation_adapter(paths, config, training, tmp_path / "first.pt")
    second = fit_activation_adapter(paths, config, training, tmp_path / "second.pt")

    assert first.final_loss < first.initial_loss
    assert first.initial_loss == pytest.approx(second.initial_loss)
    assert first.final_loss == pytest.approx(second.final_loss)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            load_activation_adapter(first.checkpoint).state_dict().values(),
            load_activation_adapter(second.checkpoint).state_dict().values(),
            strict=True,
        )
    )
