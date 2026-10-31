"""Detached Model 1 activation traces and their perceptual belief decoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, eq=False)
class Model1ActivationPacket:
    episode_id: str
    event_ordinal: int
    checkpoint_id: str
    pre_activation: np.ndarray
    post_activation: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be nonempty")
        if type(self.event_ordinal) is not int or self.event_ordinal < 0:
            raise ValueError("event ordinal must be a nonnegative integer")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("checkpoint_id must be nonempty")
        for name in ("pre_activation", "post_activation"):
            value = np.array(getattr(self, name), dtype=np.float32, copy=True)
            if value.ndim != 1 or value.size < 1:
                raise ValueError("activations must be nonempty vectors")
            if not np.isfinite(value).all():
                raise ValueError("activations must be finite")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.pre_activation.shape != self.post_activation.shape:
            raise ValueError("pre and post activations must have the same shape")

    @property
    def activation_difference(self) -> np.ndarray:
        value = self.post_activation - self.pre_activation
        value.flags.writeable = False
        return value


@dataclass(frozen=True)
class ActivationAdapterConfig:
    input_size: int
    height: int
    width: int
    hidden_size: int = 64

    def __post_init__(self) -> None:
        if min(self.input_size, self.height, self.width, self.hidden_size) < 1:
            raise ValueError("adapter dimensions must be positive")
        if min(self.height, self.width) < 8:
            raise ValueError("belief dimensions must be at least eight")


@dataclass(frozen=True)
class ActivationAdapterTrainingConfig:
    epochs: int = 100
    learning_rate: float = 1.0e-3
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0 or self.seed < 0:
            raise ValueError("adapter training values must be positive")


@dataclass(frozen=True)
class ActivationAdapterTrainingResult:
    checkpoint: Path
    initial_loss: float
    final_loss: float


class ActivationBeliefAdapter(nn.Module):
    """Decode pre/post perceptual states with one shared observation map."""

    def __init__(self, config: ActivationAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.height * config.width),
        )

    def _decode(self, activation: torch.Tensor) -> torch.Tensor:
        if (
            activation.ndim != 2
            or activation.shape[1] != self.config.input_size
            or not activation.is_floating_point()
        ):
            raise ValueError("activation must be floating [batch, input_size]")
        activation = activation.detach()
        if not torch.isfinite(activation).all():
            raise ValueError("activation must be finite")
        return torch.sigmoid(self.decoder(activation)).reshape(
            activation.shape[0], self.config.height, self.config.width
        )

    def forward(
        self, pre_activation: torch.Tensor, post_activation: torch.Tensor
    ) -> torch.Tensor:
        if pre_activation.shape != post_activation.shape:
            raise ValueError("pre and post activation shapes must match")
        pre = self._decode(pre_activation)
        post = self._decode(post_activation)
        return torch.stack((pre, post, post - pre), dim=1)


def activation_reconstruction_loss(
    beliefs: torch.Tensor,
    pre_target: torch.Tensor,
    post_target: torch.Tensor,
) -> torch.Tensor:
    if beliefs.ndim != 4 or beliefs.shape[1] != 3:
        raise ValueError("beliefs must have shape [batch, 3, height, width]")
    if pre_target.shape != beliefs[:, 0].shape or post_target.shape != beliefs[:, 1].shape:
        raise ValueError("belief targets must match adapter output")
    if not torch.isfinite(pre_target).all() or not torch.isfinite(post_target).all():
        raise ValueError("belief targets must be finite")
    return F.mse_loss(beliefs[:, 0], pre_target) + F.mse_loss(
        beliefs[:, 1], post_target
    )


def activation_candidate_ranking_loss(
    beliefs: torch.Tensor,
    positive_candidate: torch.Tensor,
    negative_candidates: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    if beliefs.ndim != 4 or beliefs.shape[1] != 3:
        raise ValueError("beliefs must have shape [batch, 3, height, width]")
    if positive_candidate.shape != beliefs[:, 1].shape:
        raise ValueError("positive candidate must match the post-belief shape")
    if (
        negative_candidates.ndim != 4
        or negative_candidates.shape[0] != beliefs.shape[0]
        or negative_candidates.shape[2:] != beliefs.shape[2:]
        or negative_candidates.shape[1] < 1
    ):
        raise ValueError("negative candidates must have shape [batch, count, height, width]")
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("ranking margin must be finite and positive")
    post = beliefs[:, 1]
    positive_distance = (post - positive_candidate).square().mean(dim=(-2, -1))
    negative_distance = (post[:, None] - negative_candidates).square().mean(
        dim=(-2, -1)
    )
    return F.relu(margin + positive_distance[:, None] - negative_distance).mean()


def save_activation_adapter(
    adapter: ActivationBeliefAdapter, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "activation-belief-adapter-v1",
            "config": asdict(adapter.config),
            "state_dict": adapter.state_dict(),
        },
        path,
    )
    return path


def load_activation_adapter(path: str | Path) -> ActivationBeliefAdapter:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "activation-belief-adapter-v1":
        raise ValueError("unsupported activation adapter checkpoint")
    adapter = ActivationBeliefAdapter(ActivationAdapterConfig(**payload["config"]))
    adapter.load_state_dict(payload["state_dict"])
    return adapter.eval()


def fit_activation_adapter(
    event_paths: list[Path] | tuple[Path, ...],
    config: ActivationAdapterConfig,
    training: ActivationAdapterTrainingConfig,
    checkpoint: str | Path,
) -> ActivationAdapterTrainingResult:
    if not event_paths:
        raise ValueError("at least one recorded activation event is required")
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "pre_activation",
            "post_activation",
            "pre_belief",
            "post_belief",
        )
    }
    for path in event_paths:
        with np.load(Path(path), allow_pickle=False) as payload:
            for name in arrays:
                if name not in payload:
                    raise ValueError(f"recorded event is missing {name}")
                arrays[name].append(np.asarray(payload[name], dtype=np.float32))
    stacked = {name: np.stack(values) for name, values in arrays.items()}
    if stacked["pre_activation"].shape[1:] != (config.input_size,) or stacked[
        "post_activation"
    ].shape[1:] != (config.input_size,):
        raise ValueError("recorded activation size does not match adapter")
    belief_shape = (config.height, config.width)
    if stacked["pre_belief"].shape[1:] != belief_shape or stacked[
        "post_belief"
    ].shape[1:] != belief_shape:
        raise ValueError("recorded belief shape does not match adapter")
    if not all(np.isfinite(values).all() for values in stacked.values()):
        raise ValueError("recorded adapter data must be finite")

    torch.manual_seed(training.seed)
    adapter = ActivationBeliefAdapter(config)
    pre = torch.from_numpy(stacked["pre_activation"])
    post = torch.from_numpy(stacked["post_activation"])
    pre_target = torch.from_numpy(stacked["pre_belief"])
    post_target = torch.from_numpy(stacked["post_belief"])
    optimizer = torch.optim.Adam(adapter.parameters(), lr=training.learning_rate)

    def measure() -> torch.Tensor:
        return activation_reconstruction_loss(
            adapter(pre, post), pre_target, post_target
        )

    initial_loss = float(measure().detach())
    adapter.train()
    for _ in range(training.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = measure()
        loss.backward()
        optimizer.step()
    adapter.eval()
    final_loss = float(measure().detach())
    saved = save_activation_adapter(adapter, checkpoint)
    return ActivationAdapterTrainingResult(saved, initial_loss, final_loss)
