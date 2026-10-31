"""Validated belief-image ablation and activation-derived producer boundaries."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from empathy.reasoning.beliefs import BeliefEvent, build_belief_history

from .activation_adapter import ActivationBeliefAdapter, Model1ActivationPacket
from .contracts import BeliefInputs


@dataclass(frozen=True, eq=False)
class Model1BeliefPacket:
    episode_id: str
    event_ordinal: int
    pre_belief: np.ndarray
    post_belief: np.ndarray

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be nonempty")
        if type(self.event_ordinal) is not int or self.event_ordinal < 0:
            raise ValueError("event ordinal must be a nonnegative integer")
        for name in ("pre_belief", "post_belief"):
            value = np.array(getattr(self, name), dtype=np.float32, copy=True)
            if (
                value.ndim != 2
                or min(value.shape) < 8
                or not np.isfinite(value).all()
                or (value < 0).any()
                or (value > 1).any()
            ):
                raise ValueError("beliefs must be finite [H,W] images in [0,1], H,W>=8")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.pre_belief.shape != self.post_belief.shape:
            raise ValueError("belief shapes must match")


def parse_model1_packet(values: Mapping[str, object]) -> Model1BeliefPacket:
    if set(values) != {"episode_id", "event_ordinal", "pre_belief", "post_belief"}:
        raise ValueError("packet fields must be exactly episode_id, event_ordinal, pre/post_belief")
    return Model1BeliefPacket(**values)


def adapt_model1_events(events: Sequence[Model1BeliefPacket]) -> BeliefInputs:
    if not events or not all(isinstance(event, Model1BeliefPacket) for event in events):
        raise ValueError("a nonempty sequence of validated belief packets is required")
    if len({event.episode_id for event in events}) != 1:
        raise ValueError("events must belong to one episode")
    if any(a.event_ordinal >= b.event_ordinal for a, b in zip(events, events[1:], strict=False)):
        raise ValueError("event ordinals must be strictly increasing")
    if len({event.pre_belief.shape for event in events}) != 1:
        raise ValueError("all event resolutions must match")
    history = build_belief_history(
        [BeliefEvent(e.event_ordinal, e.pre_belief, e.post_belief) for e in events]
    )
    return BeliefInputs(history.tensor[None], history.mask[None])


def adapt_model1_activations(
    events: Sequence[Model1ActivationPacket],
    adapter: ActivationBeliefAdapter,
    *,
    control_mode: str = "full",
    seed: int = 0,
) -> BeliefInputs:
    if not events or not all(isinstance(event, Model1ActivationPacket) for event in events):
        raise ValueError("a nonempty sequence of activation packets is required")
    if not isinstance(adapter, ActivationBeliefAdapter):
        raise TypeError("activation adapter is required")
    if len({event.episode_id for event in events}) != 1:
        raise ValueError("activation events must belong to one episode")
    if len({event.checkpoint_id for event in events}) != 1:
        raise ValueError("activation events must use one Model 1 checkpoint")
    if any(a.event_ordinal >= b.event_ordinal for a, b in zip(events, events[1:], strict=False)):
        raise ValueError("activation event ordinals must be strictly increasing")
    if any(event.pre_activation.size != adapter.config.input_size for event in events):
        raise ValueError("activation size does not match adapter")
    if control_mode not in {"full", "shuffled"}:
        raise ValueError("unknown activation control mode")
    if type(seed) is not int or seed < 0:
        raise ValueError("activation control seed must be a nonnegative integer")

    device = next(adapter.parameters()).device
    pre_values = np.stack([event.pre_activation for event in events])
    post_values = np.stack([event.post_activation for event in events])
    if control_mode == "shuffled":
        permutation = np.random.default_rng(seed).permutation(adapter.config.input_size)
        pre_values = pre_values[:, permutation].copy()
        post_values = post_values[:, permutation].copy()
    pre = torch.from_numpy(pre_values).to(device)
    post = torch.from_numpy(post_values).to(device)
    was_training = adapter.training
    adapter.eval()
    try:
        with torch.inference_mode():
            decoded = adapter(pre, post).cpu().numpy().astype(np.float32)
    finally:
        adapter.train(was_training)
    history = build_belief_history(
        [
            BeliefEvent(event.event_ordinal, belief[0], belief[1])
            for event, belief in zip(events, decoded, strict=True)
        ]
    )
    return BeliefInputs(history.tensor[None], history.mask[None])
