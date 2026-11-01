"""Private perception runner used by Model 2's activation imagination loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from empathy.model2.activation_adapter import Model1ActivationPacket
from empathy.model2.imagination import CounterfactualUnavailable
from empathy.physics.simulate import BallState
from empathy.physics.world import WorldCode

from .config import Model1Config
from .data import collate_model1, make_model1_record
from .model import Model1


@dataclass(frozen=True)
class CounterfactualEpisode:
    episode_id: str
    event_ordinal: int
    initial_state: BallState

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("counterfactual episode id must be nonempty")
        if type(self.event_ordinal) is not int or self.event_ordinal < 0:
            raise ValueError("counterfactual event ordinal must be nonnegative")


class FrozenModel1CounterfactualRunner:
    """Own candidate clips privately and expose only detached Model 1 traces."""

    def __init__(
        self,
        model: Model1,
        config: Model1Config,
        episodes: tuple[CounterfactualEpisode, ...],
        *,
        checkpoint_id: str,
    ) -> None:
        if not episodes:
            raise ValueError("counterfactual runner needs at least one episode")
        if not checkpoint_id:
            raise ValueError("Model 1 checkpoint id must be nonempty")
        self.model = model.eval().requires_grad_(False)
        self.config = config
        self.episodes = episodes
        self.checkpoint_id = checkpoint_id

    def activations_for(
        self, world: WorldCode
    ) -> tuple[Model1ActivationPacket, ...]:
        packets: list[Model1ActivationPacket] = []
        for episode in self.episodes:
            record, reason = make_model1_record(
                self.config,
                world,
                episode.initial_state,
                example_id=f"counterfactual-{episode.episode_id}",
                split="counterfactual",
                family_level="custom",
                seed=episode.event_ordinal,
            )
            if record is None:
                raise CounterfactualUnavailable(
                    f"candidate world produced no comparable event: {reason}"
                )
            batch = collate_model1([record], self.config)
            with torch.inference_mode():
                pre = self.model.forward_with_trace(
                    batch.pre_frames, batch.pre_times, batch.pre_mask
                ).activation[0]
                post = self.model.forward_with_trace(
                    batch.post_frames, batch.post_times, batch.post_mask
                ).activation[0]
            packets.append(
                Model1ActivationPacket(
                    episode.episode_id,
                    episode.event_ordinal,
                    self.checkpoint_id,
                    pre.cpu().numpy(),
                    post.cpu().numpy(),
                )
            )
        return tuple(packets)

