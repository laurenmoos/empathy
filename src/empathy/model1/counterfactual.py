"""Private perception runner used by Model 2's activation imagination loop."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from empathy.model2.activation_adapter import Model1ActivationPacket
from empathy.model2.imagination import CounterfactualUnavailable
from empathy.physics.config import RenderConfig, SimulationConfig
from empathy.physics.events import extract_reappearance
from empathy.physics.render import Occluder, render_sequence
from empathy.physics.simulate import BallState, rollout
from empathy.physics.world import WorldCode

from .config import Model1Config
from .data import collate_model1, make_model1_record
from .model import Model1
from .online import StepwiseWorldModel
from .online_train import ObservedSeeingEpisode, observed_states_from_path


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


class FrozenOnlineSeeingCounterfactualRunner:
    """Run candidate clips through one frozen online Seeing checkpoint."""

    def __init__(
        self,
        model: StepwiseWorldModel,
        world_latent: torch.Tensor,
        config: Model1Config,
        episodes: tuple[CounterfactualEpisode, ...],
        *,
        checkpoint_id: str,
    ) -> None:
        if not episodes or not checkpoint_id:
            raise ValueError("online counterfactual runner needs episodes and a checkpoint")
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.world_latent = world_latent.detach().clone()
        self.config = config
        self.episodes = episodes
        self.checkpoint_id = checkpoint_id

    @classmethod
    def from_observed(
        cls,
        model: StepwiseWorldModel,
        world_latent: torch.Tensor,
        config: Model1Config,
        events: tuple[tuple[CounterfactualEpisode, ObservedSeeingEpisode], ...],
        *,
        checkpoint_id: str,
    ) -> tuple[FrozenOnlineSeeingCounterfactualRunner, tuple[Model1ActivationPacket, ...]]:
        runner = cls(
            model,
            world_latent,
            config,
            tuple(event for event, _ in events),
            checkpoint_id=checkpoint_id,
        )
        observed = tuple(
            runner._packet(event, episode)
            for event, episode in events
        )
        return runner, observed

    def _activation(
        self,
        frames: torch.Tensor,
        mask: torch.Tensor,
        state: torch.Tensor,
    ) -> np.ndarray:
        with torch.inference_mode():
            belief = self.model.initial_belief(
                frames,
                mask,
                state,
                self.world_latent.expand(frames.shape[0], -1),
            )
        return torch.cat((belief.mean, belief.log_scale), dim=-1)[0].cpu().numpy()

    def _activations(
        self,
        items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> list[np.ndarray]:
        if not items:
            return []
        max_steps = max(frames.shape[1] for frames, _, _ in items)
        _, _, channels, height, width = items[0][0].shape
        frames = torch.zeros(
            (len(items), max_steps, channels, height, width), dtype=torch.float32
        )
        mask = torch.zeros((len(items), max_steps), dtype=torch.bool)
        states = torch.cat([state for _, _, state in items], dim=0)
        for index, (item_frames, item_mask, _) in enumerate(items):
            steps = item_frames.shape[1]
            frames[index, :steps] = item_frames[0]
            mask[index, :steps] = item_mask[0]
        with torch.inference_mode():
            belief = self.model.initial_belief(
                frames,
                mask,
                states,
                self.world_latent.expand(len(items), -1),
            )
        values = torch.cat((belief.mean, belief.log_scale), dim=-1).cpu().numpy()
        return [value for value in values]

    def _packet(
        self,
        event: CounterfactualEpisode,
        episode: ObservedSeeingEpisode,
    ) -> Model1ActivationPacket:
        return Model1ActivationPacket(
            event.episode_id,
            event.event_ordinal,
            self.checkpoint_id,
            self._activation(episode.pre_frames, episode.pre_mask, episode.entry_state),
            self._activation(episode.post_frames, episode.post_mask, episode.exit_state),
        )

    def _observed_episode(
        self,
        event: CounterfactualEpisode,
        world: WorldCode,
    ) -> ObservedSeeingEpisode:
        simulation = SimulationConfig(dt=0.1, steps=40, seed=event.event_ordinal)
        trajectory = rollout(event.initial_state, world, simulation)
        if not trajectory.valid:
            raise CounterfactualUnavailable("candidate trajectory left the world")
        occluder_values = self.config.dataset.occluder
        sequence = render_sequence(
            trajectory,
            (Occluder(**occluder_values),),
            RenderConfig(
                width=self.config.dataset.frame_size,
                height=self.config.dataset.frame_size,
                ball_radius=2,
                min_hidden_steps=2,
            ),
        )
        reappearance = extract_reappearance(sequence)
        if reappearance is None:
            raise CounterfactualUnavailable("candidate produced no comparable reappearance")

        def clip(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
            values = torch.from_numpy(
                sequence.frames[start:end].astype(np.float32) / 255.0
            )[None, :, None]
            return values, torch.ones((1, values.shape[1]), dtype=torch.bool)

        pre_start = max(0, reappearance.hidden_start - self.config.dataset.clip_steps)
        post_end = min(len(sequence.frames), reappearance.step + 8)
        pre_frames, pre_mask = clip(pre_start, reappearance.hidden_start)
        post_frames, post_mask = clip(reappearance.step, post_end)
        path = torch.tensor(
            [(state.x, state.y) for state in trajectory.states], dtype=torch.float32
        )
        try:
            entry_state, exit_state, departure = observed_states_from_path(
                path,
                hidden_start=reappearance.hidden_start,
                reappearance_step=reappearance.step,
                dt=simulation.dt,
                left_edge=float(occluder_values["x_min"]),
            )
        except ValueError as error:
            raise CounterfactualUnavailable(
                "candidate reappearance is not comparable"
            ) from error
        return ObservedSeeingEpisode(
            pre_frames,
            pre_mask,
            post_frames,
            post_mask,
            entry_state,
            exit_state,
            departure,
            hidden_steps=reappearance.step - reappearance.hidden_start + 1,
            dt=simulation.dt,
        )

    def activations_for(
        self, world: WorldCode
    ) -> tuple[Model1ActivationPacket, ...]:
        return tuple(
            self._packet(event, self._observed_episode(event, world))
            for event in self.episodes
        )

    def activations_for_many(
        self, worlds: tuple[WorldCode, ...]
    ) -> tuple[tuple[Model1ActivationPacket, ...] | None, ...]:
        valid: list[tuple[int, CounterfactualEpisode, ObservedSeeingEpisode]] = []
        unavailable: set[int] = set()
        for world_index, world in enumerate(worlds):
            try:
                episodes = tuple(
                    (event, self._observed_episode(event, world))
                    for event in self.episodes
                )
            except CounterfactualUnavailable:
                unavailable.add(world_index)
                continue
            valid.extend((world_index, event, episode) for event, episode in episodes)

        pre = self._activations(
            [
                (episode.pre_frames, episode.pre_mask, episode.entry_state)
                for _, _, episode in valid
            ]
        )
        post = self._activations(
            [
                (episode.post_frames, episode.post_mask, episode.exit_state)
                for _, _, episode in valid
            ]
        )
        packets: list[list[Model1ActivationPacket]] = [[] for _ in worlds]
        for (world_index, event, _), pre_value, post_value in zip(
            valid, pre, post, strict=True
        ):
            packets[world_index].append(
                Model1ActivationPacket(
                    event.episode_id,
                    event.event_ordinal,
                    self.checkpoint_id,
                    pre_value,
                    post_value,
                )
            )
        return tuple(
            None if index in unavailable else tuple(world_packets)
            for index, world_packets in enumerate(packets)
        )

