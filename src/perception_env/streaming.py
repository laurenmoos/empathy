"""Belief-only contracts and bounded feedback for continual world inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .beliefs import BeliefEvent, build_belief_history
from .config import RenderConfig, SimulationConfig
from .events import extract_reappearance
from .records import belief_entropy
from .render import Occluder, render_sequence
from .simulate import BallState, rollout
from .world import WorldCode, WorldSchema


def _array(values: Any, dtype: Any) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, eq=False)
class OcclusionEpisode:
    event_id: str
    context_id: str
    frames: np.ndarray
    visible: np.ndarray
    reappearance_step: int
    reappearance_xy: tuple[float, float]
    evaluation: dict[str, Any]

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        visible = np.asarray(self.visible)
        if frames.ndim not in {3, 4} or visible.shape != (len(frames),):
            raise ValueError("frames and visibility must share a time axis")
        if visible.dtype != np.bool_:
            raise ValueError("visibility must be boolean")
        if not 0 <= self.reappearance_step < len(frames) or not visible[self.reappearance_step]:
            raise ValueError("reappearance must name a visible frame")
        if not all(0.0 <= value <= 1.0 for value in self.reappearance_xy):
            raise ValueError("reappearance coordinates must lie in [0,1]")
        object.__setattr__(self, "frames", _array(frames, np.float32))
        object.__setattr__(self, "visible", _array(visible, bool))


@dataclass(frozen=True, eq=False)
class Model2Features:
    pre_belief: np.ndarray
    post_belief: np.ndarray
    belief_difference: np.ndarray
    entropy: np.ndarray
    history_mask: np.ndarray

    def __post_init__(self) -> None:
        pre = np.asarray(self.pre_belief)
        post = np.asarray(self.post_belief)
        difference = np.asarray(self.belief_difference)
        entropy = np.asarray(self.entropy)
        mask = np.asarray(self.history_mask)
        if pre.ndim != 3 or pre.shape[0] != 4:
            raise ValueError("belief fields must have shape [4,height,width]")
        if post.shape != pre.shape or difference.shape != pre.shape:
            raise ValueError("belief fields must share a shape")
        if entropy.shape != (4,) or mask.shape != (4,) or mask.dtype != np.bool_:
            raise ValueError("entropy and history_mask must have length four")
        if not np.array_equal(difference, post - pre):
            raise ValueError("belief_difference must equal post_belief - pre_belief")
        if not np.allclose(entropy, belief_entropy(pre)):
            raise ValueError("entropy must be derived from pre_belief")
        if (np.diff(mask.astype(int)) < 0).any() or not mask.any():
            raise ValueError("history_mask must be a nonempty valid suffix")
        if not all(np.isfinite(item).all() for item in (pre, post, difference, entropy)):
            raise ValueError("belief features must be finite")
        object.__setattr__(self, "pre_belief", _array(pre, np.float32))
        object.__setattr__(self, "post_belief", _array(post, np.float32))
        object.__setattr__(self, "belief_difference", _array(difference, np.float32))
        object.__setattr__(self, "entropy", _array(entropy, np.float32))
        object.__setattr__(self, "history_mask", _array(mask, bool))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Model2Features:
        allowed = {
            "pre_belief",
            "post_belief",
            "belief_difference",
            "entropy",
            "history_mask",
        }
        if set(values) != allowed:
            raise ValueError("Model 2 feature fields must be exactly belief-derived fields")
        return cls(**{name: values[name] for name in allowed})

    @classmethod
    def from_events(cls, events: Sequence[BeliefEvent], length: int = 4) -> Model2Features:
        history = build_belief_history(events, length=length)
        if length != 4:
            raise ValueError("Model 2 history length is fixed at four")
        return cls(
            history.tensor[:, 0],
            history.tensor[:, 1],
            history.tensor[:, 2],
            belief_entropy(history.tensor[:, 0]),
            history.mask,
        )

    @property
    def stacked(self) -> np.ndarray:
        return np.stack((self.pre_belief, self.post_belief, self.belief_difference), axis=1)


@dataclass(frozen=True, eq=False)
class StreamWorldPrediction:
    flat_world: np.ndarray
    activity_probabilities: np.ndarray
    parameter_values: np.ndarray
    confidence: float

    def __post_init__(self) -> None:
        schema = WorldSchema.v1()
        flat = np.asarray(self.flat_world, dtype=float)
        schema.from_flat(flat)
        activity = np.asarray(self.activity_probabilities, dtype=float)
        parameters = np.asarray(self.parameter_values, dtype=float)
        if activity.shape != (5,) or parameters.shape != (7,):
            raise ValueError("prediction vectors do not match world schema")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0,1]")
        object.__setattr__(self, "flat_world", _array(flat, np.float64))
        object.__setattr__(self, "activity_probabilities", _array(activity, np.float32))
        object.__setattr__(self, "parameter_values", _array(parameters, np.float32))


class BeliefRuntime(Protocol):
    def propose(
        self, episode: OcclusionEpisode, world_prior: np.ndarray | None
    ) -> np.ndarray: ...

    def assimilate(
        self, episode: OcclusionEpisode, pre_belief: np.ndarray
    ) -> np.ndarray: ...


class WorldRuntime(Protocol):
    def predict(self, features: Model2Features) -> StreamWorldPrediction: ...


@dataclass(frozen=True)
class FeedbackIteration:
    index: int
    prediction: StreamWorldPrediction
    covered: bool


@dataclass(frozen=True)
class StreamUpdate:
    event_id: str
    status: str
    event: BeliefEvent
    iterations: tuple[FeedbackIteration, ...]


@dataclass(frozen=True, eq=False)
class StreamingSessionState:
    world_flat: np.ndarray
    events: tuple[BeliefEvent, ...]
    processed_event_ids: tuple[str, ...]
    latent_update_count: int
    weight_update_count: int

    def to_mapping(self) -> dict[str, Any]:
        return {
            "world_flat": self.world_flat.tolist(),
            "events": [
                {"step": event.step, "pre": event.pre.tolist(), "post": event.post.tolist()}
                for event in self.events
            ],
            "processed_event_ids": list(self.processed_event_ids),
            "latent_update_count": self.latent_update_count,
            "weight_update_count": self.weight_update_count,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> StreamingSessionState:
        return cls(
            _array(values["world_flat"], np.float64),
            tuple(
                BeliefEvent(int(item["step"]), np.asarray(item["pre"]), np.asarray(item["post"]))
                for item in values["events"]
            ),
            tuple(str(item) for item in values["processed_event_ids"]),
            int(values["latent_update_count"]),
            int(values["weight_update_count"]),
        )


class StreamingSession:
    def __init__(
        self,
        belief_runtime: BeliefRuntime,
        world_runtime: WorldRuntime,
        possibility: Callable[[WorldCode, OcclusionEpisode], np.ndarray],
        *,
        initial_world_flat: np.ndarray,
        max_feedback_iterations: int,
        outcome_tolerance_pixels: int,
    ) -> None:
        if max_feedback_iterations < 1 or outcome_tolerance_pixels < 0:
            raise ValueError("feedback bounds must be nonnegative and nonzero")
        self.belief_runtime = belief_runtime
        self.world_runtime = world_runtime
        self.possibility = possibility
        self.max_feedback_iterations = max_feedback_iterations
        self.outcome_tolerance_pixels = outcome_tolerance_pixels
        self._world_flat = _array(initial_world_flat, np.float64)
        WorldSchema.v1().from_flat(self._world_flat)
        self._events: list[BeliefEvent] = []
        self._processed_ids: list[str] = []
        self._latent_updates = 0
        self._weight_updates = 0

    def _covered(self, image: np.ndarray, episode: OcclusionEpisode) -> bool:
        height, width = image.shape
        column = int(round(episode.reappearance_xy[0] * (width - 1)))
        row = int(round((1.0 - episode.reappearance_xy[1]) * (height - 1)))
        radius = self.outcome_tolerance_pixels
        return bool(
            np.asarray(image, dtype=bool)[
                max(0, row - radius) : min(height, row + radius + 1),
                max(0, column - radius) : min(width, column + radius + 1),
            ].any()
        )

    def process(self, episode: OcclusionEpisode) -> StreamUpdate:
        if episode.event_id in self._processed_ids:
            raise ValueError("event already processed")
        iterations: list[FeedbackIteration] = []
        prior: np.ndarray | None = None
        final_event: BeliefEvent | None = None
        status = "unresolved"
        for index in range(self.max_feedback_iterations):
            pre = np.asarray(self.belief_runtime.propose(episode, prior), dtype=np.float32)
            post = np.asarray(self.belief_runtime.assimilate(episode, pre), dtype=np.float32)
            final_event = BeliefEvent(episode.reappearance_step, pre, post)
            features = Model2Features.from_events([*self._events, final_event])
            prediction = self.world_runtime.predict(features)
            world = WorldSchema.v1().from_flat(prediction.flat_world)
            covered = self._covered(self.possibility(world, episode), episode)
            iterations.append(FeedbackIteration(index, prediction, covered))
            self._world_flat = prediction.flat_world
            self._latent_updates += 1
            if covered:
                status = "covered"
                break
            prior = prediction.flat_world
        assert final_event is not None
        self._events = [*self._events, final_event][-4:]
        self._processed_ids.append(episode.event_id)
        return StreamUpdate(episode.event_id, status, final_event, tuple(iterations))

    def snapshot(self) -> StreamingSessionState:
        return StreamingSessionState(
            self._world_flat,
            tuple(self._events),
            tuple(self._processed_ids),
            self._latent_updates,
            self._weight_updates,
        )


def generate_stream(
    world: WorldCode,
    initial_states: Sequence[BallState],
    occluders: Sequence[Occluder],
    simulation: SimulationConfig,
    render: RenderConfig,
    stream_id: str,
) -> tuple[OcclusionEpisode, ...]:
    """Generate accepted occlusion episodes under one persistent physical world."""
    schema = WorldSchema.v1()
    schema.validate(world)
    world_flat = world.to_flat(schema).tolist()
    episodes: list[OcclusionEpisode] = []
    for initial in initial_states:
        trajectory = rollout(initial, world, simulation)
        sequence = render_sequence(trajectory, tuple(occluders), render)
        event = extract_reappearance(sequence)
        if event is None:
            continue
        index = len(episodes)
        episodes.append(
            OcclusionEpisode(
                event_id=f"{stream_id}-{index:03d}",
                context_id=f"{stream_id}-context-{index:03d}",
                frames=sequence.frames.astype(np.float32) / 255.0,
                visible=sequence.visible,
                reappearance_step=event.step,
                reappearance_xy=(event.x, event.y),
                evaluation={
                    "true_world_flat": world_flat,
                    "initial_state": [initial.x, initial.y, initial.vx, initial.vy],
                    "seed": simulation.seed,
                },
            )
        )
    return tuple(episodes)

