"""Live local inference from user-selected toy-world parameters."""

from __future__ import annotations

import base64
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import RenderConfig, SimulationConfig
from .events import extract_reappearance
from .model1.config import Model1Config
from .model1.data import Model1ExampleRecord, collate_model1
from .model1.rasterize import rasterize_mixture
from .model1.train import load_checkpoint
from .model2.adapters import Model1BeliefPacket, adapt_model1_events
from .model2.contracts import BeliefInputs, WorldPrediction
from .model2.decode import decode_prediction
from .render import Occluder, render_sequence
from .simulate import BallState, rollout
from .world import WorldCode, WorldSchema


class LiveInferenceError(ValueError):
    """A requested world cannot produce a valid learned inference event."""


DEFAULT_LIVE_INITIAL_STATE = BallState(0.1, 0.5, 0.18, 0.0)


@dataclass(frozen=True)
class LiveParameters:
    gravity: float = 0.0
    wind: float = 0.0
    damping: float = 0.0

    def __post_init__(self) -> None:
        values = (self.gravity, self.wind, self.damping)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            for value in values
        ):
            raise ValueError("live parameters must be numbers, not booleans")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("live parameters must be finite")
        if not -1.0 <= self.gravity <= 1.0 or not -1.0 <= self.wind <= 1.0:
            raise ValueError("gravity and wind must lie in [-1, 1]")
        if not 0.0 <= self.damping <= 1.0:
            raise ValueError("damping must lie in [0, 1]")


def world_from_parameters(parameters: LiveParameters) -> WorldCode:
    """Map physical UI values to a canonical v1 world code."""
    active: dict[str, dict[str, float]] = {}
    if abs(parameters.gravity) > 0.05:
        active["vertical_acceleration"] = {"a_y": parameters.gravity}
    if abs(parameters.wind) > 0.05:
        active["horizontal_acceleration"] = {"a_x": parameters.wind}
    if parameters.damping > 0.01:
        active["damping"] = {"lambda": parameters.damping}
    return WorldSchema.v1().world(**active)


def _encoded_u8(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype=np.uint8).tobytes()).decode("ascii")


def _operator_payload(flat: np.ndarray) -> dict[str, dict[str, object]]:
    schema = WorldSchema.v1()
    world = schema.from_flat(flat)
    result: dict[str, dict[str, object]] = {}
    for spec, slot in zip(schema.specs, world.slots, strict=True):
        result[slot.name] = {
            "active": slot.active,
            "parameters": slot.parameter_dict(spec) if slot.active else {},
        }
    return result


@dataclass(frozen=True, eq=False)
class LiveInferenceResult:
    frames: np.ndarray
    hidden_start: int
    hidden_end: int
    reappearance_step: int
    pre_belief: np.ndarray
    post_belief: np.ndarray
    ground_truth_flat: np.ndarray
    guessed_flat: np.ndarray

    def to_payload(self) -> dict[str, object]:
        height, width = self.frames.shape[1:]
        belief_height, belief_width = self.pre_belief.shape
        return {
            "schema_version": "live-inference-v1",
            "ground_truth": {
                "width": width,
                "height": height,
                "frame_count": len(self.frames),
                "hidden_start": self.hidden_start,
                "hidden_end": self.hidden_end,
                "reappearance_step": self.reappearance_step,
                "frames_base64": _encoded_u8(self.frames),
                "world_flat": self.ground_truth_flat.tolist(),
                "operators": _operator_payload(self.ground_truth_flat),
            },
            "model1": {
                "width": belief_width,
                "height": belief_height,
                "pre_belief_u8": _encoded_u8(
                    np.clip(np.rint(self.pre_belief * 255), 0, 255)
                ),
                "post_belief_u8": _encoded_u8(
                    np.clip(np.rint(self.post_belief * 255), 0, 255)
                ),
            },
            "model2": {
                "predicted_world_flat": self.guessed_flat.tolist(),
                "operators": _operator_payload(self.guessed_flat),
            },
        }


class LiveInferenceEngine:
    """Own loaded models and run one deterministic inference request at a time."""

    def __init__(
        self,
        model1,
        model1_config: Model1Config,
        model2_predict: Callable[[BeliefInputs], WorldPrediction],
        *,
        model2_runtime=None,
    ) -> None:
        self.model1 = model1
        self.model1_config = model1_config
        self.model2_predict = model2_predict
        self.model2_runtime = model2_runtime
        self._lock = threading.RLock()

    @classmethod
    def from_checkpoints(cls, model1_checkpoint: Path, model2_checkpoint: Path):
        payload = torch.load(Path(model1_checkpoint), map_location="cpu", weights_only=False)
        config: Model1Config = payload["config"]
        model1, _ = load_checkpoint(model1_checkpoint, expected_config=config, device="cpu")
        del model2_checkpoint
        from .live_thinking import SMCModel2Runtime

        runtime = SMCModel2Runtime()
        return cls(
            model1,
            config,
            runtime.predict,
            model2_runtime=runtime,
        )

    def thinking_session(self, parameters: LiveParameters):
        """Create one fixed-world stream backed by the loaded online Model 2 runtime."""
        if self.model2_runtime is None:
            raise RuntimeError("thinking sessions require an online Model 2 runtime")
        from .live_thinking import LiveThinkingSession

        runtime = (
            self.model2_runtime.fork()
            if hasattr(self.model2_runtime, "fork")
            else self.model2_runtime
        )
        return LiveThinkingSession(
            self.model1,
            self.model1_config,
            runtime,
            parameters,
            model_lock=self._lock,
        )

    def _record(
        self,
        parameters: LiveParameters,
        initial_state: BallState = DEFAULT_LIVE_INITIAL_STATE,
    ) -> tuple[Model1ExampleRecord, np.ndarray]:
        world = world_from_parameters(parameters)
        simulation = SimulationConfig(dt=0.1, steps=40, seed=0)
        trajectory = rollout(initial_state, world, simulation)
        if not trajectory.valid:
            raise LiveInferenceError(
                f"world trajectory is invalid: {trajectory.termination_reason.value}"
            )
        dataset = self.model1_config.dataset
        render_config = RenderConfig(
            width=dataset.frame_size,
            height=dataset.frame_size,
            ball_radius=2,
            min_hidden_steps=2,
        )
        sequence = render_sequence(
            trajectory,
            (Occluder(**dataset.occluder),),
            render_config,
        )
        event = extract_reappearance(sequence)
        if event is None:
            reason = sequence.rejection_reason or "no visible reappearance"
            raise LiveInferenceError(f"world has no valid reappearance: {reason}")
        horizon = dataset.prediction_steps
        target = np.zeros((horizon, 4), dtype=np.float32)
        target_mask = np.zeros(horizon, dtype=bool)
        available = trajectory.states[event.hidden_start : event.hidden_start + horizon]
        for index, state in enumerate(available):
            target[index] = (state.x, state.y, state.vx, state.vy)
            target_mask[index] = True
        empty_belief = np.zeros((dataset.frame_size, dataset.frame_size), dtype=np.float32)
        record = Model1ExampleRecord(
            example_id="live",
            split="live",
            family_level="custom",
            seed=0,
            world_id="live",
            world_flat=world.to_flat(WorldSchema.v1()).astype(np.float32),
            frames=sequence.frames,
            hidden_start=event.hidden_start,
            hidden_end=event.hidden_end,
            reappearance_step=event.step,
            target_states=target,
            target_mask=target_mask,
            oracle_pre_belief=empty_belief,
            actual_path_belief=empty_belief,
        )
        return record, sequence.frames

    def infer(self, parameters: LiveParameters) -> LiveInferenceResult:
        record, frames = self._record(parameters)
        batch = collate_model1([record], self.model1_config)
        size = self.model1_config.dataset.frame_size
        with self._lock, torch.inference_mode():
            pre_prediction = self.model1(batch.pre_frames, batch.pre_times, batch.pre_mask)
            post_prediction = self.model1(batch.post_frames, batch.post_times, batch.post_mask)
            pre = rasterize_mixture(
                pre_prediction, height=size, width=size, step_mask=batch.target_mask
            ).collapsed[0]
            post = rasterize_mixture(
                post_prediction, height=size, width=size, step_mask=batch.target_mask
            ).collapsed[0]
            pre_array = pre.cpu().numpy().astype(np.float32)
            post_array = post.cpu().numpy().astype(np.float32)
            inputs: BeliefInputs = adapt_model1_events(
                [
                    Model1BeliefPacket(
                        episode_id="live",
                        event_ordinal=record.reappearance_step,
                        pre_belief=pre_array,
                        post_belief=post_array,
                    )
                ]
            )
            prediction = self.model2_predict(inputs)
            guessed = decode_prediction(prediction, WorldSchema.v1()).flat[0].astype(np.float32)
        return LiveInferenceResult(
            frames=frames,
            hidden_start=record.hidden_start,
            hidden_end=record.hidden_end,
            reappearance_step=record.reappearance_step,
            pre_belief=pre_array,
            post_belief=post_array,
            ground_truth_flat=record.world_flat,
            guessed_flat=guessed,
        )
