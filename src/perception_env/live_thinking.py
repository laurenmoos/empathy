"""Continuous offset-varied episodes with visible belief-only thought steps."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from .live_inference import (
    LiveInferenceEngine,
    LiveInferenceError,
    LiveParameters,
    _encoded_u8,
    _operator_payload,
)
from .model1.data import collate_model1
from .model1.rasterize import rasterize_mixture
from .model1.types import TrajectoryMixture
from .model2.adapters import Model1BeliefPacket, adapt_model1_events
from .model2.coherence import ConstraintRecovery, evaluate_constraint_recovery
from .model2.contracts import BeliefInputs, WorldPrediction
from .model2.decode import decode_prediction
from .model2.losses import world_code_loss
from .model2.smc import BeliefWorldSMC, SMCConfig, SMCResult
from .model2.train import predict as predict_model2
from .simulate import BallState
from .world import WorldSchema


class Model2Runtime(Protocol):
    weight_update_count: int

    def predict(self, inputs: BeliefInputs) -> WorldPrediction: ...

    def complete_event(self, inputs: BeliefInputs) -> dict[str, float | int]: ...


class OnlineModel2Runtime:
    """Serialize belief-only prediction and one-step online adaptation."""

    def __init__(self, model, *, learning_rate: float = 1.0e-4) -> None:
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.0
        )
        self.weight_update_count = 0
        self._lock = threading.RLock()

    def predict(self, inputs: BeliefInputs) -> WorldPrediction:
        with self._lock:
            return predict_model2(self.model, inputs)

    def fork(self) -> OnlineModel2Runtime:
        """Start an independent adaptation stream from the loaded checkpoint state."""
        with self._lock:
            return OnlineModel2Runtime(copy.deepcopy(self.model).eval())

    def train_step(
        self, inputs: BeliefInputs, target_world_flat: np.ndarray
    ) -> dict[str, float | int]:
        with self._lock:
            device = next(self.model.parameters()).device
            beliefs = torch.tensor(inputs.beliefs, device=device)
            history_mask = torch.tensor(inputs.history_mask, device=device)
            targets = torch.tensor(target_world_flat[None], dtype=torch.float32, device=device)
            identifiable = torch.ones((1, 7), dtype=torch.bool, device=device)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(beliefs, history_mask)
            loss = world_code_loss(prediction, targets, identifiable)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.model.eval()
            self.weight_update_count += 1
            return {
                "loss": float(loss.total.detach().cpu()),
                "activity_loss": float(loss.activity.detach().cpu()),
                "parameter_loss": float(loss.parameters.detach().cpu()),
                "weight_update_count": self.weight_update_count,
            }

    def complete_event(self, inputs: BeliefInputs) -> dict[str, float | int]:
        del inputs
        return {"loss": 0.0, "weight_update_count": self.weight_update_count}


class SMCModel2Runtime:
    """Expose the README's belief-only particle search through the live runtime."""

    def __init__(self, *, seed: int = 7, config: SMCConfig | None = None) -> None:
        self.config = config or SMCConfig(
            particle_count=12,
            rounds=1,
            seed=seed,
            grid_size=6,
            sinkhorn_iterations=10,
        )
        self.weight_update_count = 0
        self._last_result: SMCResult | None = None

    def fork(self) -> SMCModel2Runtime:
        return SMCModel2Runtime(config=self.config)

    def infer(self, inputs: BeliefInputs) -> SMCResult:
        self._last_result = BeliefWorldSMC(self.config).infer(inputs)
        return self._last_result

    def predict(self, inputs: BeliefInputs) -> WorldPrediction:
        flat = self.infer(inputs).best_world
        activity_indices = (0, 2, 4, 6, 9)
        parameter_indices = (1, 3, 5, 7, 8, 10, 11)
        activity = torch.tensor(flat[list(activity_indices)], dtype=torch.float32)[None]
        parameters = torch.tensor(flat[list(parameter_indices)], dtype=torch.float32)[None]
        logits = torch.where(activity.bool(), torch.tensor(10.0), torch.tensor(-10.0))
        return WorldPrediction(logits, parameters)

    def complete_event(self, inputs: BeliefInputs) -> dict[str, float | int]:
        if self._last_result is None:
            self.infer(inputs)
        assert self._last_result is not None
        return {
            "loss": float(self._last_result.event_discrepancies[0].mean()),
            "weight_update_count": self.weight_update_count,
        }


@dataclass(frozen=True, eq=False)
class LiveThoughtStep:
    index: int
    label: str
    pre_belief: np.ndarray
    belief: np.ndarray
    guessed_flat: np.ndarray
    guessed_parameters: np.ndarray
    paths: np.ndarray
    path_weights: np.ndarray
    constraint_recovery: ConstraintRecovery

    def to_payload(self) -> dict[str, object]:
        height, width = self.belief.shape
        return {
            "index": self.index,
            "label": self.label,
            "width": width,
            "height": height,
            "pre_belief_u8": _encoded_u8(
                np.clip(np.rint(self.pre_belief * 255), 0, 255)
            ),
            "belief_u8": _encoded_u8(np.clip(np.rint(self.belief * 255), 0, 255)),
            "predicted_world_flat": self.guessed_flat.tolist(),
            "predicted_parameters": self.guessed_parameters.tolist(),
            "paths": self.paths.tolist(),
            "path_weights": self.path_weights.tolist(),
            "operators": _operator_payload(self.guessed_flat),
            "constraint_recovery": self.constraint_recovery.to_payload(),
        }


@dataclass(frozen=True, eq=False)
class LiveThinkingEpisode:
    episode_index: int
    initial_state: BallState
    frames: np.ndarray
    occluder: tuple[float, float, float, float]
    hidden_start: int
    hidden_end: int
    reappearance_step: int
    thought_steps: tuple[LiveThoughtStep, ...]
    ground_truth_flat: np.ndarray
    ground_truth_path: np.ndarray
    training: dict[str, float | int]
    latent_update_count: int
    weight_update_count: int

    def to_payload(self) -> dict[str, object]:
        height, width = self.frames.shape[1:]
        return {
            "schema_version": "live-thinking-v1",
            "episode_index": self.episode_index,
            "initial_state": [
                self.initial_state.x,
                self.initial_state.y,
                self.initial_state.vx,
                self.initial_state.vy,
            ],
            "ground_truth": {
                "width": width,
                "height": height,
                "frame_count": len(self.frames),
                "hidden_start": self.hidden_start,
                "hidden_end": self.hidden_end,
                "reappearance_step": self.reappearance_step,
                "occluder": list(self.occluder),
                "world_flat": self.ground_truth_flat.tolist(),
                "path": self.ground_truth_path.tolist(),
                "operators": _operator_payload(self.ground_truth_flat),
            },
            "thought_steps": [step.to_payload() for step in self.thought_steps],
            "training": dict(self.training),
            "latent_update_count": self.latent_update_count,
            "weight_update_count": self.weight_update_count,
        }


def _subset(prediction: TrajectoryMixture, count: int) -> TrajectoryMixture:
    count = min(max(count, 1), prediction.positions.shape[1])
    indices = torch.topk(prediction.logits, count, dim=1).indices
    position_indices = indices[:, :, None, None].expand(
        -1, -1, prediction.positions.shape[2], prediction.positions.shape[3]
    )
    return TrajectoryMixture(
        positions=torch.gather(prediction.positions, 1, position_indices),
        logits=torch.gather(prediction.logits, 1, indices),
        entry_state=prediction.entry_state,
    )


DEFAULT_INITIAL_STATES = (
    BallState(0.1, 0.42, 0.18, 0.00),
    BallState(0.1, 0.50, 0.16, 0.02),
    BallState(0.1, 0.58, 0.20, -0.02),
    BallState(0.1, 0.46, 0.20, 0.03),
    BallState(0.1, 0.54, 0.17, -0.03),
    BallState(0.1, 0.38, 0.19, 0.01),
    BallState(0.1, 0.62, 0.18, -0.01),
    BallState(0.1, 0.48, 0.21, 0.00),
)


class LiveThinkingSession:
    """Generate episodes in one fixed world and adapt Model 2 after each guess."""

    def __init__(
        self,
        model1,
        model1_config,
        model2_runtime: Model2Runtime,
        parameters: LiveParameters,
        *,
        initial_states: tuple[BallState, ...] = DEFAULT_INITIAL_STATES,
        max_offset_attempts: int = 8,
        model_lock: threading.RLock | None = None,
    ) -> None:
        if not initial_states or max_offset_attempts < 1:
            raise ValueError("thinking session requires offsets and a positive attempt bound")
        self.model1 = model1
        self.model1_config = model1_config
        self.model2_runtime = model2_runtime
        self.parameters = parameters
        self.initial_states = initial_states
        self.max_offset_attempts = max_offset_attempts
        self._model_lock = model_lock or threading.RLock()
        self._session_lock = threading.Lock()
        self._history: list[Model1BeliefPacket] = []
        self._offset_cursor = 0
        self._episode_index = 0
        self._completed_events = 0
        self._latent_updates = 0

    @property
    def completed_event_count(self) -> int:
        return self._completed_events

    def _next_record(self, engine: LiveInferenceEngine):
        for _ in range(self.max_offset_attempts):
            initial = self.initial_states[self._offset_cursor % len(self.initial_states)]
            self._offset_cursor += 1
            try:
                record, frames, full_path = engine._record(self.parameters, initial)
            except LiveInferenceError:
                continue
            return initial, record, frames, full_path
        raise LiveInferenceError("no valid offset produced an occluded reappearance")

    def next_episode(self) -> LiveThinkingEpisode:
        with self._session_lock, self._model_lock:
            engine = LiveInferenceEngine(
                self.model1,
                self.model1_config,
                self.model2_runtime.predict,
            )
            initial, record, frames, full_path = self._next_record(engine)
            batch = collate_model1([record], self.model1_config)
            size = self.model1_config.dataset.frame_size
            with torch.inference_mode():
                pre_prediction = self.model1(batch.pre_frames, batch.pre_times, batch.pre_mask)
                post_prediction = self.model1(batch.post_frames, batch.post_times, batch.post_mask)
                mode_count = pre_prediction.positions.shape[1]
                counts = (1, min(2, mode_count), mode_count)
                visual_predictions = [
                    _subset(pre_prediction, count) for count in counts
                ]
                visual_predictions.append(post_prediction)
                beliefs = [
                    rasterize_mixture(
                        prediction,
                        height=size,
                        width=size,
                        step_mask=batch.target_mask,
                    ).collapsed[0].cpu().numpy().astype(np.float32)
                    for prediction in visual_predictions
                ]

            labels = (
                "first possibility",
                "more possibilities",
                "full belief",
                "reappearance revision",
            )
            previous = np.zeros_like(beliefs[0])
            steps: list[LiveThoughtStep] = []
            final_inputs: BeliefInputs | None = None
            for index, (label, belief, visual_prediction) in enumerate(
                zip(labels, beliefs, visual_predictions, strict=True)
            ):
                packet = Model1BeliefPacket(
                    episode_id="live-thinking",
                    event_ordinal=self._episode_index,
                    pre_belief=previous,
                    post_belief=belief,
                )
                final_inputs = adapt_model1_events([*self._history, packet])
                prediction = self.model2_runtime.predict(final_inputs)
                guessed_parameters = (
                    prediction.normalized_parameters[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                guessed = decode_prediction(prediction, WorldSchema.v1()).flat[0].astype(
                    np.float32
                )
                visual_paths = (
                    visual_prediction.positions[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                entry_position = full_path[record.hidden_start]
                entries = np.broadcast_to(
                    entry_position, (len(visual_paths), 1, 2)
                ).copy()
                visual_paths = np.concatenate((entries, visual_paths), axis=1)
                steps.append(
                    LiveThoughtStep(
                        index,
                        label,
                        previous.copy(),
                        belief.copy(),
                        guessed,
                        guessed_parameters,
                        visual_paths,
                        torch.softmax(visual_prediction.logits[0], dim=0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32),
                        evaluate_constraint_recovery(record.world_flat, guessed),
                    )
                )
                previous = belief
            assert final_inputs is not None
            training = self.model2_runtime.complete_event(final_inputs)
            final_packet = Model1BeliefPacket(
                episode_id="live-thinking",
                event_ordinal=self._episode_index,
                pre_belief=beliefs[-2],
                post_belief=beliefs[-1],
            )
            self._history = [*self._history, final_packet][-3:]
            self._latent_updates += len(steps)
            result = LiveThinkingEpisode(
                episode_index=self._episode_index,
                initial_state=initial,
                frames=frames,
                occluder=tuple(
                    float(self.model1_config.dataset.occluder[name])
                    for name in ("x_min", "x_max", "y_min", "y_max")
                ),
                hidden_start=record.hidden_start,
                hidden_end=record.hidden_end,
                reappearance_step=record.reappearance_step,
                thought_steps=tuple(steps),
                ground_truth_flat=record.world_flat,
                ground_truth_path=full_path,
                training=training,
                latent_update_count=self._latent_updates,
                weight_update_count=self.model2_runtime.weight_update_count,
            )
            self._episode_index += 1
            self._completed_events += 1
            return result
