"""Continuous offset-varied episodes with visible belief-only thought steps."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from ..model1.counterfactual import (
    CounterfactualEpisode,
    FrozenOnlineSeeingCounterfactualRunner,
)
from ..model1.online import OnlineSeeingConfig, StepwiseWorldModel
from ..model1.online_train import (
    ObservedSeeingEpisode,
    OnlineSeeingRuntime,
    observed_states_from_path,
)
from ..model2.coherence import ConstraintRecovery, evaluate_constraint_recovery
from ..model2.contracts import (
    BeliefInputs,
    ParameterHypothesis,
    ParameterPosterior,
    WorldPrediction,
)
from ..model2.decode import decode_prediction
from ..model2.imagination import ActivationImaginationLoop
from ..model2.losses import world_code_loss
from ..model2.smc import (
    ContinualActivationWorldSMC,
    ContinualBeliefWorldSMC,
    SMCConfig,
    SMCResult,
    weighted_hypothesis_clusters,
    weighted_parameter_summary,
)
from ..model2.train import predict as predict_model2
from ..physics.simulate import BallState
from ..physics.world import WorldSchema
from .inference import (
    LiveInferenceEngine,
    LiveInferenceError,
    LiveParameters,
    _encoded_u8,
    _operator_payload,
)
from .observer import AsyncActivationModel2Observer, Model1ActivationCheckpoint


class Model2Runtime(Protocol):
    weight_update_count: int

    def predict(self, inputs: BeliefInputs) -> WorldPrediction: ...

    def predict_imagination(
        self, imagination: ActivationImaginationLoop
    ) -> WorldPrediction: ...

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
            particle_count=24,
            rounds=1,
            seed=seed,
            grid_size=6,
            sinkhorn_iterations=10,
            score_temperature=8.0,
            jitter_scale=0.015,
        )
        self.weight_update_count = 0
        self._last_result: SMCResult | None = None
        self._search = ContinualBeliefWorldSMC(self.config)
        self._activation_search = ContinualActivationWorldSMC(self.config)

    def fork(self) -> SMCModel2Runtime:
        return SMCModel2Runtime(config=self.config)

    def infer(self, inputs: BeliefInputs) -> SMCResult:
        self._last_result = self._search.update(inputs)
        return self._last_result

    def predict(self, inputs: BeliefInputs) -> WorldPrediction:
        result = self.infer(inputs)
        return self._prediction_from_result(result)

    def predict_imagination(
        self, imagination: ActivationImaginationLoop
    ) -> WorldPrediction:
        self._last_result = self._activation_search.update(imagination)
        return self._prediction_from_result(self._last_result)

    @staticmethod
    def _prediction_from_result(result: SMCResult) -> WorldPrediction:
        flat = result.best_world
        activity_indices = (0, 2, 4, 6, 9)
        parameter_indices = (1, 3, 5, 7, 8, 10, 11)
        activity = torch.tensor(flat[list(activity_indices)], dtype=torch.float32)[None]
        parameters = torch.tensor(flat[list(parameter_indices)], dtype=torch.float32)[None]
        logits = torch.where(activity.bool(), torch.tensor(10.0), torch.tensor(-10.0))
        assert result.parameter_particles is not None
        mean, intervals = weighted_parameter_summary(
            result.parameter_particles, result.weights
        )
        hypotheses = tuple(
            ParameterHypothesis(parameters, weight)
            for parameters, weight in weighted_hypothesis_clusters(
                result.parameter_particles, result.weights
            )
        )
        posterior = ParameterPosterior(
            result.parameter_particles,
            result.weights,
            mean,
            intervals,
            result.update_count,
            hypotheses,
        )
        return WorldPrediction(logits, parameters, posterior)

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
    parameter_posterior: ParameterPosterior | None = None

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
            "parameter_posterior": None
            if self.parameter_posterior is None
            else {
                "particles": self.parameter_posterior.particles.tolist(),
                "weights": self.parameter_posterior.weights.tolist(),
                "mean": self.parameter_posterior.mean.tolist(),
                "intervals": self.parameter_posterior.intervals.tolist(),
                "update_count": self.parameter_posterior.update_count,
                "hypotheses": [
                    {
                        "parameters": hypothesis.parameters.tolist(),
                        "weight": hypothesis.weight,
                    }
                    for hypothesis in self.parameter_posterior.hypotheses
                ],
            },
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
    model1_checkpoint_version: int
    model2_checkpoint_version: int | None

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
            "model1_checkpoint_version": self.model1_checkpoint_version,
            "model2_checkpoint_version": self.model2_checkpoint_version,
        }


DEFAULT_INITIAL_STATES = (
    BallState(0.1, 0.50, 0.18, 0.0),
    BallState(0.1, 0.08, 0.18, 0.0),
    BallState(0.1, 0.92, 0.18, 0.0),
    BallState(0.1, 0.15, 0.18, 0.0),
    BallState(0.1, 0.85, 0.18, 0.0),
    BallState(0.1, 0.22, 0.18, 0.0),
    BallState(0.1, 0.78, 0.18, 0.0),
    BallState(0.1, 0.29, 0.18, 0.0),
    BallState(0.1, 0.71, 0.18, 0.0),
    BallState(0.1, 0.36, 0.18, 0.0),
    BallState(0.1, 0.64, 0.18, 0.0),
    BallState(0.1, 0.43, 0.18, 0.0),
    BallState(0.1, 0.57, 0.18, 0.0),
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
        max_offset_attempts: int = len(DEFAULT_INITIAL_STATES),
        model_lock: threading.RLock | None = None,
        model2_observer: AsyncActivationModel2Observer | None = None,
    ) -> None:
        if not initial_states or max_offset_attempts < 1:
            raise ValueError("thinking session requires offsets and a positive attempt bound")
        self.model1 = model1
        self.model1_config = model1_config
        self.model2_runtime = model2_runtime
        self.parameters = parameters
        self.initial_states = initial_states
        self._adaptive_vertical_sweep = initial_states is DEFAULT_INITIAL_STATES
        self.max_offset_attempts = max_offset_attempts
        self._model_lock = model_lock or threading.RLock()
        self._session_lock = threading.Lock()
        self._offset_cursor = 0
        self._episode_index = 0
        self._completed_events = 0
        self._latent_updates = 0
        seeing_config = OnlineSeeingConfig(
            channels=(4, 8, 12),
            representation_size=24,
            world_size=8,
            sample_count=24,
        )
        occluder = model1_config.dataset.occluder
        self.seeing_runtime = OnlineSeeingRuntime(
            StepwiseWorldModel(seeing_config, hidden_size=32),
            seed=7,
            occluder=(float(occluder["x_min"]), float(occluder["x_max"])),
        )
        self.model2_observer = model2_observer or AsyncActivationModel2Observer(
            model2_runtime.predict_imagination
        )

    @property
    def completed_event_count(self) -> int:
        return self._completed_events

    def _next_record(self, engine: LiveInferenceEngine):
        for _ in range(self.max_offset_attempts):
            initial = self.initial_states[self._offset_cursor % len(self.initial_states)]
            self._offset_cursor += 1
            if self._adaptive_vertical_sweep:
                horizon = 3.9
                vertical_velocity = (
                    (0.5 - initial.y) / horizon
                    - 0.5 * self.parameters.gravity * horizon
                )
                initial = BallState(
                    initial.x,
                    initial.y,
                    initial.vx,
                    vertical_velocity,
                )
            try:
                record, frames, full_path = engine._record(self.parameters, initial)
            except LiveInferenceError:
                continue
            return initial, record, frames, full_path
        raise LiveInferenceError("no valid offset produced an occluded reappearance")

    def _observed_episode(self, record, frames, full_path) -> ObservedSeeingEpisode:
        clip_steps = self.model1_config.dataset.clip_steps
        pre_start = max(0, record.hidden_start - clip_steps)
        post_end = min(len(frames), record.reappearance_step + 8)

        def clip(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
            values = torch.from_numpy(frames[start:end].astype(np.float32) / 255.0)
            values = values[None, :, None]
            return values, torch.ones((1, values.shape[1]), dtype=torch.bool)

        pre_frames, pre_mask = clip(pre_start, record.hidden_start)
        post_frames, post_mask = clip(record.reappearance_step, post_end)
        entry_state, exit_state, departure = observed_states_from_path(
            torch.from_numpy(full_path),
            hidden_start=record.hidden_start,
            reappearance_step=record.reappearance_step,
            dt=0.1,
            left_edge=float(self.model1_config.dataset.occluder["x_min"]),
        )
        return ObservedSeeingEpisode(
            pre_frames,
            pre_mask,
            post_frames,
            post_mask,
            entry_state,
            exit_state,
            departure,
            hidden_steps=record.reappearance_step - record.hidden_start + 1,
        )

    @staticmethod
    def _rasterize_paths(paths: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
        image = np.zeros((size, size), dtype=np.float32)
        normalized = weights / max(float(weights.sum()), 1.0e-8)
        for path, weight in zip(paths, normalized, strict=True):
            columns = np.rint(np.clip(path[:, 0], 0, 1) * (size - 1)).astype(int)
            rows = np.rint((1.0 - np.clip(path[:, 1], 0, 1)) * (size - 1)).astype(int)
            for row, column in zip(rows, columns, strict=True):
                image[max(0, row - 1) : row + 2, max(0, column - 1) : column + 2] += weight
        maximum = float(image.max())
        return image / maximum if maximum > 0 else image

    def next_episode(self) -> LiveThinkingEpisode:
        with self._session_lock, self._model_lock:
            latest_guess = self.model2_observer.latest
            engine = LiveInferenceEngine(
                self.model1,
                self.model1_config,
                self.model2_runtime.predict,
            )
            initial, record, frames, full_path = self._next_record(engine)
            size = self.model1_config.dataset.frame_size
            observed = self._observed_episode(record, frames, full_path)
            frozen = self.seeing_runtime.begin(observed)
            seeing_result = self.seeing_runtime.complete(frozen, observed, train=True)
            all_paths = seeing_result.paths[0].cpu().numpy().astype(np.float32)
            entries = np.broadcast_to(
                observed.entry_state[0, :2].numpy(), (len(all_paths), 1, 2)
            ).copy()
            all_paths = np.concatenate((entries, all_paths), axis=1)
            posterior_weights = seeing_result.posterior_weights[0].cpu().numpy().astype(np.float32)
            counts = (4, 12, len(all_paths), len(all_paths))
            visual_paths = [all_paths[:count] for count in counts]
            visual_weights = [
                np.full(count, 1.0 / count, dtype=np.float32) for count in counts[:-1]
            ]
            visual_weights.append(posterior_weights)
            beliefs = [
                self._rasterize_paths(paths, weights, size)
                for paths, weights in zip(visual_paths, visual_weights, strict=True)
            ]

            labels = (
                "first possibility",
                "more possibilities",
                "full belief",
                "reappearance revision",
            )
            previous = np.zeros_like(beliefs[0])
            steps: list[LiveThoughtStep] = []
            if latest_guess is None:
                guessed_parameters = np.zeros(7, dtype=np.float32)
                guessed = WorldSchema.v1().world().to_flat(WorldSchema.v1()).astype(
                    np.float32
                )
                model2_checkpoint_version = None
                parameter_posterior = None
            else:
                guessed_parameters = (
                    latest_guess.prediction.normalized_parameters[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                guessed = decode_prediction(
                    latest_guess.prediction, WorldSchema.v1()
                ).flat[0].astype(np.float32)
                model2_checkpoint_version = latest_guess.version
                parameter_posterior = latest_guess.prediction.posterior
            for index, (label, belief, step_paths, step_weights) in enumerate(
                zip(labels, beliefs, visual_paths, visual_weights, strict=True)
            ):
                steps.append(
                    LiveThoughtStep(
                        index,
                        label,
                        previous.copy(),
                        belief.copy(),
                        guessed,
                        guessed_parameters,
                        step_paths,
                        step_weights,
                        evaluate_constraint_recovery(record.world_flat, guessed),
                        parameter_posterior,
                    )
                )
                previous = belief
            model1_checkpoint_version = seeing_result.metrics.updates
            event = CounterfactualEpisode(
                "live-thinking", self._episode_index, initial
            )
            runner, observed_activations = (
                FrozenOnlineSeeingCounterfactualRunner.from_observed(
                    self.seeing_runtime.model,
                    self.seeing_runtime.world_latent,
                    self.model1_config,
                    ((event, observed),),
                    checkpoint_id=f"online-seeing-{model1_checkpoint_version}",
                )
            )
            self.model2_observer.submit(
                Model1ActivationCheckpoint(
                    model1_checkpoint_version,
                    ActivationImaginationLoop(observed_activations, runner),
                )
            )
            training = dict(
                loss=0.0,
                weight_update_count=self.model2_runtime.weight_update_count,
                seeing_loss=seeing_result.metrics.prediction_loss,
                seeing_updates=seeing_result.metrics.updates,
            )
            self._latent_updates += 1
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
                model1_checkpoint_version=model1_checkpoint_version,
                model2_checkpoint_version=model2_checkpoint_version,
            )
            self._episode_index += 1
            self._completed_events += 1
            return result

    def close(self) -> None:
        self.model2_observer.close()
