"""Saved-record streaming timeline and deterministic quick demonstration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .config import RenderConfig, SimulationConfig
from .continual import ContinualLearner, ReplayBuffer
from .render import Occluder
from .simulate import BallState
from .stream_records import (
    LoadedStreamingRun,
    load_streaming_run,
    save_streaming_run,
)
from .streaming import (
    Model2Features,
    OcclusionEpisode,
    StreamingSession,
    StreamWorldPrediction,
    generate_stream,
)
from .visualize import VisualArtifact
from .world import WorldCode, WorldSchema


@dataclass(frozen=True)
class StreamingDemoConfig:
    history_length: int
    event_count: int
    max_feedback_iterations: int
    outcome_tolerance_pixels: int
    replay_capacity: int
    warmup: int
    batch_size: int
    update_every: int
    enable_weight_updates: bool

    def __post_init__(self) -> None:
        values = (
            self.history_length,
            self.event_count,
            self.max_feedback_iterations,
            self.replay_capacity,
            self.warmup,
            self.batch_size,
            self.update_every,
        )
        if any(value < 1 for value in values) or self.outcome_tolerance_pixels < 0:
            raise ValueError("streaming configuration values are out of range")
        if self.history_length != 4:
            raise ValueError("Model 2 history length is fixed at four")


def load_streaming_demo_config(path: Path) -> StreamingDemoConfig:
    values = yaml.safe_load(Path(path).read_text())
    return StreamingDemoConfig(**values["streaming"])


@dataclass(frozen=True)
class StreamingDemoResult:
    run_path: Path
    timeline_path: Path
    manifest_path: Path
    index_path: Path
    deterministic_hash: str


def _save_timeline(
    figure: plt.Figure, path: Path, metadata: dict[str, Any]
) -> VisualArtifact:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=96,
        bbox_inches="tight",
        metadata={"Software": "perception-env"},
    )
    plt.close(figure)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return VisualArtifact(path, metadata, metadata_path)


def _clip_strip(episode: OcclusionEpisode) -> np.ndarray:
    indices = np.unique(
        np.linspace(0, len(episode.frames) - 1, min(4, len(episode.frames)), dtype=int)
    )
    return np.concatenate([episode.frames[index] for index in indices], axis=1)


def render_streaming_timeline(
    run: LoadedStreamingRun, output: Path
) -> VisualArtifact:
    """Render first- and second-order streaming state from reloaded records only."""
    if not isinstance(run, LoadedStreamingRun) or not run.updates:
        raise ValueError("a nonempty reloaded streaming run is required")
    if len(run.episodes) != len(run.updates):
        raise ValueError("stream episodes and updates must align")
    rows = len(run.updates)
    fig, axes = plt.subplots(rows, 6, figsize=(15.5, max(3.1 * rows, 4.0)), squeeze=False)
    schema = WorldSchema.v1()
    weight_counts: dict[int, int] = {}
    for log in run.continual_logs:
        weight_counts[log.observed_events] = max(
            weight_counts.get(log.observed_events, 0), log.weight_update_count
        )

    for row, (episode, update) in enumerate(
        zip(run.episodes, run.updates, strict=True)
    ):
        axes[row, 0].imshow(_clip_strip(episode), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Model 1 · observed clip")
        axes[row, 1].imshow(update.event.pre, cmap="magma", vmin=0, vmax=1)
        axes[row, 1].set_title("pre belief")
        axes[row, 2].imshow(update.event.post, cmap="magma", vmin=0, vmax=1)
        axes[row, 2].set_title("post belief")
        limit = max(float(np.abs(update.event.difference).max()), 1.0e-6)
        axes[row, 3].imshow(
            update.event.difference,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axes[row, 3].set_title("belief difference")

        prediction_ax = axes[row, 4]
        prediction_ax.axis("off")
        last = update.iterations[-1]
        predicted_world = schema.from_flat(last.prediction.flat_world)
        active: list[str] = []
        for spec, slot in zip(schema.specs, predicted_world.slots, strict=True):
            if slot.active:
                parameters = slot.parameter_dict(spec)
                active.append(
                    f"{slot.name}: "
                    + ", ".join(f"{name}={value:.2f}" for name, value in parameters.items())
                )
        prediction_ax.set_title("Model 2 · world guess", loc="left")
        prediction_ax.text(
            0.0,
            0.92,
            "\n".join(
                [
                    *(active or ["constant motion"]),
                    f"confidence: {last.prediction.confidence:.2f}",
                    f"feedback iterations: {len(update.iterations)}",
                    f"covered: {last.covered}",
                ]
            ),
            va="top",
            family="monospace",
            fontsize=8,
        )

        evaluation_ax = axes[row, 5]
        evaluation_ax.axis("off")
        evaluation_ax.set_facecolor("#f0eee8")
        true_world = schema.from_flat(
            np.asarray(episode.evaluation["true_world_flat"], dtype=float)
        )
        true_active: list[str] = []
        for spec, slot in zip(schema.specs, true_world.slots, strict=True):
            if slot.active:
                parameters = slot.parameter_dict(spec)
                true_active.append(
                    f"{slot.name}: "
                    + ", ".join(f"{name}={value:.2f}" for name, value in parameters.items())
                )
        latent_updates = sum(len(item.iterations) for item in run.updates[: row + 1])
        evaluation_ax.set_title("Ground truth · evaluation only", loc="left")
        evaluation_ax.text(
            0.0,
            0.92,
            "\n".join(
                [
                    *(true_active or ["constant motion"]),
                    f"event: {episode.event_id}",
                    f"status: {update.status}",
                    f"latent updates: {latent_updates}",
                    f"weight updates: {weight_counts.get(row + 1, 0)}",
                ]
            ),
            va="top",
            family="monospace",
            fontsize=8,
        )
        for column in range(4):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(f"event {row}", weight="bold")

    metadata = {
        "kind": "streaming_timeline",
        "source": "reloaded_streaming_records",
        "panels": [
            "first_order_clip",
            "pre_belief",
            "post_belief",
            "belief_difference",
            "model2_world_prediction",
            "ground_truth_evaluation",
        ],
        "stream_id": run.metadata["stream_id"],
        "seed": run.metadata["seed"],
        "schema_version": run.metadata["schema_version"],
        "checkpoint_hashes": run.metadata["checkpoint_hashes"],
        "event_ids": [item.event_id for item in run.episodes],
        "statuses": [item.status for item in run.updates],
        "feedback_iterations": [len(item.iterations) for item in run.updates],
        "latent_update_count": run.session_state.latent_update_count,
        "weight_update_count": max(
            (
                log.weight_update_count
                for log in run.continual_logs
            ),
            default=run.session_state.weight_update_count,
        ),
    }
    fig.suptitle("Streaming second-order perception", weight="bold")
    fig.text(
        0.01,
        0.005,
        f"stream={metadata['stream_id']} · seed={metadata['seed']} · "
        f"schema={metadata['schema_version']} · events={rows} · "
        f"model1={metadata['checkpoint_hashes']['model1'][:20]} · "
        f"model2={metadata['checkpoint_hashes']['model2'][:20]}",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    return _save_timeline(fig, Path(output), metadata)


class _FixtureBeliefRuntime:
    def propose(
        self, episode: OcclusionEpisode, world_prior: np.ndarray | None
    ) -> np.ndarray:
        height, width = episode.frames.shape[-2:]
        image = np.zeros((height, width), dtype=np.float32)
        first_visible = int(np.flatnonzero(episode.visible)[0])
        frame = episode.frames[first_visible]
        row = int(np.argmax(frame.max(axis=1))) if frame.any() else height // 2
        image[row, max(0, width // 4) : min(width, 3 * width // 4)] = (
            0.65 if world_prior is not None else 0.35
        )
        return image

    def assimilate(
        self, episode: OcclusionEpisode, pre_belief: np.ndarray
    ) -> np.ndarray:
        post = np.asarray(pre_belief, dtype=np.float32).copy()
        height, width = post.shape
        column = int(round(episode.reappearance_xy[0] * (width - 1)))
        row = int(round((1.0 - episode.reappearance_xy[1]) * (height - 1)))
        post[row, column] = 1.0
        return post


class _FixtureWorldRuntime:
    def __init__(self, target: WorldCode) -> None:
        schema = WorldSchema.v1()
        self._target = target.to_flat(schema)
        self._wrong = schema.world().to_flat(schema)
        self._calls = 0
        self._activity_indices = (0, 2, 4, 6, 9)
        self._parameter_indices = (1, 3, 5, 7, 8, 10, 11)

    def predict(self, features: Model2Features) -> StreamWorldPrediction:
        del features
        self._calls += 1
        flat = self._wrong if self._calls % 2 else self._target
        activity = flat[list(self._activity_indices)]
        parameters = flat[list(self._parameter_indices)]
        confidence = 0.35 if self._calls % 2 else 0.9
        return StreamWorldPrediction(flat, activity, parameters, confidence)


def _possibility_fixture(target: WorldCode, size: int):
    schema = WorldSchema.v1()
    target_flat = target.to_flat(schema)

    def possibility(world: WorldCode, episode: OcclusionEpisode) -> np.ndarray:
        image = np.zeros((size, size), dtype=bool)
        if np.array_equal(world.to_flat(schema), target_flat):
            column = int(round(episode.reappearance_xy[0] * (size - 1)))
            row = int(round((1.0 - episode.reappearance_xy[1]) * (size - 1)))
            image[row, column] = True
        return image

    return possibility


def _canonical_run_hash(run: LoadedStreamingRun) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(run.metadata, sort_keys=True).encode())
    for episode, update in zip(run.episodes, run.updates, strict=True):
        for array in (episode.frames, episode.visible, update.event.pre, update.event.post):
            values = np.asarray(array)
            digest.update(str(values.shape).encode())
            digest.update(str(values.dtype).encode())
            digest.update(values.tobytes())
        digest.update(update.status.encode())
        for iteration in update.iterations:
            digest.update(iteration.prediction.flat_world.tobytes())
            digest.update(bytes([iteration.covered]))
    digest.update(run.session_state.world_flat.tobytes())
    digest.update(
        json.dumps(
            [
                {
                    "observed_events": log.observed_events,
                    "latent_update_count": log.latent_update_count,
                    "weight_update_count": log.weight_update_count,
                    "metrics": log.metrics,
                }
                for log in run.continual_logs
            ],
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_quick_streaming_demo(
    output: Path,
    *,
    seed: int = 7,
    event_count: int = 4,
) -> StreamingDemoResult:
    """Run deterministic fixture models through the real streaming contracts."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if event_count < 1:
        raise ValueError("event count must be positive")
    output.mkdir(parents=True)
    config = StreamingDemoConfig(
        history_length=4,
        event_count=event_count,
        max_feedback_iterations=3,
        outcome_tolerance_pixels=1,
        replay_capacity=64,
        warmup=4,
        batch_size=4,
        update_every=1,
        enable_weight_updates=False,
    )
    schema = WorldSchema.v1()
    target = schema.world(horizontal_acceleration={"a_x": 0.2})
    initial_states = tuple(
        BallState(0.12, 0.38 + 0.24 * index / max(event_count - 1, 1), 0.25, 0.0)
        for index in range(event_count)
    )
    render_size = 24
    episodes = generate_stream(
        target,
        initial_states,
        (Occluder(0.35, 0.55, 0.05, 0.95),),
        SimulationConfig(dt=0.1, steps=32, seed=seed),
        RenderConfig(
            width=render_size,
            height=render_size,
            ball_radius=1,
            min_hidden_steps=3,
        ),
        "quick-stream",
    )
    if len(episodes) != event_count:
        raise RuntimeError("quick fixture did not produce every requested event")

    model1_hash = "fixture-" + hashlib.sha256(b"model1-belief-runtime-v1").hexdigest()
    model2_hash = "fixture-" + hashlib.sha256(b"model2-world-runtime-v1").hexdigest()
    session = StreamingSession(
        _FixtureBeliefRuntime(),
        _FixtureWorldRuntime(target),
        _possibility_fixture(target, render_size),
        initial_world_flat=schema.world().to_flat(schema),
        max_feedback_iterations=config.max_feedback_iterations,
        outcome_tolerance_pixels=config.outcome_tolerance_pixels,
    )
    learner = ContinualLearner(
        ReplayBuffer(config.replay_capacity, seed=seed),
        lambda batch: {"loss": float(len(batch))},
        warmup=config.warmup,
        batch_size=config.batch_size,
        update_every=config.update_every,
    )
    updates = []
    for episode in episodes:
        update = session.process(episode)
        updates.append(update)
        learner.observe(update)
    state = session.snapshot()
    metadata = {
        "stream_id": "quick-stream",
        "seed": seed,
        "schema_version": schema.version,
        "checkpoint_hashes": {"model1": model1_hash, "model2": model2_hash},
        "runtime": "deterministic_fixture",
        "config": asdict(config),
    }
    run_path = save_streaming_run(
        output / "run",
        episodes,
        tuple(updates),
        state,
        tuple(learner.logs),
        metadata,
    )
    loaded = load_streaming_run(run_path)
    artifact = render_streaming_timeline(loaded, output / "streaming-timeline.png")
    index_path = output / "index.html"
    index_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Streaming perception demo</title></head><body>"
        "<h1>Streaming second-order perception</h1>"
        "<p>Deterministic runtime fixtures use the production belief-only contracts.</p>"
        f"<img src='{artifact.path.name}' alt='streaming belief revision timeline'>"
        "</body></html>"
    )
    deterministic_hash = _canonical_run_hash(loaded)
    output_paths = [
        "index.html",
        "streaming-timeline.json",
        "streaming-timeline.png",
        "run/arrays.npz",
        "run/metadata.json",
        "run/streaming-summary.json",
    ]
    manifest = {
        "command": "package_local_quick_streaming_demo",
        "runtime": "deterministic_fixture",
        "seed": seed,
        "event_count": event_count,
        "accepted_count": len(episodes),
        "coverage_by_event": [update.status == "covered" for update in updates],
        "feedback_iterations": [len(update.iterations) for update in updates],
        "latent_update_count": state.latent_update_count,
        "weight_update_count": state.weight_update_count,
        "checkpoint_hashes": metadata["checkpoint_hashes"],
        "schema_version": schema.version,
        "config": asdict(config),
        "output_paths": output_paths,
        "output_content_hashes": {
            path: _file_hash(output / path) for path in output_paths
        },
        "deterministic_hash": deterministic_hash,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return StreamingDemoResult(
        run_path,
        artifact.path,
        manifest_path,
        index_path,
        deterministic_hash,
    )
