"""Durable streaming episode records that reload without simulation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .beliefs import BeliefEvent
from .continual import ContinualLog
from .streaming import (
    FeedbackIteration,
    OcclusionEpisode,
    StreamingSessionState,
    StreamUpdate,
    StreamWorldPrediction,
)


def save_stream(path: Path, episodes: tuple[OcclusionEpisode, ...]) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not episodes:
        raise ValueError("cannot save an empty stream")
    path.mkdir(parents=True)
    metadata = {
        "episodes": [
            {
                "event_id": item.event_id,
                "context_id": item.context_id,
                "reappearance_step": item.reappearance_step,
                "reappearance_xy": list(item.reappearance_xy),
                "evaluation": item.evaluation,
            }
            for item in episodes
        ]
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    np.savez_compressed(
        path / "arrays.npz",
        frames=np.stack([item.frames for item in episodes]),
        visible=np.stack([item.visible for item in episodes]),
    )
    return path


def load_stream(path: Path) -> tuple[OcclusionEpisode, ...]:
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
        frames = arrays["frames"]
        visible = arrays["visible"]
    rows = metadata["episodes"]
    if len(rows) != len(frames) or len(rows) != len(visible):
        raise ValueError("stream metadata and arrays disagree")
    return tuple(
        OcclusionEpisode(
            event_id=row["event_id"],
            context_id=row["context_id"],
            frames=frames[index],
            visible=visible[index],
            reappearance_step=int(row["reappearance_step"]),
            reappearance_xy=tuple(float(value) for value in row["reappearance_xy"]),
            evaluation=dict(row["evaluation"]),
        )
        for index, row in enumerate(rows)
    )


@dataclass(frozen=True)
class LoadedStreamingRun:
    episodes: tuple[OcclusionEpisode, ...]
    updates: tuple[StreamUpdate, ...]
    session_state: StreamingSessionState
    continual_logs: tuple[ContinualLog, ...]
    metadata: dict[str, Any]


def _summary(
    updates: tuple[StreamUpdate, ...],
    session_state: StreamingSessionState,
    continual_logs: tuple[ContinualLog, ...],
) -> dict[str, Any]:
    weight_update_count = max(
        (log.weight_update_count for log in continual_logs),
        default=session_state.weight_update_count,
    )
    return {
        "coverage_by_event": {
            update.event_id: update.status == "covered" for update in updates
        },
        "feedback_iterations": {
            update.event_id: len(update.iterations) for update in updates
        },
        "statuses": {update.event_id: update.status for update in updates},
        "latent_update_count": session_state.latent_update_count,
        "weight_update_count": max(session_state.weight_update_count, weight_update_count),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_streaming_run(
    path: Path,
    episodes: tuple[OcclusionEpisode, ...],
    updates: tuple[StreamUpdate, ...],
    session_state: StreamingSessionState,
    continual_logs: tuple[ContinualLog, ...],
    metadata: dict[str, Any],
) -> Path:
    """Persist a complete streaming run for inference-free inspection."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not episodes or len(episodes) != len(updates):
        raise ValueError("episodes and updates must have the same nonzero length")
    if any(
        episode.event_id != update.event_id
        for episode, update in zip(episodes, updates, strict=True)
    ):
        raise ValueError("episode and update identifiers must align")
    required_metadata = {"stream_id", "seed", "schema_version", "checkpoint_hashes"}
    if not required_metadata <= set(metadata):
        raise ValueError("streaming run metadata is incomplete")
    path.mkdir(parents=True)

    arrays: dict[str, np.ndarray] = {"session_world_flat": session_state.world_flat}
    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for episode_index, (episode, update) in enumerate(
        zip(episodes, updates, strict=True)
    ):
        arrays[f"episode_{episode_index}_frames"] = episode.frames
        arrays[f"episode_{episode_index}_visible"] = episode.visible
        arrays[f"update_{episode_index}_pre"] = update.event.pre
        arrays[f"update_{episode_index}_post"] = update.event.post
        episode_rows.append(
            {
                "event_id": episode.event_id,
                "context_id": episode.context_id,
                "reappearance_step": episode.reappearance_step,
                "reappearance_xy": list(episode.reappearance_xy),
                "evaluation": episode.evaluation,
            }
        )
        iteration_rows: list[dict[str, Any]] = []
        for iteration_index, iteration in enumerate(update.iterations):
            prefix = f"update_{episode_index}_iteration_{iteration_index}"
            arrays[f"{prefix}_world"] = iteration.prediction.flat_world
            arrays[f"{prefix}_activity"] = iteration.prediction.activity_probabilities
            arrays[f"{prefix}_parameters"] = iteration.prediction.parameter_values
            iteration_rows.append(
                {
                    "index": iteration.index,
                    "confidence": iteration.prediction.confidence,
                    "covered": iteration.covered,
                }
            )
        update_rows.append(
            {
                "event_id": update.event_id,
                "status": update.status,
                "belief_step": update.event.step,
                "iterations": iteration_rows,
            }
        )

    for event_index, event in enumerate(session_state.events):
        arrays[f"session_event_{event_index}_pre"] = event.pre
        arrays[f"session_event_{event_index}_post"] = event.post

    payload = {
        "metadata": metadata,
        "episodes": episode_rows,
        "updates": update_rows,
        "session": {
            "processed_event_ids": list(session_state.processed_event_ids),
            "latent_update_count": session_state.latent_update_count,
            "weight_update_count": session_state.weight_update_count,
            "event_steps": [event.step for event in session_state.events],
        },
        "continual_logs": [asdict(log) for log in continual_logs],
    }
    (path / "metadata.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True)
    )
    (path / "streaming-summary.json").write_text(
        json.dumps(
            _summary(updates, session_state, continual_logs),
            indent=2,
            sort_keys=True,
        )
    )
    np.savez_compressed(path / "arrays.npz", **arrays)
    return path


def load_streaming_run(path: Path) -> LoadedStreamingRun:
    """Reload every object needed for a streaming timeline without inference."""
    path = Path(path)
    payload = json.loads((path / "metadata.json").read_text())
    with np.load(path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}

    episodes: list[OcclusionEpisode] = []
    updates: list[StreamUpdate] = []
    for episode_index, (episode_row, update_row) in enumerate(
        zip(payload["episodes"], payload["updates"], strict=True)
    ):
        episode = OcclusionEpisode(
            event_id=str(episode_row["event_id"]),
            context_id=str(episode_row["context_id"]),
            frames=arrays[f"episode_{episode_index}_frames"],
            visible=arrays[f"episode_{episode_index}_visible"],
            reappearance_step=int(episode_row["reappearance_step"]),
            reappearance_xy=tuple(float(value) for value in episode_row["reappearance_xy"]),
            evaluation=dict(episode_row["evaluation"]),
        )
        event = BeliefEvent(
            int(update_row["belief_step"]),
            arrays[f"update_{episode_index}_pre"],
            arrays[f"update_{episode_index}_post"],
        )
        iterations: list[FeedbackIteration] = []
        for iteration_index, iteration_row in enumerate(update_row["iterations"]):
            prefix = f"update_{episode_index}_iteration_{iteration_index}"
            prediction = StreamWorldPrediction(
                arrays[f"{prefix}_world"],
                arrays[f"{prefix}_activity"],
                arrays[f"{prefix}_parameters"],
                float(iteration_row["confidence"]),
            )
            iterations.append(
                FeedbackIteration(
                    int(iteration_row["index"]),
                    prediction,
                    bool(iteration_row["covered"]),
                )
            )
        update = StreamUpdate(
            str(update_row["event_id"]),
            str(update_row["status"]),
            event,
            tuple(iterations),
        )
        if episode.event_id != update.event_id:
            raise ValueError("saved episode and update identifiers disagree")
        episodes.append(episode)
        updates.append(update)

    session_row = payload["session"]
    session_events = tuple(
        BeliefEvent(
            int(step),
            arrays[f"session_event_{index}_pre"],
            arrays[f"session_event_{index}_post"],
        )
        for index, step in enumerate(session_row["event_steps"])
    )
    session_state = StreamingSessionState(
        arrays["session_world_flat"],
        session_events,
        tuple(str(value) for value in session_row["processed_event_ids"]),
        int(session_row["latent_update_count"]),
        int(session_row["weight_update_count"]),
    )
    continual_logs = tuple(
        ContinualLog(
            int(row["observed_events"]),
            int(row["latent_update_count"]),
            int(row["weight_update_count"]),
            None if row["metrics"] is None else dict(row["metrics"]),
        )
        for row in payload["continual_logs"]
    )
    return LoadedStreamingRun(
        tuple(episodes),
        tuple(updates),
        session_state,
        continual_logs,
        dict(payload["metadata"]),
    )
