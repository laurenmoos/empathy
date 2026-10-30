"""Durable streaming episode records that reload without simulation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .streaming import OcclusionEpisode


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
