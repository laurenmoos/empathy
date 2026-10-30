"""Deterministic Model 1 figures rendered only from saved prediction values."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from perception_env.visualize import VisualArtifact

from .evaluate import EvaluationSummary, Model1PredictionRecord


def _save(fig: plt.Figure, path: Path, metadata: dict[str, Any]) -> VisualArtifact:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=96, bbox_inches="tight", metadata={"Software": "perception-env"})
    plt.close(fig)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return VisualArtifact(path, metadata, sidecar)


def _plot_paths(ax, positions: np.ndarray, weights: np.ndarray, target: np.ndarray, mask):
    scale = max(float(weights.max()), 1.0e-8)
    for path, weight in zip(positions, weights, strict=True):
        ax.plot(path[:, 0], path[:, 1], color="#2474b5", alpha=0.15 + 0.75 * weight / scale)
    ax.plot(target[mask, 0], target[mask, 1], color="#e4572e", linewidth=2, label="truth")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.grid(alpha=0.15)


def render_prediction_panel(
    record: Model1PredictionRecord, path: Path
) -> VisualArtifact:
    arrays = record.arrays
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    pre_valid = np.flatnonzero(arrays["pre_mask"])
    post_valid = np.flatnonzero(arrays["post_mask"])
    axes[0, 0].imshow(arrays["pre_frames"][pre_valid[-1], 0], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("first-order pre clip")
    axes[1, 0].imshow(
        arrays["post_frames"][post_valid[-1], 0], cmap="gray", vmin=0, vmax=1
    )
    axes[1, 0].set_title("first-order assimilated clip")
    _plot_paths(
        axes[0, 1],
        arrays["pre_positions"],
        arrays["pre_weights"],
        arrays["target_positions"],
        arrays["target_mask"],
    )
    axes[0, 1].set_title("pre trajectory mixture")
    _plot_paths(
        axes[1, 1],
        arrays["post_positions"],
        arrays["post_weights"],
        arrays["target_positions"],
        arrays["target_mask"],
    )
    axes[1, 1].set_title("post trajectory mixture")
    for axis, name, title, lower, upper in (
        (axes[0, 2], "pre_belief", "belief-only pre", 0, 1),
        (axes[1, 2], "post_belief", "belief-only post", 0, 1),
        (axes[0, 3], "belief_difference", "belief-only difference", -1, 1),
        (axes[1, 3], "actual_path_belief", "actual-path target", 0, 1),
    ):
        axis.imshow(arrays[name], cmap="coolwarm" if lower < 0 else "magma", vmin=lower, vmax=upper)
        axis.set_title(title)
    for axis in (axes[0, 0], axes[1, 0], axes[0, 2], axes[1, 2], axes[0, 3], axes[1, 3]):
        axis.axis("off")
    fig.suptitle(
        f"{record.example_id} · {record.family_level} · seed {record.seed} · "
        f"pre minADE {record.metrics.get('pre_min_ade', float('nan')):.3f} · "
        f"confidence {record.metrics.get('pre_confidence', float('nan')):.3f} · "
        f"belief IoU {record.metrics.get('pre_belief_iou', float('nan')):.3f}"
    )
    fig.tight_layout()
    return _save(
        fig,
        path,
        {
            "kind": "model1_prediction_panel",
            "example_id": record.example_id,
            "family_level": record.family_level,
            "seed": record.seed,
            "source": "saved_prediction_record",
        },
    )


def render_calibration_plot(
    summary: EvaluationSummary, path: Path
) -> VisualArtifact:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, color in (("pre", "#2474b5"), ("post", "#e4572e")):
        values = summary.calibration[name]
        points = [
            (confidence, accuracy)
            for confidence, accuracy in zip(
                values["bin_confidence"], values["bin_accuracy"], strict=True
            )
            if confidence is not None
        ]
        if points:
            axes[0].plot(
                *zip(*points, strict=True),
                marker="o",
                label=f"{name} ECE={values['ece']:.3f}",
                color=color,
            )
        coverage = values["credible_coverage"]
        axes[1].plot(
            [0.5, 0.8, 0.9],
            [coverage["0.5"], coverage["0.8"], coverage["0.9"]],
            marker="o",
            label=name,
            color=color,
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4)
    axes[0].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="confidence",
        ylabel="accuracy",
        title="top-mode reliability",
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4)
    axes[1].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="credible mass",
        ylabel="empirical coverage",
        title="credible-mode coverage",
    )
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    return _save(
        fig,
        path,
        {
            "kind": "model1_calibration",
            "seed": summary.seed,
            "config_hash": summary.config_hash,
            "checkpoint_hash": summary.checkpoint_hash,
        },
    )


def render_failure_gallery(
    records: Sequence[Model1PredictionRecord], path: Path, *, top_n: int = 8
) -> VisualArtifact:
    if top_n < 1 or not records:
        raise ValueError("failure gallery needs records and positive top_n")
    selected = sorted(
        records, key=lambda record: record.metrics.get("pre_min_ade", 0.0), reverse=True
    )[:top_n]
    fig, axes = plt.subplots(len(selected), 2, figsize=(8, 3 * len(selected)), squeeze=False)
    for row, record in enumerate(selected):
        axes[row, 0].imshow(record.arrays["pre_belief"], cmap="magma", vmin=0, vmax=1)
        axes[row, 1].imshow(record.arrays["post_belief"], cmap="magma", vmin=0, vmax=1)
        axes[row, 0].set_title(
            f"{record.example_id} · {record.family_level} · pre minADE "
            f"{record.metrics.get('pre_min_ade', float('nan')):.3f}"
        )
        axes[row, 1].set_title("post belief")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    fig.tight_layout()
    return _save(
        fig,
        path,
        {
            "kind": "model1_failure_gallery",
            "example_ids": [record.example_id for record in selected],
            "families": [record.family_level for record in selected],
            "source": "saved_prediction_records",
        },
    )
