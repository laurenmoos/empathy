"""Deterministic visual diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .world import WorldCode, WorldSchema


@dataclass(frozen=True)
class VisualArtifact:
    path: Path
    metadata: dict[str, Any]
    metadata_path: Path


def _save_visual(fig: plt.Figure, path: Path, metadata: dict[str, Any]) -> VisualArtifact:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=96, bbox_inches="tight", metadata={"Software": "perception-env"})
    plt.close(fig)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return VisualArtifact(path=path, metadata=metadata, metadata_path=metadata_path)


def render_world_card(
    world: WorldCode, schema: WorldSchema, path: Path
) -> VisualArtifact:
    schema.validate(world)
    rows: list[list[str]] = []
    slot_metadata: list[dict[str, Any]] = []
    for spec, slot in zip(schema.specs, world.slots):
        parameters = slot.parameter_dict(spec)
        rows.append(
            [
                spec.name.replace("_", " "),
                "active" if slot.active else "inactive",
                ", ".join(f"{key}={value:.3f}" for key, value in parameters.items()),
            ]
        )
        slot_metadata.append(
            {"name": spec.name, "active": slot.active, "parameters": parameters}
        )

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axis("off")
    ax.set_title(f"World card · schema {world.schema_version}", loc="left", weight="bold")
    table = ax.table(
        cellText=rows,
        colLabels=["operator", "state", "physical parameters"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.38, 0.18, 0.44],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    metadata: dict[str, Any] = {
        "kind": "world_card",
        "schema_version": world.schema_version,
        "operators": list(schema.slot_names),
        "slots": slot_metadata,
    }
    return _save_visual(fig, Path(path), metadata)


def render_trajectory_strip(
    trajectory: "Trajectory", world: WorldCode, schema: WorldSchema, path: Path
) -> VisualArtifact:
    from .simulate import Trajectory

    if not isinstance(trajectory, Trajectory):
        raise TypeError("trajectory must be a Trajectory")
    schema.validate(world)
    xs = [state.x for state in trajectory.states]
    ys = [state.y for state in trajectory.states]
    times = list(range(len(trajectory.states)))
    active = [slot.name for slot in world.slots if slot.active]

    fig, (path_ax, summary_ax) = plt.subplots(
        1, 2, figsize=(8.5, 3.8), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    path_ax.plot(xs, ys, color="#315b7d", linewidth=1.2, alpha=0.8)
    scatter = path_ax.scatter(xs, ys, c=times, cmap="viridis", s=24, zorder=3)
    path_ax.scatter([xs[0]], [ys[0]], marker="o", color="#2d8c5a", s=55, label="start")
    path_ax.scatter([xs[-1]], [ys[-1]], marker="x", color="#b13a3a", s=60, label="end")
    vertical_wall = world.slot("vertical_wall")
    if vertical_wall.active:
        path_ax.axvline(vertical_wall.parameters[0], color="#c47f2b", linestyle="--")
    horizontal_wall = world.slot("horizontal_wall")
    if horizontal_wall.active:
        path_ax.axhline(horizontal_wall.parameters[0], color="#c47f2b", linestyle="--")
    path_ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", title="Latent trajectory")
    path_ax.set_aspect("equal")
    path_ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=path_ax, label="step", fraction=0.046)

    summary_ax.axis("off")
    summary_ax.set_title("Operator summary", loc="left", weight="bold")
    lines = [f"seed: {trajectory.seed}", f"valid: {trajectory.valid}"]
    for spec, slot in zip(schema.specs, world.slots):
        if slot.active:
            values = slot.parameter_dict(spec)
            lines.append(
                f"{slot.name}\n  "
                + ", ".join(f"{name}={value:.3f}" for name, value in values.items())
            )
    if not active:
        lines.append("constant motion only")
    summary_ax.text(0.0, 0.94, "\n\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.suptitle("Deterministic world rollout", x=0.06, ha="left", weight="bold")

    metadata: dict[str, Any] = {
        "kind": "trajectory_strip",
        "panels": ["latent trajectory", "operator summary"],
        "axes": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "seed": trajectory.seed,
        "active_operators": active,
        "state_count": len(trajectory.states),
        "collision_count": len(trajectory.collisions),
    }
    return _save_visual(fig, Path(path), metadata)


def render_observation_strip(
    sequence: "RenderedSequence", path: Path
) -> VisualArtifact:
    from .events import extract_reappearance
    from .render import RenderedSequence, _hidden_runs

    if not isinstance(sequence, RenderedSequence):
        raise TypeError("sequence must be a RenderedSequence")
    fig = plt.figure(figsize=(11.0, 6.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    latent_ax = fig.add_subplot(grid[0, 0])
    frames_ax = fig.add_subplot(grid[0, 1])
    timeline_ax = fig.add_subplot(grid[1, :])

    xs = [state.x for state in sequence.latent_states]
    ys = [state.y for state in sequence.latent_states]
    latent_ax.plot(xs, ys, color="#315b7d", marker="o", markersize=3)
    for occluder in sequence.occluders:
        latent_ax.add_patch(
            plt.Rectangle(
                (occluder.x_min, occluder.y_min),
                occluder.x_max - occluder.x_min,
                occluder.y_max - occluder.y_min,
                color="#777777",
                alpha=0.35,
            )
        )
    latent_ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", title="Latent path")
    latent_ax.set_aspect("equal")

    count = min(6, len(sequence.frames))
    indices = np.linspace(0, len(sequence.frames) - 1, count, dtype=int)
    frame_strip = np.concatenate([sequence.frames[index] for index in indices], axis=1)
    frames_ax.imshow(frame_strip, cmap="gray", vmin=0, vmax=255)
    frames_ax.set_title("Visible frames")
    frames_ax.set_xticks(
        [(offset + 0.5) * sequence.config.width for offset in range(count)],
        labels=[str(index) for index in indices],
    )
    frames_ax.set_yticks([])
    frames_ax.set_xlabel("sampled frame step")

    steps = np.arange(len(sequence.frames))
    timeline_ax.step(steps, sequence.visible.astype(int), where="mid", label="visible")
    timeline_ax.fill_between(
        steps,
        0,
        sequence.fully_hidden.astype(int),
        step="mid",
        alpha=0.3,
        color="#b13a3a",
        label="fully hidden",
    )
    timeline_ax.set(
        ylim=(-0.05, 1.15),
        xlabel="step",
        ylabel="visibility",
        title="Visibility timeline",
    )
    timeline_ax.legend(loc="upper right")
    event = extract_reappearance(sequence)
    if event is not None:
        timeline_ax.axvline(event.step, color="#2d8c5a", linestyle="--", label="reappearance")
    fig.suptitle("Observation transform: latent motion → occluded frames", weight="bold")
    fig.tight_layout()

    hidden_intervals = [list(run) for run in _hidden_runs(sequence.fully_hidden)]
    metadata: dict[str, Any] = {
        "kind": "observation_strip",
        "panels": ["latent path", "visible frames", "visibility timeline"],
        "seed": sequence.seed,
        "hidden_intervals": hidden_intervals,
        "reappearance_step": event.step if event is not None else None,
        "rejection_reason": sequence.rejection_reason,
    }
    return _save_visual(fig, Path(path), metadata)


def render_operator_comparison(
    trajectories: list["Trajectory"],
    worlds: list[WorldCode],
    schema: WorldSchema,
    *,
    operator: str,
    path: Path,
) -> VisualArtifact:
    from .simulate import Trajectory

    if len(trajectories) != len(worlds) or not trajectories:
        raise ValueError("trajectories and worlds must have the same nonzero length")
    if not all(isinstance(item, Trajectory) for item in trajectories):
        raise TypeError("all trajectories must be Trajectory instances")
    seeds = {trajectory.seed for trajectory in trajectories}
    initials = {trajectory.states[0] for trajectory in trajectories}
    if len(seeds) != 1 or len(initials) != 1:
        raise ValueError("operator comparison requires a shared seed and initial state")
    parameter_values = [world.slot(operator).parameters[0] for world in worlds]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(trajectories)))
    for trajectory, value, color in zip(trajectories, parameter_values, colors):
        xs = [state.x for state in trajectory.states]
        ys = [state.y for state in trajectory.states]
        ax.plot(xs, ys, marker="o", markersize=3, color=color, label=f"{value:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", title="One-operator comparison")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(title=operator.replace("_", " "))
    fig.tight_layout()
    metadata: dict[str, Any] = {
        "kind": "operator_comparison",
        "operator": operator,
        "parameter_values": parameter_values,
        "same_seed": True,
        "same_initial_state": True,
        "seed": trajectories[0].seed,
    }
    return _save_visual(fig, Path(path), metadata)


def render_belief_panel(
    result: "ReachabilityResult",
    event: "BeliefEvent",
    render: "RenderConfig",
    path: Path,
    *,
    true_endpoint: tuple[float, float] | None = None,
) -> VisualArtifact:
    from .beliefs import BeliefEvent
    from .config import RenderConfig
    from .reachability import ReachabilityResult, rasterize_endpoints

    if not isinstance(result, ReachabilityResult) or not isinstance(event, BeliefEvent):
        raise TypeError("result and event must use reachability and belief record types")
    if event.pre.shape != (render.height, render.width):
        raise ValueError("belief image shape must match render configuration")

    occupancy_stack = rasterize_endpoints(result, render)
    occupancy = occupancy_stack[-1] if len(occupancy_stack) else np.zeros(event.pre.shape, bool)
    endpoints = result.endpoints[result.query_steps[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.6))
    axes_flat = axes.ravel()

    paths_ax = axes_flat[0]
    for trajectory in result.trajectories:
        xs = [state.x for state in trajectory.states]
        ys = [state.y for state in trajectory.states]
        paths_ax.plot(xs, ys, alpha=0.3, linewidth=1)
    paths_ax.set(xlim=(0, 1), ylim=(0, 1), title="Sampled paths", xlabel="x", ylabel="y")

    endpoint_ax = axes_flat[1]
    endpoint_ax.scatter(
        [state.x for state in endpoints],
        [state.y for state in endpoints],
        s=18,
        color="#315b7d",
    )
    endpoint_ax.set(xlim=(0, 1), ylim=(0, 1), title="Endpoint scatter", xlabel="x", ylabel="y")

    axes_flat[2].imshow(occupancy, cmap="gray", vmin=0, vmax=1)
    axes_flat[2].set_title("Binary occupancy")
    axes_flat[3].imshow(event.pre, cmap="magma", vmin=0, vmax=1)
    axes_flat[3].set_title("Pre belief")
    axes_flat[4].imshow(event.post, cmap="magma", vmin=0, vmax=1)
    axes_flat[4].set_title("Post belief")
    difference_limit = max(float(np.max(np.abs(event.difference))), 1.0e-6)
    axes_flat[5].imshow(
        event.difference,
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes_flat[5].set_title("Belief difference")

    diagnostic_ax = axes_flat[6]
    diagnostic_ax.imshow(event.post, cmap="gray", vmin=0, vmax=1)
    if true_endpoint is not None:
        column = true_endpoint[0] * (render.width - 1)
        row = (1.0 - true_endpoint[1]) * (render.height - 1)
        diagnostic_ax.scatter([column], [row], marker="x", color="#31d27c", s=75)
    diagnostic_ax.set_title("First-order diagnostic")
    axes_flat[7].axis("off")
    axes_flat[7].text(
        0.0,
        1.0,
        "Model 2 receives only:\n• pre belief\n• post belief\n• belief difference",
        va="top",
        fontsize=10,
    )
    for ax in axes_flat[2:7]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Reachability and second-order belief representation", weight="bold")
    fig.tight_layout()

    metadata: dict[str, Any] = {
        "kind": "belief_panel",
        "panels": [
            "sampled paths",
            "endpoint scatter",
            "binary occupancy",
            "pre belief",
            "post belief",
            "belief difference",
            "first-order diagnostic",
        ],
        "model2_channels": [
            "pre_belief",
            "post_belief",
            "belief_difference",
        ],
        "first_order_excluded": True,
        "endpoint_count": len(endpoints),
        "query_step": result.query_steps[-1],
    }
    return _save_visual(fig, Path(path), metadata)


def render_dataset_contact_sheet(
    records: list["BenchmarkRecord"], schema: WorldSchema, path: Path
) -> VisualArtifact:
    from .records import BenchmarkRecord

    if not records or not all(isinstance(record, BenchmarkRecord) for record in records):
        raise ValueError("records must be a nonempty list of BenchmarkRecord instances")
    fig, axes = plt.subplots(len(records), 3, figsize=(10.5, 3.1 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        world = record.world(schema)
        active = [slot.name for slot in world.slots if slot.active]
        axes[row, 0].axis("off")
        axes[row, 0].set_title("World code", loc="left")
        axes[row, 0].text(
            0.0,
            0.95,
            f"{record.family_level} · seed {record.seed}\n"
            + ("\n".join(active) if active else "constant motion"),
            va="top",
            family="monospace",
        )
        sequence = record.sequence()
        middle = len(sequence.frames) // 2
        axes[row, 1].imshow(sequence.frames[middle], cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"Observed frame · t={middle}")
        axes[row, 2].imshow(record.arrays["pre_belief"], cmap="magma", vmin=0, vmax=1)
        axes[row, 2].set_title("Pre belief")
        for column in (1, 2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.suptitle("Reloaded benchmark records", weight="bold")
    fig.tight_layout()
    metadata: dict[str, Any] = {
        "kind": "dataset_contact_sheet",
        "source": "reloaded_records",
        "record_ids": [record.record_id for record in records],
        "family_levels": [record.family_level for record in records],
        "panels_per_record": ["world code", "observed frame", "pre belief"],
    }
    return _save_visual(fig, Path(path), metadata)


def render_shattering_matrix(
    estimate: "ShatteringEstimate", path: Path
) -> VisualArtifact:
    from .shattering import ShatteringEstimate, count_labelings

    if not isinstance(estimate, ShatteringEstimate):
        raise TypeError("estimate must be a ShatteringEstimate")
    matrix = estimate.matrix
    unique_rows, inverse, counts = np.unique(
        matrix.values, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse)
    grouped = matrix.values[order]

    fig, (matrix_ax, summary_ax) = plt.subplots(
        1, 2, figsize=(9.5, max(3.5, 0.22 * len(grouped) + 1.8)),
        gridspec_kw={"width_ratios": [1.8, 1.0]},
    )
    matrix_ax.imshow(grouped, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    matrix_ax.set(
        xlabel="probe",
        ylabel="world (grouped by labeling)",
        title="World × probe possibility labels",
    )
    for index in estimate.selected_probe_indices:
        matrix_ax.add_patch(
            plt.Rectangle(
                (index - 0.5, -0.5),
                1,
                len(grouped),
                fill=False,
                edgecolor="#e07a2d",
                linewidth=2,
            )
        )
    summary_ax.axis("off")
    summary_ax.set_title("Empirical shattering estimate", loc="left", weight="bold")
    summary_ax.text(
        0.0,
        0.95,
        "\n".join(
            [
                f"metric: {estimate.metric_name}",
                f"dimension: {estimate.dimension}",
                f"selected probes: {list(estimate.selected_probe_indices)}",
                f"realized: {estimate.realized_labelings}/{estimate.possible_labelings}",
                f"distinct rows: {len(unique_rows)}",
                f"duplicates: {int(np.sum(counts - 1))}",
                f"budget: {estimate.metadata['budget']}",
                f"exhaustive: {estimate.exhaustive}",
            ]
        ),
        va="top",
        family="monospace",
    )
    fig.tight_layout()
    metadata: dict[str, Any] = {
        "kind": "shattering_matrix",
        "dimension": estimate.dimension,
        "selected_probe_indices": list(estimate.selected_probe_indices),
        "distinct_labelings": count_labelings(matrix),
        "world_count": matrix.values.shape[0],
        "probe_count": matrix.values.shape[1],
        "budget": estimate.metadata["budget"],
        "metric_name": estimate.metric_name,
    }
    return _save_visual(fig, Path(path), metadata)
