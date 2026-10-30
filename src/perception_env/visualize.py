"""Deterministic visual diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

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
