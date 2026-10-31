"""Visual proof from saved arrays only; no simulation or network inference."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from empathy.physics.world import WorldSchema
from empathy.reasoning.visualize import VisualArtifact, _save_visual

from .data import array_hash


def select_exemplars(world_ids, metrics, predicted_rasters):
    """Stable representatives of scientific outcomes, with explicit absent categories."""
    rows = metrics["per_example"]
    categories = {
        "success": lambda i: rows[i]["equivalent"],
        "failure": lambda i: not rows[i]["equivalent"],
        "empty": lambda i: not predicted_rasters[i].any(axis=(-2, -1)).all(),
        "equivalent_distinct_code": lambda i: rows[i]["equivalent"] and not rows[i]["exact_code"],
    }
    order = sorted(range(len(world_ids)), key=lambda i: (world_ids[i], i))
    return {
        name: next((i for i in order if matches(i)), None) for name, matches in categories.items()
    }


def render_evaluation(
    dataset_root: Path, evaluation_root: Path, output: Path
) -> tuple[VisualArtifact, ...]:
    root, output = Path(evaluation_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, key in (("metrics.json", "metrics_hash"), ("training.json", "training_hash")):
        if (
            key in manifest
            and hashlib.sha256((root / filename).read_bytes()).hexdigest() != manifest[key]
        ):
            raise ValueError("saved evaluation metadata hash mismatch")
    metrics = json.loads((root / "metrics.json").read_text())
    with np.load(root / "predictions.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if array_hash(arrays) != manifest["array_hash"]:
        raise ValueError("saved prediction hash mismatch")
    output.mkdir(parents=True)
    artifacts = []
    metadata = {
        **manifest,
        "source_arrays": str((root / "predictions.npz").resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
    }
    row = min(range(len(manifest["world_ids"])), key=lambda i: manifest["world_ids"][i])

    def save(fig, name):
        fig.suptitle(name.replace("_", " ").title() + "\nOracle-bootstrap beliefs", fontsize=13)
        fig.text(
            0.01,
            0.008,
            f"seed {manifest['seed']} · dataset {manifest['dataset_hash'][:12]} · "
            f"checkpoint {manifest['checkpoint_hash'][:12]} · saved arrays only",
            fontsize=7,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        artifacts.append(_save_visual(fig, output / f"{name}.png", {**metadata, "kind": name}))

    fig, axes = plt.subplots(4, 3, figsize=(9, 10))
    for event in range(4):
        for channel, title in enumerate(("Pre", "Post", "Post − pre")):
            ax = axes[event, channel]
            ax.imshow(
                arrays["beliefs"][row, event, channel],
                cmap="coolwarm" if channel == 2 else "viridis",
                vmin=-1 if channel == 2 else 0,
                vmax=1,
            )
            ax.set_title(
                f"{title} · event {event + 1}"
                + ("" if arrays["history_mask"][row, event] else " (padding)")
            )
            ax.axis("off")
    save(fig, "belief_history")

    schema = WorldSchema.v1()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    rows = []
    for label, key in [("True", "true_worlds"), ("Predicted", "decoded_worlds")]:
        world = schema.from_flat(arrays[key][row])
        for spec, slot in zip(schema.specs, world.slots, strict=True):
            values = slot.parameter_dict(spec)
            rows.append(
                [
                    label,
                    spec.name.replace("_", "\n"),
                    str(slot.active),
                    ", ".join(f"{name}={value:.3f}" for name, value in values.items()),
                    ", ".join(
                        f"{(values[p.name] - p.minimum) / (p.maximum - p.minimum):.3f}"
                        if slot.active
                        else "0 (inactive)"
                        for p in spec.parameters
                    ),
                ]
            )
    table = ax.table(
        cellText=rows,
        colLabels=["World", "Operator", "Active", "Physical", "Normalized"],
        loc="center",
        cellLoc="left",
        colWidths=[0.1, 0.23, 0.1, 0.32, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    save(fig, "world_cards")

    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(metrics["per_operator"])
    x = np.arange(len(names))
    for offset, metric in [(-0.18, "precision"), (0.18, "recall")]:
        values = [metrics["per_operator"][n][metric]["value"] for n in names]
        ax.bar(x + offset, [v if v is not None else 0 for v in values], width=0.35, label=metric)
        for i, value in enumerate(values):
            if value is None:
                ax.text(i + offset, 0.03, "N/A", rotation=90, ha="center", fontsize=8)
    ax.set_xticks(x, [n.replace("_", "\n") for n in names])
    ax.set_ylim(0, 1.1)
    ax.legend()
    save(fig, "activity_recovery")

    fig, ax = plt.subplots(figsize=(11, 4))
    errors = metrics["parameter_errors"]
    labels = [f"{spec.name}\n{p.name}" for spec in schema.specs for p in spec.parameters]
    ax.bar(range(7), [e["value"] or 0 for e in errors])
    ax.set_xticks(range(7), [label.replace("_", " ") for label in labels], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Raw normalized MAE")
    ax.text(
        0.01,
        0.97,
        "True-active, audited-identifiable fields only",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    for i, error in enumerate(errors):
        ax.text(
            i,
            (error["value"] or 0) + 0.03,
            f"n={error['denominator']}" + ("\nN/A" if error["value"] is None else ""),
            ha="center",
            fontsize=8,
        )
    save(fig, "parameter_errors")

    fig, axes = plt.subplots(4, 3, figsize=(9, 10))
    for event in range(4):
        truth = arrays["true_rasters"][row, event].astype(float)
        recovered = arrays["predicted_rasters"][row, event].astype(float)
        for column, (values, title) in enumerate(
            [(truth, "True"), (recovered, "Predicted"), (recovered - truth, "Predicted − true")]
        ):
            axes[event, column].imshow(
                values,
                vmin=-1 if column == 2 else 0,
                vmax=1,
                cmap="coolwarm" if column == 2 else "viridis",
            )
            axes[event, column].set_title(f"{title} · context {event + 1}")
            axes[event, column].axis("off")
    save(fig, "possibility_maps")

    exemplars = select_exemplars(manifest["world_ids"], metrics, arrays["predicted_rasters"])
    metadata["exemplar_rows"] = exemplars
    fig, axes = plt.subplots(4, 3, figsize=(9, 10))
    for case, (name, chosen) in enumerate(exemplars.items()):
        for ax in axes[case]:
            ax.axis("off")
        if chosen is None:
            axes[case, 1].text(
                0.5, 0.5, name.replace("_", " ") + "\nNo example present", ha="center", va="center"
            )
            continue
        event = 3
        if name == "empty":
            event = int(np.flatnonzero(~arrays["predicted_rasters"][chosen].any(axis=(-2, -1)))[0])
        truth = arrays["true_rasters"][chosen, event].astype(float)
        predicted = arrays["predicted_rasters"][chosen, event].astype(float)
        for column, (value, label) in enumerate(
            [(truth, "True"), (predicted, "Predicted"), (predicted - truth, "Difference")]
        ):
            axes[case, column].imshow(
                value,
                vmin=-1 if column == 2 else 0,
                vmax=1,
                cmap="coolwarm" if column == 2 else "viridis",
            )
            axes[case, column].set_title(name.replace("_", " ") + "\n" + label, fontsize=9)
    save(fig, "outcome_exemplars")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    candidates = [
        i
        for i, item in enumerate(metrics["per_example"])
        if item["equivalent"] and not item["exact_code"]
    ]
    if candidates:
        i = min(candidates, key=lambda j: manifest["world_ids"][j])
        ax.text(
            0.02,
            0.7,
            "Equivalent on saved probes; different operator codes.\n"
            "This does not establish universal equivalence.",
            fontsize=12,
        )
        ax.text(
            0.02,
            0.45,
            f"True: {np.round(arrays['true_worlds'][i], 3)}\n"
            f"Predicted: {np.round(arrays['decoded_worlds'][i], 3)}",
            fontsize=10,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No distinct-code equivalent example in this saved evaluation.",
            ha="center",
            va="center",
        )
    save(fig, "equivalent_code_gallery")

    fig, ax = plt.subplots(figsize=(8, 4))
    if (root / "training.json").exists():
        history = json.loads((root / "training.json").read_text())["epochs"]
        for key in ("train_total", "validation_total"):
            ax.plot([e["epoch"] for e in history], [e[key] for e in history], marker="o", label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Masked training objective")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No training history in saved evaluation (e.g. constant baseline).",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    save(fig, "training_curves")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    comparison_path = root / "controls.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text())
        rows = [
            [name, f"{value['world_macro_agreement']:.4f}"] for name, value in comparison.items()
        ]
        ax.table(
            cellText=rows,
            colLabels=["Control", "World-macro possibility agreement"],
            loc="center",
            cellLoc="center",
        ).scale(1, 1.6)
    else:
        ax.text(
            0.5,
            0.5,
            "Control comparison unavailable in this saved evaluation.",
            ha="center",
            va="center",
        )
    save(fig, "control_comparison")
    index = '<!doctype html><meta charset="utf-8"><title>Model 2 saved visual proof</title>'
    index += "<h1>Model 2 · Oracle-bootstrap beliefs</h1><p>Finite held-out probes; "
    index += (
        html.escape(str(manifest.get("heldout_scope", "scope recorded in evaluation"))) + "</p>"
    )
    index += f'<p><a href="{html.escape(metadata["source_arrays"])}">Saved source arrays</a></p>'
    for artifact in artifacts:
        index += f"<h2>{artifact.path.stem.replace('_', ' ')}</h2>"
        index += f'<a href="{artifact.path.with_suffix(".json").name}">Provenance</a><br>'
        index += f'<img style="max-width:100%" src="{artifact.path.name}">'
    (output / "index.html").write_text(index)
    return tuple(artifacts)
