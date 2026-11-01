"""Command-line entry point for reproducible environment demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .physics.config import RenderConfig, SimulationConfig
from .physics.events import extract_reappearance
from .physics.families import load_family, sample_worlds
from .physics.render import Occluder, render_sequence
from .physics.simulate import BallState, rollout
from .physics.world import WorldCode, WorldSchema
from .reasoning.beliefs import BeliefEvent, build_belief_history, superpose
from .reasoning.identifiability import group_equivalent_worlds
from .reasoning.reachability import Probe, enumerate_reachable, rasterize_endpoints
from .reasoning.records import (
    BenchmarkRecord,
    Model2InputRecord,
    belief_entropy,
    load_record,
    save_record,
)
from .reasoning.shattering import LabelMatrix, build_label_matrix, search_shattered_set
from .reasoning.visualize import (
    render_belief_panel,
    render_dataset_contact_sheet,
    render_equivalence_gallery,
    render_observation_strip,
    render_shattering_matrix,
    render_trajectory_strip,
    render_world_card,
)


def _prepare_output(output: Path, force: bool) -> None:
    output = Path(output)
    if output.exists():
        if not force:
            raise FileExistsError(f"output already exists: {output}")
        if output.resolve() in {Path.cwd().resolve(), Path("/").resolve()}:
            raise ValueError("refusing to replace a broad output directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def _config_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _content_hashes(output: Path, paths: Sequence[str]) -> dict[str, str]:
    return {
        path: hashlib.sha256((output / path).read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def _validate_source_output(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("source and output paths overlap")


def _versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("empathy", "numpy", "matplotlib", "pillow", "pyyaml"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unknown"
    return result


def _write_manifest(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    stable = {
        key: value
        for key, value in manifest.items()
        if key not in {"wall_clock_duration_seconds", "output_directory", "deterministic_hash"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    manifest["deterministic_hash"] = hashlib.sha256(encoded).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _html_index(output: Path, title: str, images: Sequence[Path]) -> Path:
    cards = "\n".join(
        f'<figure><img src="{image.name}" alt="{image.stem}">'
        f"<figcaption>{image.stem}</figcaption></figure>"
        for image in images
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font:16px system-ui;margin:2rem;background:#f5f2eb;color:#17232d}}
main{{display:grid;gap:1.5rem}}
figure{{margin:0;background:white;padding:1rem;border-radius:12px}}
img{{max-width:100%;height:auto}}
figcaption{{font-weight:600;margin-top:.5rem}}
</style></head><body><h1>{title}</h1>
<p>Every panel is regenerated from saved benchmark records.</p>
<main>{cards}</main></body></html>"""
    path = output / "index.html"
    path.write_text(html)
    return path


def sample_run(
    *, config_path: str | Path, level: str, count: int, seed: int, output: str | Path, force: bool
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    output = Path(output)
    _prepare_output(output, force)
    family = load_family(config_path, level)
    worlds = sample_worlds(family, count, seed)
    payload = [world.to_flat(family.schema).tolist() for world in worlds]
    (output / "worlds.json").write_text(json.dumps(payload, indent=2))
    manifest = {
        "command": "sample",
        "arguments": {"level": level, "count": count},
        "config_hash": _config_hash(config_path),
        "schema_versions": {"world": family.schema.version, "family": family.family_version},
        "seed": seed,
        "package_versions": _versions(),
        "output_paths": ["worlds.json"],
        "output_content_hashes": _content_hashes(output, ["worlds.json"]),
        "rejection_counts": {"accepted": count, "rejected": 0},
        "wall_clock_duration_seconds": round(time.perf_counter() - started, 6),
        "output_directory": str(output.resolve()),
    }
    return _write_manifest(output, manifest)


def _demo_worlds(schema: WorldSchema) -> list[tuple[str, WorldCode]]:
    return [
        ("W0", schema.world()),
        ("W1", schema.world(vertical_acceleration={"a_y": -0.1})),
        ("W2", schema.world(horizontal_acceleration={"a_x": -0.06})),
    ]


def _posterior_belief(pre: np.ndarray, state: BallState) -> np.ndarray:
    height, width = pre.shape
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    center_x = state.x * (width - 1)
    center_y = (1.0 - state.y) * (height - 1)
    gaussian = np.exp(-((columns - center_x) ** 2 + (rows - center_y) ** 2) / 8.0)
    gaussian /= max(float(gaussian.max()), 1.0e-12)
    return np.clip(0.25 * pre + 0.75 * gaussian, 0.0, 1.0).astype(np.float32)


def build_demo(
    *, config_path: str | Path, seed: int, output: str | Path, force: bool, quick: bool = False
) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"configuration not found: {config_path}")
    output = Path(output)
    _prepare_output(output, force)
    schema = WorldSchema.v1()
    base_family = load_family(config_path, "W2")
    simulation = SimulationConfig(dt=0.1, steps=32, seed=seed)
    render_config = RenderConfig(
        width=32 if quick else 64,
        height=32 if quick else 64,
        ball_radius=1 if quick else 2,
        min_hidden_steps=3,
    )
    occluders = (Occluder(0.35, 0.55, 0.05, 0.95),)
    initial = BallState(0.12, 0.72, 0.25, 0.0)
    query_step = 20
    entry_grid = base_family.entry_grid
    if quick:
        entry_grid = type(entry_grid)(
            entry_grid.x_bounds,
            entry_grid.y_bounds,
            entry_grid.vx_bounds,
            entry_grid.vy_bounds,
            (2, 2, 2, 2),
        )

    records: list[BenchmarkRecord] = []
    worlds: list[WorldCode] = []
    reachability_results = []
    belief_events: list[BeliefEvent] = []
    visual_paths: list[Path] = []
    rejection_counts: dict[str, int] = {"accepted": 0}
    records_root = output / "records"
    for level, world in _demo_worlds(schema):
        worlds.append(world)
        trajectory = rollout(initial, world, simulation)
        sequence = render_sequence(trajectory, occluders, render_config)
        observed = extract_reappearance(sequence)
        if observed is None:
            raise RuntimeError(f"demo world {level} did not produce an observable reappearance")
        reachability = enumerate_reachable(
            entry_grid,
            world,
            (query_step, observed.step),
            simulation,
            context_id="shared-occlusion-context",
        )
        reachability_results.append(reachability)
        valid_paths = [item for item in reachability.trajectories if item.valid]
        pre = superpose(valid_paths, np.ones(len(valid_paths)), render_config)
        actual_state = trajectory.states[observed.step]
        belief = BeliefEvent(observed.step, pre, _posterior_belief(pre, actual_state))
        belief_events.append(belief)
        record = BenchmarkRecord.from_components(
            record_id=f"example-{level}",
            family_level=level,
            schema=schema,
            world=world,
            trajectory=trajectory,
            sequence=sequence,
            reachability=reachability,
            belief=belief,
            rejection_reason=sequence.rejection_reason,
        )
        records.append(record)
        save_record(record, records_root)
        if sequence.rejection_reason is None:
            rejection_counts["accepted"] += 1
        else:
            reason = sequence.rejection_reason
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        visual_paths.extend(
            [
                render_world_card(world, schema, output / f"world-{level}.png").path,
                render_trajectory_strip(
                    trajectory, world, schema, output / f"trajectory-{level}.png"
                ).path,
                render_observation_strip(sequence, output / f"observation-{level}.png").path,
                render_belief_panel(
                    reachability,
                    belief,
                    render_config,
                    output / f"belief-{level}.png",
                    true_endpoint=(actual_state.x, actual_state.y),
                ).path,
            ]
        )

    reloaded = [load_record(records_root / record.record_id) for record in records]
    contact = render_dataset_contact_sheet(
        reloaded, schema, output / "world-family-contact-sheet.png"
    )
    visual_paths.append(contact.path)
    overlay = render_belief_panel(
        reachability_results[-1],
        belief_events[-1],
        render_config,
        output / "reachability-overlays.png",
        true_endpoint=(
            records[-1].trajectory().states[belief_events[-1].step].x,
            records[-1].trajectory().states[belief_events[-1].step].y,
        ),
    )
    visual_paths.append(overlay.path)

    probes: list[Probe] = []
    for result in reachability_results:
        endpoint = result.endpoints[query_step][len(result.endpoints[query_step]) // 2]
        probes.append(
            Probe("shared-occlusion-context", endpoint.x, endpoint.y, query_step, 0.025)
        )
    probes.extend(
        [
            Probe("shared-occlusion-context", 0.35, 0.45, query_step, 0.025),
            Probe("shared-occlusion-context", 0.55, 0.65, query_step, 0.025),
        ]
    )
    labels = build_label_matrix(
        list(range(len(worlds))),
        probes,
        labeler=lambda index, probe: reachability_results[index].label(probe),
        metadata={
            "operator_support": [level for level, _ in _demo_worlds(schema)],
            "uncertainty_grid": entry_grid.to_metadata(),
            "tolerance": 0.025,
            "search_complete": True,
        },
    )
    estimate = search_shattered_set(labels, max_n=3, budget=100, seed=seed)
    shattering_artifact = render_shattering_matrix(estimate, output / "shattering-matrix.png")
    visual_paths.append(shattering_artifact.path)
    np.savez_compressed(output / "labels.npz", values=labels.values)
    shattering_data = {
        "world_ids": list(labels.world_ids),
        "probes": [
            {
                "context_id": item.context_id,
                "x": item.x,
                "y": item.y,
                "step": item.step,
                "tolerance": item.tolerance,
            }
            for item in labels.probes
        ],
        "metadata": labels.metadata,
    }
    (output / "shattering-data.json").write_text(
        json.dumps(shattering_data, indent=2, sort_keys=True)
    )

    rasters = np.stack([
        rasterize_endpoints(result, render_config)[result.query_steps.index(query_step)]
        for result in reachability_results
    ])
    identifiability = group_equivalent_worlds(
        worlds, labels, tolerance=1.0e-8, possibility_rasters=rasters
    )
    equivalence_artifact = render_equivalence_gallery(
        identifiability, output / "equivalence-gallery.png"
    )
    visual_paths.append(equivalence_artifact.path)

    persistent_world = worlds[-1]
    persistent_events: list[BeliefEvent] = []
    observed_steps: list[int] = []
    for episode, initial_y in enumerate((0.49, 0.50, 0.51, 0.52)):
        episode_initial = BallState(initial.x, initial_y, initial.vx, initial.vy)
        episode_trajectory = rollout(episode_initial, persistent_world, simulation)
        episode_sequence = render_sequence(episode_trajectory, occluders, render_config)
        event = extract_reappearance(episode_sequence)
        if event is None:
            raise RuntimeError("persistent demo episode did not produce a reappearance")
        episode_reachability = enumerate_reachable(
            entry_grid,
            persistent_world,
            (event.step,),
            simulation,
            context_id="persistent-W2-context",
        )
        paths = [item for item in episode_reachability.trajectories if item.valid]
        pre = superpose(paths, np.ones(len(paths)), render_config)
        state = episode_trajectory.states[event.step]
        persistent_events.append(
            BeliefEvent(
                episode * (simulation.steps + 1) + event.step,
                pre,
                _posterior_belief(pre, state),
            )
        )
        observed_steps.append(event.step)
    history = build_belief_history(persistent_events, length=4)
    model2 = Model2InputRecord.from_mapping(
        {
            "pre_belief": history.tensor[:, 0],
            "post_belief": history.tensor[:, 1],
            "belief_difference": history.tensor[:, 2],
            "entropy": belief_entropy(history.tensor[:, 0]),
            "history_mask": history.mask,
            "target_world": worlds[-1].to_flat(schema),
        }
    )
    np.savez_compressed(
        output / "model2-input.npz",
        pre_belief=model2.pre_belief,
        post_belief=model2.post_belief,
        belief_difference=model2.belief_difference,
        entropy=model2.entropy,
        history_mask=model2.history_mask,
        target_world=model2.target_world,
    )
    _html_index(output, "Perceptual coherence environment demo", visual_paths)

    output_paths = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "command": "demo",
        "arguments": {"quick": quick},
        "config_hash": _config_hash(config_path),
        "schema_versions": {"world": schema.version, "family": base_family.family_version},
        "seed": seed,
        "package_versions": _versions(),
        "output_paths": output_paths,
        "output_content_hashes": _content_hashes(output, output_paths),
        "rejection_counts": rejection_counts,
        "shattering": {
            **estimate.metadata,
            "metric_name": estimate.metric_name,
            "dimension": estimate.dimension,
        },
        "model2_history": {
            "world_level": "W2",
            "event_count": len(persistent_events),
            "all_events_observed": True,
            "event_steps": observed_steps,
        },
        "wall_clock_duration_seconds": round(time.perf_counter() - started, 6),
        "output_directory": str(output.resolve()),
    }
    return _write_manifest(output, manifest)


def inspect_run(*, run: str | Path, output: str | Path, force: bool) -> dict[str, Any]:
    started = time.perf_counter()
    run = Path(run)
    output = Path(output)
    if not (run / "manifest.json").exists():
        raise FileNotFoundError(f"saved run manifest not found: {run}")
    _validate_source_output(run, output)
    _prepare_output(output, force)
    record_paths = sorted(path for path in (run / "records").iterdir() if path.is_dir())
    records = [load_record(path) for path in record_paths]
    artifact = render_dataset_contact_sheet(
        records, WorldSchema.v1(), output / "world-family-contact-sheet.png"
    )
    _html_index(output, "Saved-run inspection", [artifact.path])
    source = json.loads((run / "manifest.json").read_text())
    manifest = {
        "command": "inspect",
        "arguments": {"source_hash": source["deterministic_hash"]},
        "config_hash": source["config_hash"],
        "schema_versions": source["schema_versions"],
        "seed": source["seed"],
        "package_versions": _versions(),
        "output_paths": ["index.html", "world-family-contact-sheet.png"],
        "output_content_hashes": _content_hashes(
            output, ["index.html", "world-family-contact-sheet.png"]
        ),
        "rejection_counts": source["rejection_counts"],
        "wall_clock_duration_seconds": round(time.perf_counter() - started, 6),
        "output_directory": str(output.resolve()),
    }
    return _write_manifest(output, manifest)


def estimate_saved_run(
    *, run: str | Path, output: str | Path, force: bool, seed: int
) -> dict[str, Any]:
    started = time.perf_counter()
    run = Path(run)
    output = Path(output)
    required = (run / "manifest.json", run / "labels.npz", run / "shattering-data.json")
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"saved run data not found: {missing[0]}")
    _validate_source_output(run, output)
    _prepare_output(output, force)
    with np.load(run / "labels.npz", allow_pickle=False) as archive:
        values = archive["values"]
    payload = json.loads((run / "shattering-data.json").read_text())
    labels = LabelMatrix(
        values=values,
        world_ids=tuple(payload["world_ids"]),
        probes=tuple(Probe(**item) for item in payload["probes"]),
        metadata=payload["metadata"],
    )
    estimate = search_shattered_set(labels, max_n=3, budget=100, seed=seed)
    artifact = render_shattering_matrix(estimate, output / "shattering-matrix.png")
    _html_index(output, "Shattering estimate", [artifact.path])
    manifest = {
        "command": "estimate-shattering",
        "arguments": {"source": str(run)},
        "config_hash": json.loads((run / "manifest.json").read_text())["config_hash"],
        "schema_versions": {"world": "v1", "family": "v1"},
        "seed": seed,
        "package_versions": _versions(),
        "output_paths": ["index.html", "shattering-matrix.png"],
        "output_content_hashes": _content_hashes(
            output, ["index.html", "shattering-matrix.png"]
        ),
        "rejection_counts": {},
        "shattering": {
            **estimate.metadata,
            "metric_name": estimate.metric_name,
            "dimension": estimate.dimension,
        },
        "wall_clock_duration_seconds": round(time.perf_counter() - started, 6),
        "output_directory": str(output.resolve()),
    }
    return _write_manifest(output, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="empathy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="sample a versioned world family")
    sample.add_argument("--config", required=True)
    sample.add_argument("--level", default="W2")
    sample.add_argument("--count", type=int, default=8)
    sample.add_argument("--seed", type=int, default=7)
    sample.add_argument("--output", required=True)
    sample.add_argument("--force", action="store_true")

    inspect = subparsers.add_parser("inspect", help="recreate visuals from a saved run")
    inspect.add_argument("--run", required=True)
    inspect.add_argument("--output", required=True)
    inspect.add_argument("--force", action="store_true")

    estimate = subparsers.add_parser("estimate-shattering", help="rerun shattering search")
    estimate.add_argument("--run", required=True)
    estimate.add_argument("--output", required=True)
    estimate.add_argument("--seed", type=int, default=7)
    estimate.add_argument("--force", action="store_true")

    demo = subparsers.add_parser("demo", help="build the complete visual benchmark")
    demo.add_argument("--config", required=True)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output", required=True)
    demo.add_argument("--quick", action="store_true")
    demo.add_argument("--force", action="store_true")

    subparsers.add_parser("model1", help="train and inspect the trajectory-belief model")
    subparsers.add_parser("model2", help="train and inspect the inverse-world model")

    online = subparsers.add_parser(
        "train-seeing", help="watch the Seeing model learn online from repeated throws"
    )
    online.add_argument("--episodes", type=int, default=50)
    online.add_argument("--heldout-episodes", type=int, default=0)
    online.add_argument("--log-dir", default="runs/seeing")
    online.add_argument("--seed", type=int, default=7)

    live = subparsers.add_parser("live-demo", help="serve the interactive learned-inference demo")
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--port", type=int, default=8765)
    live.add_argument("--root", default=".")
    live.add_argument(
        "--model1-checkpoint",
        default="checkpoints/model1.pt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["live-demo"]
    if arguments and arguments[0] == "model1":
        from .model1.cli import main as model1_main

        return model1_main(arguments[1:])
    if arguments and arguments[0] == "model2":
        from .model2.cli import main as model2_main

        return model2_main(arguments[1:])
    parser = _parser()
    args = parser.parse_args(arguments)
    try:
        if args.command == "sample":
            sample_run(
                config_path=args.config,
                level=args.level,
                count=args.count,
                seed=args.seed,
                output=args.output,
                force=args.force,
            )
        elif args.command == "inspect":
            inspect_run(run=args.run, output=args.output, force=args.force)
        elif args.command == "estimate-shattering":
            estimate_saved_run(run=args.run, output=args.output, force=args.force, seed=args.seed)
        elif args.command == "demo":
            build_demo(
                config_path=args.config,
                seed=args.seed,
                output=args.output,
                force=args.force,
                quick=args.quick,
            )
        elif args.command == "live-demo":
            from .live.server import run_live_server

            run_live_server(
                host=args.host,
                port=args.port,
                root=Path(args.root),
                model1_checkpoint=Path(args.model1_checkpoint),
            )
        elif args.command == "train-seeing":
            from .model1.online_cli import run_heldout_training, run_online_training

            if args.heldout_episodes:
                run_heldout_training(
                    source_episodes=args.episodes,
                    heldout_episodes=args.heldout_episodes,
                    log_dir=Path(args.log_dir),
                    seed=args.seed,
                )
            else:
                run_online_training(
                    episodes=args.episodes,
                    log_dir=Path(args.log_dir),
                    seed=args.seed,
                )
    except (OSError, ValueError, KeyError) as error:
        print(f"empathy: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

