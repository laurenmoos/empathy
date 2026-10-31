"""Detached, held-out finite-oracle evaluation after belief-only inference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from empathy.physics.config import RenderConfig
from empathy.physics.world import WorldSchema
from empathy.reasoning.reachability import Probe, rasterize_endpoints
from empathy.reasoning.shattering import LabelMatrix, search_shattered_set

from .config import EvaluationConfig
from .contracts import BeliefInputs, SchemaLayout, WorldPrediction, WorldTargets
from .data import BeliefDataset, array_hash
from .decode import decode_prediction
from .fixtures import _reachability
from .train import load_model, predict


@dataclass(frozen=True)
class EvaluationResult:
    root: Path
    metrics: dict[str, object]
    prediction_path: Path
    manifest_path: Path


def _ratio(numerator, denominator):
    return {
        "value": float(numerator / denominator) if denominator else None,
        "denominator": int(denominator),
    }


def world_interval(values, seed=7):
    values = np.asarray(values, float)
    if len(values) < 2:
        return {"lower": None, "upper": None, "status": "insufficient_worlds"}
    samples = np.random.default_rng(seed).choice(values, (1000, len(values)), replace=True).mean(1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"lower": float(low), "upper": float(high), "status": "world_bootstrap_1000"}


def score_predictions(
    prediction: WorldPrediction, targets: WorldTargets, evaluator: Mapping[str, object]
) -> dict[str, object]:
    schema = WorldSchema.v1()
    layout = SchemaLayout.from_schema(schema)
    decoded = decode_prediction(prediction, schema)
    true = targets.world_flat
    if len(decoded.flat) != len(true):
        raise ValueError("prediction/target row mismatch")
    active = true[:, layout.activity_indices].astype(bool)
    recovered = decoded.flat[:, layout.activity_indices].astype(bool)
    support = active[:, layout.parameter_to_slot] & targets.parameter_identifiable
    statuses = evaluator.get("audit_statuses", ["unspecified"] * len(true))
    support &= np.array([s != "unresolved" for s in statuses])[:, None]
    error = np.abs(
        prediction.normalized_parameters.detach().cpu().numpy() - true[:, layout.parameter_indices]
    )
    ranges = np.array([p.maximum - p.minimum for spec in schema.specs for p in spec.parameters])
    labels = np.asarray(evaluator["true_labels"], bool)
    predicted = np.asarray(evaluator["predicted_labels"], bool)
    rasters = np.asarray(evaluator["true_rasters"], bool)
    recovered_rasters = np.asarray(evaluator["predicted_rasters"], bool)
    if labels.shape != predicted.shape or rasters.shape != recovered_rasters.shape:
        raise ValueError("oracle array shape mismatch")
    if len(labels) != len(true) or len(rasters) != len(true):
        raise ValueError("oracle row mismatch")
    agreement = (labels == predicted).reshape(len(true), -1).mean(1)
    union = (rasters | recovered_rasters).sum(axis=(-2, -1))
    intersection = (rasters & recovered_rasters).sum(axis=(-2, -1))
    iou = np.divide(intersection, union, out=np.ones(union.shape, float), where=union != 0)
    exact = np.isclose(decoded.flat, true, atol=1e-6, rtol=0).all(1)
    world_ids = list(evaluator["world_ids"])
    families = list(evaluator["family_levels"])
    per_world = {
        world: float(agreement[np.array(world_ids) == world].mean())
        for world in sorted(set(world_ids))
    }
    return {
        "activity_exact_match": float((active == recovered).all(1).mean()),
        "activity_precision": _ratio((active & recovered).sum(), recovered.sum()),
        "activity_recall": _ratio((active & recovered).sum(), active.sum()),
        "per_operator": {
            spec.name: {
                "precision": _ratio((active[:, i] & recovered[:, i]).sum(), recovered[:, i].sum()),
                "recall": _ratio((active[:, i] & recovered[:, i]).sum(), active[:, i].sum()),
            }
            for i, spec in enumerate(schema.specs)
        },
        "parameter_mae": _ratio(error[support].sum(), support.sum()),
        "physical_parameter_mae": _ratio((error * ranges)[support].sum(), support.sum()),
        "parameter_errors": [
            _ratio(error[:, i][support[:, i]].sum(), support[:, i].sum()) for i in range(7)
        ],
        "exact_code_match": float(exact.mean()),
        "projection_rate": float((decoded.projection_counts > 0).mean()),
        "possibility_agreement": float(agreement.mean()),
        "possibility_equivalent_rate": float((agreement == 1).mean()),
        "positive_recall": _ratio((labels & predicted).sum(), labels.sum()),
        "negative_recall": _ratio((~labels & ~predicted).sum(), (~labels).sum()),
        "raster_iou": float(iou.mean()),
        "empty_raster_pairs": int((union == 0).sum()),
        "empty_predicted_contexts": int((~recovered_rasters.any(axis=(-2, -1))).sum()),
        "audit_scope": "finite_sampled_comparison_bank",
        "audit_statuses": list(statuses),
        "per_example": [
            {
                "world_id": world_ids[i],
                "agreement": float(agreement[i]),
                "iou": float(iou[i].mean()),
                "exact_code": bool(exact[i]),
                "equivalent": bool(agreement[i] == 1),
            }
            for i in range(len(true))
        ],
        "per_world_agreement": per_world,
        "world_macro_agreement": float(np.mean(list(per_world.values()))),
        "agreement_interval": world_interval(list(per_world.values())),
        "family_macro_agreement": {
            family: float(
                np.mean(
                    [
                        per_world[w]
                        for w in set(world_ids[i] for i, f in enumerate(families) if f == family)
                    ]
                )
            )
            for family in sorted(set(families))
        },
    }


def _search(true_worlds, predicted_worlds, metadata, indices, config, world_ids):
    contexts = sorted(
        [
            (m["context_id"], m, probes)
            for i in indices
            for m, probes in zip(
                metadata["contexts"][i], metadata["heldout_probes"][i], strict=True
            )
        ],
        key=lambda item: item[0],
    )[:3]
    reports, saved = [], {}
    for number, (identity, context, raw_probes) in enumerate(contexts):
        probes = tuple(Probe(**p) for p in raw_probes)
        report = {
            "context_id": identity,
            "metric_name": "context_conditioned",
            "seeds": list(config.search_seeds),
            "budget": config.budget,
            "max_n": config.max_n,
            "entry_grid": context["entry_grid"],
            "simulation_config": context["simulation_config"],
            "probes": raw_probes,
            "world_sampling": "first_saved_episode_per_held_out_world",
            "world_ids": world_ids,
            "tolerance": metadata["tolerance"],
            "probe_sampling": "half_uniform_half_endpoint_disk_radius_0.05",
        }
        for name, worlds in [("true", true_worlds), ("recovered", predicted_worlds)]:
            results = [_reachability(world, context) for world in worlds]
            values = np.array([[result.label(p) for p in probes] for result in results], bool)
            matrix = LabelMatrix(values, tuple(map(str, range(len(worlds)))), probes, {})
            searches = [
                search_shattered_set(matrix, config.max_n, config.budget, seed, True)
                for seed in config.search_seeds
            ]
            report[name] = {
                "dimensions": [s.dimension for s in searches],
                "exhaustive": [s.exhaustive for s in searches],
                "distinct_label_rows": len(np.unique(values, axis=0)),
                "search_seed_min": min(s.dimension for s in searches),
                "search_seed_max": max(s.dimension for s in searches),
            }
            saved[f"shattering_{number}_{name}"] = values
        reports.append(report)
    return reports, saved


def evaluate_model(
    checkpoint: Path, dataset_root: Path, config: EvaluationConfig, output: Path
) -> EvaluationResult:
    output, checkpoint, dataset_root = Path(output), Path(checkpoint), Path(dataset_root)
    if output.exists():
        raise FileExistsError(output)
    dataset = BeliefDataset(dataset_root, "test")
    indices = dataset.indices
    inputs = BeliefInputs(dataset.inputs.beliefs[indices], dataset.inputs.history_mask[indices])
    targets = WorldTargets(
        dataset.targets.world_flat[indices], dataset.targets.parameter_identifiable[indices]
    )
    if checkpoint.suffix == ".json":
        payload = json.loads(checkpoint.read_text())
        if payload.get("architecture") != "constant-world-baseline-v1":
            raise ValueError("incompatible baseline architecture")
        if payload["dataset_hash"] != dataset.manifest["dataset_hash"]:
            raise ValueError("checkpoint dataset hash mismatch")
        if set(indices) & set(payload["fit_row_indices"]):
            raise ValueError("test split overlaps fitting rows")
        prediction = WorldPrediction(
            torch.tensor(payload["activity_logits"]).repeat(len(indices), 1),
            torch.tensor(payload["normalized_parameters"]).repeat(len(indices), 1),
        )
    else:
        model = load_model(checkpoint, expected_dataset_hash=dataset.manifest["dataset_hash"])
        prediction = predict(model, inputs)
        payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
        for name in ("train_row_indices", "validation_row_indices"):
            if set(indices) & set(payload["provenance"][name]):
                raise ValueError("test split overlaps fitting or selection rows")
    # Privileged context is first opened after forward inference has completed.
    raw_metadata = (dataset_root / "evaluation.json").read_text()
    if (
        hashlib.sha256(raw_metadata.encode()).hexdigest()
        != dataset.manifest["evaluation_metadata_hash"]
    ):
        raise ValueError("evaluation metadata hash mismatch")
    metadata = json.loads(raw_metadata)
    with np.load(dataset_root / "evaluation.npz", allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}
    if array_hash(source) != dataset.manifest["evaluation_hash"]:
        raise ValueError("evaluation array hash mismatch")
    schema = WorldSchema.v1()
    decoded = decode_prediction(prediction, schema)
    true_worlds = [schema.from_flat(row) for row in targets.world_flat]
    true_labels, predicted_labels, true_rasters, predicted_rasters = [], [], [], []
    for row, index in enumerate(indices):
        tl, pl, tr, pr = [], [], [], []
        for event, context in enumerate(metadata["contexts"][index]):
            probes = [Probe(**p) for p in metadata["heldout_probes"][index][event]]
            audit = {json.dumps(p, sort_keys=True) for p in metadata["audit_probes"][index][event]}
            if any(json.dumps(asdict(p), sort_keys=True) in audit for p in probes):
                raise ValueError("heldout probe overlaps audit probe")
            for world, labels, rasters in [
                (true_worlds[row], tl, tr),
                (decoded.worlds[row], pl, pr),
            ]:
                reachable = _reachability(world, context)
                labels.append([reachable.label(p) for p in probes])
                rasters.append(
                    rasterize_endpoints(
                        reachable,
                        RenderConfig(
                            width=inputs.beliefs.shape[-1], height=inputs.beliefs.shape[-2]
                        ),
                    )[0]
                )
        true_labels.append(tl)
        predicted_labels.append(pl)
        true_rasters.append(tr)
        predicted_rasters.append(pr)
    arrays = {
        "true_labels": np.asarray(true_labels, bool),
        "predicted_labels": np.asarray(predicted_labels, bool),
        "true_rasters": np.asarray(true_rasters, bool),
        "predicted_rasters": np.asarray(predicted_rasters, bool),
    }
    if not np.array_equal(arrays["true_labels"], source["heldout_labels"][indices]):
        raise ValueError("regenerated oracle labels disagree with saved truth")
    rows = [dataset.manifest["rows"][i] for i in indices]
    metrics = score_predictions(
        prediction,
        targets,
        {
            **arrays,
            "world_ids": [r["world_id"] for r in rows],
            "family_levels": [r["family_level"] for r in rows],
            "audit_statuses": [r["audit_status"] for r in rows],
        },
    )
    metrics["heldout_scope"] = metadata["heldout_scope"]
    bank_ids = sorted({row["world_id"] for row in rows})
    representatives = [
        next(i for i, row in enumerate(rows) if row["world_id"] == world) for world in bank_ids
    ]
    metrics["shattering"], search_arrays = _search(
        [true_worlds[i] for i in representatives],
        [decoded.worlds[i] for i in representatives],
        metadata,
        indices,
        config,
        bank_ids,
    )
    arrays.update(search_arrays)
    arrays.update(
        activity_logits=prediction.activity_logits.numpy(),
        normalized_parameters=prediction.normalized_parameters.numpy(),
        decoded_worlds=decoded.flat,
        true_worlds=targets.world_flat,
        parameter_identifiable=targets.parameter_identifiable,
        beliefs=inputs.beliefs,
        history_mask=inputs.history_mask,
        projection_counts=decoded.projection_counts,
    )
    manifest = {
        "dataset_hash": dataset.manifest["dataset_hash"],
        "checkpoint_hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_state_hash": payload.get("state_hash"),
        "architecture": payload["architecture"],
        "row_indices": indices.tolist(),
        "world_ids": [r["world_id"] for r in rows],
        "shattering_world_ids": bank_ids,
        "shattering_row_indices": indices[representatives].tolist(),
        "config": asdict(config),
        "seed": payload["config"]["seed"],
        "source": dataset.manifest["source"],
        "array_hash": array_hash(arrays),
        "heldout_scope": metadata["heldout_scope"],
        "fixture_config": dataset.manifest["fixture_config"],
        "family_config": dataset.manifest["family_config"],
    }
    output.mkdir(parents=True)
    np.savez_compressed(output / "predictions.npz", **arrays)
    metrics_json = json.dumps(metrics, indent=2, sort_keys=True)
    manifest["metrics_hash"] = hashlib.sha256(metrics_json.encode()).hexdigest()
    (output / "metrics.json").write_text(metrics_json)
    history = checkpoint.parent / "metrics.json"
    if history.exists() and checkpoint.suffix != ".json":
        training_json = history.read_text()
        manifest["training_hash"] = hashlib.sha256(training_json.encode()).hexdigest()
        (output / "training.json").write_text(training_json)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return EvaluationResult(output, metrics, output / "predictions.npz", output / "manifest.json")
