"""Checkpoint evaluation and durable, visualization-ready prediction records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Model1Config
from .data import Model1ExampleRecord, collate_model1
from .metrics import (
    belief_metrics,
    calibration_report,
    displacement_metrics,
    mixture_nll,
)
from .rasterize import rasterize_mixture
from .train import load_checkpoint
from .types import TrajectoryMixture


@dataclass(frozen=True, eq=False)
class Model1PredictionRecord:
    example_id: str
    family_level: str
    seed: int
    metrics: dict[str, float]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class EvaluationSummary:
    count: int
    overall: dict[str, float]
    per_family: dict[str, dict[str, float | int]]
    calibration: dict[str, dict[str, Any]]
    config_hash: str
    checkpoint_hash: str
    seed: int
    record_paths: tuple[Path, ...]

    def to_payload(self, root: Path | None = None) -> dict[str, Any]:
        paths = [
            str(path if root is None else path.relative_to(root))
            for path in self.record_paths
        ]
        return {
            "count": self.count,
            "overall": self.overall,
            "per_family": self.per_family,
            "calibration": self.calibration,
            "config_hash": self.config_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "seed": self.seed,
            "record_paths": paths,
        }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config_hash(config: Model1Config) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_prediction_record(record: Model1PredictionRecord, root: Path) -> Path:
    path = Path(root) / record.example_id
    path.mkdir(parents=True, exist_ok=False)
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "example_id": record.example_id,
                "family_level": record.family_level,
                "seed": record.seed,
                "metrics": record.metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    np.savez_compressed(path / "arrays.npz", **record.arrays)
    return path


def load_prediction_record(path: Path) -> Model1PredictionRecord:
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    with np.load(path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for values in arrays.values():
        values.flags.writeable = False
    return Model1PredictionRecord(arrays=arrays, **metadata)


def _prediction_arrays(prediction: TrajectoryMixture) -> tuple[np.ndarray, np.ndarray]:
    return (
        prediction.positions[0].detach().cpu().numpy().astype(np.float32),
        prediction.weights[0].detach().cpu().numpy().astype(np.float32),
    )


def _row_metrics(
    pre: TrajectoryMixture,
    post: TrajectoryMixture,
    batch,
    pre_belief,
    post_belief,
    sigma: float,
) -> dict[str, float]:
    pre_displacement = displacement_metrics(pre, batch.target_positions, batch.target_mask)
    post_displacement = displacement_metrics(post, batch.target_positions, batch.target_mask)
    pre_belief_values = belief_metrics(pre_belief.collapsed, batch.oracle_pre_belief)
    post_belief_values = belief_metrics(post_belief.collapsed, batch.actual_path_belief)
    values = {
        "pre_min_ade": float(pre_displacement.min_ade[0]),
        "post_min_ade": float(post_displacement.min_ade[0]),
        "pre_min_fde": float(pre_displacement.min_fde[0]),
        "post_min_fde": float(post_displacement.min_fde[0]),
        "pre_top1_ade": float(pre_displacement.top1_ade[0]),
        "post_top1_ade": float(post_displacement.top1_ade[0]),
        "pre_nll": float(
            mixture_nll(pre, batch.target_positions, batch.target_mask, sigma=sigma)[0]
        ),
        "post_nll": float(
            mixture_nll(post, batch.target_positions, batch.target_mask, sigma=sigma)[0]
        ),
        "pre_belief_iou": float(pre_belief_values["soft_iou"][0]),
        "post_belief_iou": float(post_belief_values["soft_iou"][0]),
        "pre_belief_bce": float(pre_belief_values["bce"][0]),
        "post_belief_bce": float(post_belief_values["bce"][0]),
        "pre_confidence": float(pre.weights[0].max()),
        "post_confidence": float(post.weights[0].max()),
    }
    values.update(
        {
            "min_ade_improvement": values["pre_min_ade"] - values["post_min_ade"],
            "min_fde_improvement": values["pre_min_fde"] - values["post_min_fde"],
            "nll_improvement": values["pre_nll"] - values["post_nll"],
            "belief_iou_improvement": values["post_belief_iou"]
            - values["pre_belief_iou"],
            "belief_bce_improvement": values["pre_belief_bce"]
            - values["post_belief_bce"],
        }
    )
    return values


def _mean(records: Sequence[Model1PredictionRecord]) -> dict[str, float]:
    return {
        name: float(np.mean([record.metrics[name] for record in records]))
        for name in records[0].metrics
    }


def _calibration_payload(report) -> dict[str, Any]:
    return {
        "ece": float(report.ece),
        "bin_confidence": list(report.bin_confidence),
        "bin_accuracy": list(report.bin_accuracy),
        "bin_count": list(report.bin_count),
        "credible_coverage": report.credible_coverage,
    }


def evaluate_model1(
    checkpoint: Path,
    dataset: Sequence[Model1ExampleRecord] | torch.utils.data.Dataset,
    output: Path,
    *,
    device: torch.device,
    seed: int,
) -> EvaluationSummary:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    config: Model1Config = payload["config"]
    model, _ = load_checkpoint(checkpoint, expected_config=config, device=device)
    output.mkdir(parents=True)
    records_root = output / "records"
    records_root.mkdir()
    records: list[Model1PredictionRecord] = []
    paths: list[Path] = []
    pre_predictions: list[TrajectoryMixture] = []
    post_predictions: list[TrajectoryMixture] = []
    target_positions, target_masks = [], []
    with torch.inference_mode():
        for index in range(len(dataset)):
            source = dataset[index]
            batch = collate_model1([source], config).to(device)
            pre = model(batch.pre_frames, batch.pre_times, batch.pre_mask)
            post = model(batch.post_frames, batch.post_times, batch.post_mask)
            pre_belief = rasterize_mixture(
                pre,
                height=config.dataset.frame_size,
                width=config.dataset.frame_size,
                step_mask=batch.target_mask,
            )
            post_belief = rasterize_mixture(
                post,
                height=config.dataset.frame_size,
                width=config.dataset.frame_size,
                step_mask=batch.target_mask,
            )
            metrics = _row_metrics(
                pre,
                post,
                batch,
                pre_belief,
                post_belief,
                config.evaluation.mixture_sigma,
            )
            pre_positions, pre_weights = _prediction_arrays(pre)
            post_positions, post_weights = _prediction_arrays(post)
            arrays = {
                "pre_frames": batch.pre_frames[0].cpu().numpy(),
                "pre_times": batch.pre_times[0].cpu().numpy(),
                "pre_mask": batch.pre_mask[0].cpu().numpy(),
                "post_frames": batch.post_frames[0].cpu().numpy(),
                "post_times": batch.post_times[0].cpu().numpy(),
                "post_mask": batch.post_mask[0].cpu().numpy(),
                "target_positions": batch.target_positions[0].cpu().numpy(),
                "target_mask": batch.target_mask[0].cpu().numpy(),
                "pre_positions": pre_positions,
                "pre_weights": pre_weights,
                "post_positions": post_positions,
                "post_weights": post_weights,
                "pre_time_belief": pre_belief.time_belief[0].cpu().numpy(),
                "post_time_belief": post_belief.time_belief[0].cpu().numpy(),
                "pre_belief": pre_belief.collapsed[0].cpu().numpy(),
                "post_belief": post_belief.collapsed[0].cpu().numpy(),
                "belief_difference": (
                    post_belief.collapsed[0] - pre_belief.collapsed[0]
                ).cpu().numpy(),
                "oracle_pre_belief": batch.oracle_pre_belief[0].cpu().numpy(),
                "actual_path_belief": batch.actual_path_belief[0].cpu().numpy(),
            }
            record = Model1PredictionRecord(
                source.example_id, source.family_level, source.seed, metrics, arrays
            )
            records.append(record)
            paths.append(save_prediction_record(record, records_root))
            pre_predictions.append(pre)
            post_predictions.append(post)
            target_positions.append(batch.target_positions)
            target_masks.append(batch.target_mask)
    if not records:
        raise ValueError("evaluation dataset must not be empty")
    combined_targets = torch.cat(target_positions)
    combined_masks = torch.cat(target_masks)

    def combine(items: list[TrajectoryMixture]) -> TrajectoryMixture:
        return TrajectoryMixture(
            torch.cat([item.positions for item in items]),
            torch.cat([item.logits for item in items]),
            torch.cat([item.entry_state for item in items]),
        )

    pre_calibration = calibration_report(
        combine(pre_predictions),
        combined_targets,
        combined_masks,
        radius=config.evaluation.calibration_radius,
        bins=config.evaluation.calibration_bins,
    )
    post_calibration = calibration_report(
        combine(post_predictions),
        combined_targets,
        combined_masks,
        radius=config.evaluation.calibration_radius,
        bins=config.evaluation.calibration_bins,
    )
    families = sorted({record.family_level for record in records})
    per_family = {}
    for family in families:
        selected = [record for record in records if record.family_level == family]
        per_family[family] = {"count": len(selected), **_mean(selected)}
    summary = EvaluationSummary(
        count=len(records),
        overall=_mean(records),
        per_family=per_family,
        calibration={
            "pre": _calibration_payload(pre_calibration),
            "post": _calibration_payload(post_calibration),
        },
        config_hash=_config_hash(config),
        checkpoint_hash=_file_hash(checkpoint),
        seed=seed,
        record_paths=tuple(paths),
    )
    (output / "summary.json").write_text(
        json.dumps(summary.to_payload(output), indent=2, sort_keys=True)
    )
    return summary


def load_evaluation_summary(path: Path) -> EvaluationSummary:
    root = Path(path)
    payload = json.loads((root / "summary.json").read_text())
    payload["record_paths"] = tuple(root / item for item in payload["record_paths"])
    return EvaluationSummary(**payload)
