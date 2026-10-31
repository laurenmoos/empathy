"""Train-only baselines, kept outside production inference."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from empathy.physics.world import WorldSchema

from .config import TrainingConfig
from .contracts import SchemaLayout, WorldPrediction, WorldTargets
from .data import BeliefDataset
from .train import train_model


def fit_constant_baseline(train_targets: WorldTargets) -> WorldPrediction:
    layout = SchemaLayout.from_schema(WorldSchema.v1())
    flat = train_targets.world_flat
    active = flat[:, layout.activity_indices]
    probability = np.clip(active.mean(0), 1e-6, 1 - 1e-6)
    support = (
        active[:, layout.parameter_to_slot].astype(bool) & train_targets.parameter_identifiable
    )
    parameters = np.full(7, 0.5, np.float32)
    for index in range(7):
        if support[:, index].any():
            parameters[index] = flat[support[:, index], layout.parameter_indices[index]].mean()
    return WorldPrediction(
        torch.tensor(np.log(probability / (1 - probability)))[None], torch.tensor(parameters)[None]
    )


def run_controls(dataset_root: Path, config: TrainingConfig, output: Path) -> dict[str, Path]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    dataset = BeliefDataset(dataset_root, "train")
    indices = dataset.indices
    prediction = fit_constant_baseline(
        WorldTargets(
            dataset.targets.world_flat[indices], dataset.targets.parameter_identifiable[indices]
        )
    )
    output.mkdir(parents=True)
    constant = output / "constant.json"
    constant.write_text(
        json.dumps(
            {
                "architecture": "constant-world-baseline-v1",
                "dataset_hash": dataset.manifest["dataset_hash"],
                "fit_row_indices": indices.tolist(),
                "config": asdict(config),
                "activity_logits": prediction.activity_logits.tolist(),
                "normalized_parameters": prediction.normalized_parameters.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    results = {"constant": constant}
    for mode in ("single_belief", "shuffled_history"):
        results[mode] = train_model(
            dataset_root, replace(config, input_mode=mode), output / mode
        ).best_checkpoint
    return results
