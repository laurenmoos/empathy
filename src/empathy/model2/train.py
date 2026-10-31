"""Deterministic offline fitting and checkpointed belief-only inference."""

from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from empathy.physics.world import WorldSchema

from . import INPUT_SCHEMA_VERSION
from .config import TrainingConfig
from .contracts import BeliefInputs, WorldPrediction
from .data import BeliefDataset, array_hash, collate_examples
from .losses import world_code_loss
from .network import InverseWorldCNN

ARCHITECTURE = "belief-history-cnn-v1"


@dataclass(frozen=True)
class TrainingRun:
    root: Path
    best_checkpoint: Path
    metrics_path: Path
    manifest_path: Path


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("requested MPS device is unavailable")
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("unknown device")
    return torch.device(name)


def _state_hash(model: InverseWorldCNN) -> str:
    return array_hash(
        {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
    )


def _provenance(training: BeliefDataset, validation: BeliefDataset, device: torch.device) -> dict:
    manifest = training.manifest
    rows = manifest.get("rows", [])
    worlds = manifest.get("world_ids", [])
    result = {
        "dataset_hash": manifest["dataset_hash"],
        "input_hash": manifest["input_hash"],
        "target_hash": manifest["target_hash"],
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.machine(),
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    for name, dataset in (("train", training), ("validation", validation)):
        indices = dataset.indices.tolist()
        result[f"{name}_row_indices"] = indices
        result[f"{name}_row_ids"] = [
            rows[index].get("episode_id", f"row-{index}") if index < len(rows) else f"row-{index}"
            for index in indices
        ]
        result[f"{name}_world_ids"] = [
            worlds[index] if index < len(worlds) else None for index in indices
        ]
    return result


def _measure(model, loader, device, config) -> dict[str, float | int]:
    model.eval()
    activity_sum, parameter_sum, activity_count, parameter_count = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch.beliefs.to(device), batch.history_mask.to(device))
            result = world_code_loss(
                prediction, batch.world_flat.to(device), batch.parameter_identifiable.to(device)
            )
            count = len(batch.beliefs) * 5
            activity_sum += result.activity.item() * count
            parameter_sum += result.parameters.item() * result.parameter_count
            activity_count += count
            parameter_count += result.parameter_count
    activity = activity_sum / activity_count
    parameters = parameter_sum / max(parameter_count, 1)
    return {
        "total": config.activity_weight * activity + config.parameter_weight * parameters,
        "activity": activity,
        "parameters": parameters,
        "activity_sum": activity_sum,
        "parameter_sum": parameter_sum,
        "activity_count": activity_count,
        "parameter_count": parameter_count,
    }


def train_model(dataset_root: Path, config: TrainingConfig, output: Path) -> TrainingRun:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    device = _device(config.device)
    training = BeliefDataset(dataset_root, "train")
    validation = BeliefDataset(dataset_root, "validation")
    started = time.perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    if device.type == "cpu":
        torch.set_num_threads(1)
    model = InverseWorldCNN(input_mode=config.input_mode, control_seed=config.seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    fit_loader = DataLoader(
        training,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_examples,
        num_workers=0,
    )
    train_loader = DataLoader(
        training, batch_size=config.batch_size, collate_fn=collate_examples, num_workers=0
    )
    validation_loader = DataLoader(
        validation, batch_size=config.batch_size, collate_fn=collate_examples, num_workers=0
    )
    initial = _measure(model, train_loader, device, config)
    initial_hash = _state_hash(model)
    provenance = _provenance(training, validation, device)
    output.mkdir(parents=True)
    epochs, best, best_epoch = [], float("inf"), -1
    checkpoint = {}
    for epoch in range(config.epochs):
        model.train()
        for batch in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.beliefs.to(device), batch.history_mask.to(device))
            result = world_code_loss(
                prediction,
                batch.world_flat.to(device),
                batch.parameter_identifiable.to(device),
                activity_weight=config.activity_weight,
                parameter_weight=config.parameter_weight,
            )
            result.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        train_metrics = _measure(model, train_loader, device, config)
        validation_metrics = _measure(model, validation_loader, device, config)
        epochs.append(
            {
                "epoch": epoch,
                "learning_rate": config.learning_rate,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"validation_{k}": v for k, v in validation_metrics.items()},
            }
        )
        improved = validation_metrics["total"] < best
        if improved:
            best, best_epoch = validation_metrics["total"], epoch
        checkpoint = {
            "architecture": ARCHITECTURE,
            "input_schema": INPUT_SCHEMA_VERSION,
            "world_schema": "v1",
            "field_order": list(WorldSchema.v1().field_order),
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "dataset_hash": training.manifest["dataset_hash"],
            "epoch": epoch,
            "best_validation_loss": best,
            "selected_epoch": best_epoch,
            "validation_loss": validation_metrics["total"],
            "provenance": provenance,
            "torch_version": str(torch.__version__),
            "state_hash": _state_hash(model),
        }
        if improved:
            torch.save(checkpoint, output / "best.pt")
    torch.save(checkpoint, output / "final.pt")
    metrics = {
        "initial_train_loss": initial["total"],
        "selected_epoch": best_epoch,
        "epochs": epochs,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    manifest = {
        "architecture": ARCHITECTURE,
        "input_schema": INPUT_SCHEMA_VERSION,
        "world_schema": "v1",
        "dataset_hash": training.manifest["dataset_hash"],
        "config": asdict(config),
        "selected_epoch": best_epoch,
        "best_validation_loss": best,
        "provenance": provenance,
        "train_row_indices": training.indices.tolist(),
        "validation_row_indices": validation.indices.tolist(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "initial_state_hash": initial_hash,
        "final_state_hash": _state_hash(model),
        "wall_clock_duration_seconds": time.perf_counter() - started,
        "output_paths": ["best.pt", "final.pt", "metrics.json"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return TrainingRun(
        output, output / "best.pt", output / "metrics.json", output / "manifest.json"
    )


def load_model(
    checkpoint: Path, *, device: str = "cpu", expected_dataset_hash: str | None = None
) -> InverseWorldCNN:
    target_device = _device(device)
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if (
        payload.get("architecture") != ARCHITECTURE
        or payload.get("input_schema") != INPUT_SCHEMA_VERSION
        or payload.get("world_schema") != "v1"
        or payload.get("field_order") != list(WorldSchema.v1().field_order)
    ):
        raise ValueError("checkpoint architecture or schema mismatch")
    if expected_dataset_hash is not None and payload.get("dataset_hash") != expected_dataset_hash:
        raise ValueError("checkpoint dataset hash mismatch")
    config = TrainingConfig(**payload["config"])
    model = InverseWorldCNN(input_mode=config.input_mode, control_seed=config.seed)
    model.load_state_dict(payload["state_dict"], strict=True)
    if _state_hash(model) != payload.get("state_hash"):
        raise ValueError("checkpoint state hash mismatch")
    return model.to(target_device).eval()


def predict(
    model: InverseWorldCNN, inputs: BeliefInputs, *, batch_size: int = 32
) -> WorldPrediction:
    if not isinstance(inputs, BeliefInputs):
        raise TypeError("predict accepts only validated BeliefInputs")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    activities, parameters = [], []
    try:
        with torch.no_grad():
            for start in range(0, len(inputs.beliefs), batch_size):
                # Copy read-only NumPy storage; torch must not gain a mutable alias.
                beliefs = torch.tensor(inputs.beliefs[start : start + batch_size], device=device)
                mask = torch.tensor(inputs.history_mask[start : start + batch_size], device=device)
                prediction = model(beliefs, mask)
                activities.append(prediction.activity_logits.cpu())
                parameters.append(prediction.normalized_parameters.cpu())
    finally:
        model.train(was_training)
    return WorldPrediction(torch.cat(activities), torch.cat(parameters))
