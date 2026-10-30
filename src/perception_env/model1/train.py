"""Deterministic CPU training and portable checkpoints for Model 1."""

from __future__ import annotations

import importlib.metadata
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .config import Model1Config
from .data import Model1ExampleRecord, collate_model1
from .losses import Model1LossTerms, model1_loss
from .model import Model1
from .rasterize import rasterize_mixture


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: str
    epoch: int
    seed: int
    best_validation_loss: float
    package_versions: dict[str, str]


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[dict[str, float], ...]
    best_checkpoint: Path
    last_checkpoint: Path


def seed_model1(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _versions() -> dict[str, str]:
    result = {"torch": torch.__version__, "numpy": np.__version__}
    try:
        result["perception-env"] = importlib.metadata.version("perception-env")
    except importlib.metadata.PackageNotFoundError:
        result["perception-env"] = "unknown"
    return result


def _architecture_key(config: Model1Config) -> tuple[object, ...]:
    return (
        config.schema_version,
        config.dataset.frame_size,
        config.dataset.clip_steps,
        config.dataset.prediction_steps,
        config.model,
    )


def save_checkpoint(
    path: Path,
    model: Model1,
    optimizer: torch.optim.Optimizer,
    config: Model1Config,
    *,
    epoch: int,
    seed: int,
    best_validation_loss: float,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = CheckpointMetadata(
        schema_version=config.schema_version,
        epoch=epoch,
        seed=seed,
        best_validation_loss=float(best_validation_loss),
        package_versions=_versions(),
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "config_dict": asdict(config),
            "metadata": asdict(metadata),
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path,
    *,
    expected_config: Model1Config | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Model1, CheckpointMetadata]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config: Model1Config = payload["config"]
    if expected_config is not None and _architecture_key(config) != _architecture_key(
        expected_config
    ):
        raise ValueError("checkpoint configuration does not match expected configuration")
    model = Model1(config.model, prediction_steps=config.dataset.prediction_steps).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, CheckpointMetadata(**payload["metadata"])


def _run_batch(
    model: Model1,
    batch,
    config: Model1Config,
) -> Model1LossTerms:
    pre_prediction = model(batch.pre_frames, batch.pre_times, batch.pre_mask)
    post_prediction = model(batch.post_frames, batch.post_times, batch.post_mask)
    pre_belief = rasterize_mixture(
        pre_prediction,
        height=config.dataset.frame_size,
        width=config.dataset.frame_size,
    )
    post_belief = rasterize_mixture(
        post_prediction,
        height=config.dataset.frame_size,
        width=config.dataset.frame_size,
    )
    return model1_loss(
        pre_prediction,
        post_prediction,
        pre_belief,
        post_belief,
        batch,
        config.loss,
    )


def _average(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([item[key] for item in values]))
        for key in values[0]
    }


def _term_values(terms: Model1LossTerms) -> dict[str, float]:
    return {
        "total": float(terms.total.detach()),
        "trajectory": float(terms.trajectory.detach()),
        "mode": float(terms.mode.detach()),
        "state": float(terms.state.detach()),
        "pre_belief": float(terms.pre_belief.detach()),
        "post_belief": float(terms.post_belief.detach()),
    }


def train_model1(
    config: Model1Config,
    train_dataset: Sequence[Model1ExampleRecord] | torch.utils.data.Dataset,
    validation_dataset: Sequence[Model1ExampleRecord] | torch.utils.data.Dataset,
    output: Path,
    *,
    device: torch.device,
    seed: int,
) -> TrainingResult:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    generator = seed_model1(seed)
    collate = partial(collate_model1, config=config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model = Model1(config.model, prediction_steps=config.dataset.prediction_steps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_path = output / "best.pt"
    last_path = output / "last.pt"
    history_path = output / "history.jsonl"
    for epoch in range(config.training.epochs):
        model.train()
        train_values: list[dict[str, float]] = []
        for raw_batch in train_loader:
            batch = raw_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            terms = _run_batch(model, batch, config)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            train_values.append(_term_values(terms))
        model.eval()
        validation_values: list[dict[str, float]] = []
        with torch.inference_mode():
            for raw_batch in validation_loader:
                terms = _run_batch(model, raw_batch.to(device), config)
                validation_values.append(_term_values(terms))
        train_average = _average(train_values)
        validation_average = _average(validation_values)
        row = {f"train_{key}": value for key, value in train_average.items()}
        row.update({f"validation_{key}": value for key, value in validation_average.items()})
        row["epoch"] = float(epoch)
        history.append(row)
        with history_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if validation_average["total"] < best_loss:
            best_loss = validation_average["total"]
            save_checkpoint(
                best_path,
                model,
                optimizer,
                config,
                epoch=epoch,
                seed=seed,
                best_validation_loss=best_loss,
            )
    save_checkpoint(
        last_path,
        model,
        optimizer,
        config,
        epoch=config.training.epochs - 1,
        seed=seed,
        best_validation_loss=best_loss,
    )
    return TrainingResult(tuple(history), best_path, last_path)
