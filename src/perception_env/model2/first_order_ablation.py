"""Explicit privileged diagnostic. Never imported by production training/inference."""

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig
from .contracts import WorldPrediction
from .data import BeliefDataset, array_hash
from .losses import world_code_loss
from .train import TrainingRun, _device, _provenance


class FirstOrderMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(40, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.activity = nn.Linear(128, 5)
        self.parameters_head = nn.Linear(128, 7)

    def forward(self, features, mask):
        values = torch.where(mask[:, :, None], features, 0).flatten(1)
        hidden = self.hidden(torch.cat((values, mask.float()), dim=1))
        return WorldPrediction(self.activity(hidden), self.parameters_head(hidden).sigmoid())


def train_first_order_control(
    dataset_root: Path, config: TrainingConfig, output: Path
) -> TrainingRun:
    root, output = Path(dataset_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    train, validation = BeliefDataset(root, "train"), BeliefDataset(root, "validation")
    with np.load(root / "privileged_features.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if array_hash(arrays) != train.manifest["privileged_features_hash"]:
        raise ValueError("privileged feature hash mismatch")
    if set(arrays) != {"features", "history_mask"} or arrays["features"].shape != (
        len(train.inputs.beliefs),
        4,
        9,
    ):
        raise ValueError("invalid privileged features")
    device = _device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    if device.type == "cpu":
        torch.set_num_threads(1)
    model = FirstOrderMLP().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    features = torch.tensor(arrays["features"], device=device)
    mask = torch.tensor(arrays["history_mask"], device=device)
    targets = torch.tensor(train.targets.world_flat, device=device)
    identifiable = torch.tensor(train.targets.parameter_identifiable, device=device)

    def state_hash():
        return array_hash(
            {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
        )

    def loss(indices):
        return world_code_loss(
            model(features[indices], mask[indices]),
            targets[indices],
            identifiable[indices],
            activity_weight=config.activity_weight,
            parameter_weight=config.parameter_weight,
        ).total

    initial_hash = state_hash()
    with torch.no_grad():
        initial_loss = loss(train.indices).item()
    output.mkdir(parents=True)
    rng = np.random.default_rng(config.seed)
    best, selected, epochs = float("inf"), -1, []
    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(train.indices)
        for start in range(0, len(order), config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            objective = loss(order[start : start + config.batch_size])
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss, validation_loss = (
                loss(train.indices).item(),
                loss(validation.indices).item(),
            )
        improved = validation_loss < best
        if improved:
            best, selected = validation_loss, epoch
        epochs.append(
            {"epoch": epoch, "train_total": train_loss, "validation_total": validation_loss}
        )
        payload = {
            "architecture": "privileged_first_order_ablation",
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "selected_epoch": selected,
            "best_validation_loss": best,
            "dataset_hash": train.manifest["dataset_hash"],
            "provenance": _provenance(train, validation, device),
            "state_hash": state_hash(),
        }
        if improved:
            torch.save(payload, output / "best.pt")
    torch.save(payload, output / "final.pt")
    (output / "metrics.json").write_text(
        json.dumps({"initial_train_loss": initial_loss, "epochs": epochs}, indent=2)
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "architecture": "privileged_first_order_ablation",
                "initial_state_hash": initial_hash,
                "final_state_hash": state_hash(),
                "dataset_hash": train.manifest["dataset_hash"],
                "config": asdict(config),
                "provenance": payload["provenance"],
                "privileged_features_hash": array_hash(arrays),
            },
            indent=2,
        )
    )
    return TrainingRun(
        output, output / "best.pt", output / "metrics.json", output / "manifest.json"
    )
