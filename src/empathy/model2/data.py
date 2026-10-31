"""Array-only dataset storage; evaluator truth is never opened by the loader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import INPUT_SCHEMA_VERSION
from .contracts import BeliefInputs, Model2Batch, WorldTargets


def array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, values in sorted(arrays.items()):
        array = np.ascontiguousarray(values)
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def manifest_hash(manifest: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "dataset_hash"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def save_dataset(
    root: Path, inputs: BeliefInputs, targets: WorldTargets, split: np.ndarray, metadata: dict
) -> dict:
    root = Path(root)
    split = np.asarray(split)
    if split.shape != (len(inputs.beliefs),) or not np.isin(split, [0, 1, 2]).all():
        raise ValueError("split must contain one 0/1/2 assignment per row")
    if len(targets.world_flat) != len(split):
        raise ValueError("input and target row counts differ")
    input_arrays = {"beliefs": inputs.beliefs, "history_mask": inputs.history_mask}
    target_arrays = {
        "world_flat": targets.world_flat,
        "parameter_identifiable": targets.parameter_identifiable,
        "split": split.astype(np.int64),
    }
    manifest = {
        **metadata,
        "input_schema": INPUT_SCHEMA_VERSION,
        "world_schema": "v1",
        "row_count": len(split),
        "input_hash": array_hash(input_arrays),
        "target_hash": array_hash(target_arrays),
    }
    manifest["dataset_hash"] = manifest_hash(manifest)
    root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(root / "inputs.npz", **input_arrays)
    np.savez_compressed(root / "targets.npz", **target_arrays)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


class BeliefDataset(Dataset):
    def __init__(self, root: Path, split: str):
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        if (
            self.manifest.get("input_schema") != INPUT_SCHEMA_VERSION
            or self.manifest.get("world_schema") != "v1"
            or self.manifest.get("dataset_hash") != manifest_hash(self.manifest)
        ):
            raise ValueError("dataset manifest hash or schema mismatch")
        with np.load(root / "inputs.npz", allow_pickle=False) as archive:
            raw_inputs = {name: archive[name] for name in archive.files}
        with np.load(root / "targets.npz", allow_pickle=False) as archive:
            raw_targets = {name: archive[name] for name in archive.files}
        if (
            array_hash(raw_inputs) != self.manifest["input_hash"]
            or array_hash(raw_targets) != self.manifest["target_hash"]
        ):
            raise ValueError("dataset array hash mismatch")
        self.inputs = BeliefInputs.from_mapping(raw_inputs)
        if set(raw_targets) != {"world_flat", "parameter_identifiable", "split"}:
            raise ValueError("unsupported target fields")
        self.targets = WorldTargets(
            raw_targets["world_flat"], raw_targets["parameter_identifiable"]
        )
        assignments = raw_targets["split"]
        if assignments.shape != (len(self.inputs.beliefs),):
            raise ValueError("invalid split shape")
        if not np.isin(assignments, [0, 1, 2]).all():
            raise ValueError("invalid split values")
        if split not in {"train", "validation", "test"}:
            raise ValueError("unknown split")
        self.indices = np.flatnonzero(
            assignments == {"train": 0, "validation": 1, "test": 2}[split]
        )
        if len(self.indices) == 0:
            raise ValueError(f"empty {split} split")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[BeliefInputs, WorldTargets]:
        row = self.indices[index]
        return (
            BeliefInputs(
                self.inputs.beliefs[row : row + 1], self.inputs.history_mask[row : row + 1]
            ),
            WorldTargets(
                self.targets.world_flat[row : row + 1],
                self.targets.parameter_identifiable[row : row + 1],
            ),
        )


def collate_examples(examples: Sequence[tuple[BeliefInputs, WorldTargets]]) -> Model2Batch:
    return Model2Batch(
        torch.from_numpy(np.concatenate([x.beliefs for x, _ in examples])),
        torch.from_numpy(np.concatenate([x.history_mask for x, _ in examples])),
        torch.from_numpy(np.concatenate([y.world_flat for _, y in examples])),
        torch.from_numpy(np.concatenate([y.parameter_identifiable for _, y in examples])),
    )
