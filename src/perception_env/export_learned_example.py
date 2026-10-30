"""Export one learned end-to-end example for the browser explorer."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .model2.contracts import BeliefInputs
from .model2.decode import decode_prediction
from .model2.train import load_model, predict
from .world import WorldSchema


def _encoded_u8(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype=np.uint8).tobytes()).decode("ascii")


def _hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rollout_payload(
    model1_run: Path,
    checkpoint: Path,
    model: torch.nn.Module,
    example_id: str,
) -> dict[str, object]:
    prediction_path = model1_run / "evaluation" / "records" / example_id / "arrays.npz"
    truth_path = model1_run / "data" / "records" / example_id / "arrays.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        pre = archive["pre_belief"].astype(np.float32)
        post = archive["post_belief"].astype(np.float32)
        target_positions = archive["target_positions"].astype(np.float32)
        target_mask = archive["target_mask"].astype(bool)
    with np.load(truth_path, allow_pickle=False) as archive:
        frames = archive["frames"].astype(np.uint8)
        world_flat = archive["world_flat"].astype(float)

    event_stack = np.stack((pre, post, post - pre), axis=0)
    beliefs = np.repeat(event_stack[None, None], 4, axis=1)
    inputs = BeliefInputs(beliefs, np.ones((1, 4), dtype=bool))
    model_prediction = predict(model, inputs)
    decoded = decode_prediction(model_prediction, WorldSchema.v1())
    probabilities = torch.sigmoid(model_prediction.activity_logits)[0].numpy()
    parameters = model_prediction.normalized_parameters[0].numpy()

    return {
        "example_id": example_id,
        "ground_truth": {
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
            "frame_count": int(frames.shape[0]),
            "frames_base64": _encoded_u8(frames),
            "world_flat": world_flat.tolist(),
            "target_positions": target_positions[target_mask].tolist(),
        },
        "model1": {
            "width": int(pre.shape[1]),
            "height": int(pre.shape[0]),
            "pre_belief_u8": _encoded_u8(np.clip(np.rint(pre * 255), 0, 255)),
            "post_belief_u8": _encoded_u8(np.clip(np.rint(post * 255), 0, 255)),
        },
        "model2": {
            "predicted_world_flat": decoded.flat[0].tolist(),
            "activity_probabilities": probabilities.tolist(),
            "normalized_parameters": parameters.tolist(),
        },
        "provenance": {
            "model1_prediction_sha256": _hash(prediction_path),
            "model1_truth_sha256": _hash(truth_path),
            "model2_checkpoint_sha256": _hash(checkpoint),
            "note": (
                "Model 2 receives only four repeated Model 1 belief events "
                "in this compact example."
            ),
        },
    }


def export_learned_examples(
    model1_run: Path,
    model2_run: Path,
    output: Path,
    example_ids: Sequence[str],
) -> Path:
    """Export an ordered bank of held-out learned rollouts."""
    if not example_ids:
        raise ValueError("at least one example ID is required")
    model1_run = Path(model1_run)
    checkpoint = Path(model2_run) / "training" / "best.pt"
    model = load_model(checkpoint)
    payload = {
        "schema_version": "learned-demo-v2",
        "rollouts": [
            _rollout_payload(model1_run, checkpoint, model, example_id)
            for example_id in example_ids
        ],
    }
    output = Path(output)
    output.write_text(json.dumps(payload, separators=(",", ":")))
    return output


def export_learned_example(
    model1_run: Path,
    model2_run: Path,
    output: Path,
    *,
    example_id: str = "test-000000",
) -> Path:
    """Export one rollout in the original v1 envelope."""
    model1_run = Path(model1_run)
    checkpoint = Path(model2_run) / "training" / "best.pt"
    model = load_model(checkpoint)
    payload = {
        "schema_version": "learned-demo-v1",
        **_rollout_payload(model1_run, checkpoint, model, example_id),
    }
    output = Path(output)
    output.write_text(json.dumps(payload, separators=(",", ":")))
    return output

