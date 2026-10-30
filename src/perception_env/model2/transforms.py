"""Belief-only controls with row-stable, seed-replayable transformations."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from .contracts import BeliefInputs

CONTROL_MODES = frozenset({"full", "single_belief", "shuffled_history"})


def transform_tensors(
    beliefs: torch.Tensor, history_mask: torch.Tensor, *, mode: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform validated tensors once, preserving gradients and event triples.

    Shuffle permutations depend on the seed and that row's valid beliefs, never
    batch position, batch size, filenames, targets, or first-order sidecars.
    """
    if mode not in CONTROL_MODES or type(seed) is not int or seed < 0:
        raise ValueError("unknown control mode or invalid control seed")
    if mode == "full":
        return beliefs, history_mask
    output = torch.zeros_like(beliefs)
    mask = history_mask.clone()
    if mode == "single_belief":
        output[:, -1, 0] = beliefs[:, -1, 1]
        output[:, -1, 1] = beliefs[:, -1, 1]
        mask[:] = False
        mask[:, -1] = True
    else:
        for row in range(len(beliefs)):
            indices = torch.nonzero(history_mask[row], as_tuple=True)[0]
            valid = beliefs[row, indices]
            content = valid.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            digest = hashlib.sha256(str(seed).encode() + b":" + content.tobytes()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:16], "little"))
            order = torch.as_tensor(rng.permutation(len(indices)), device=beliefs.device)
            output[row, indices] = valid[order]
    return output, mask


def transform_control(inputs: BeliefInputs, *, mode: str, seed: int) -> BeliefInputs:
    """Return a validated transformed copy for diagnostics and manual controls."""
    if not isinstance(inputs, BeliefInputs):
        raise TypeError("controls accept only BeliefInputs")
    values, mask = transform_tensors(
        torch.tensor(inputs.beliefs), torch.tensor(inputs.history_mask), mode=mode, seed=seed
    )
    return BeliefInputs(values.numpy(), mask.numpy())
