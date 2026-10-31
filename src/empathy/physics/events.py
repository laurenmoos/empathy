"""Observable occlusion-event extraction."""

from __future__ import annotations

from dataclasses import dataclass

from .render import RenderedSequence, _hidden_runs


@dataclass(frozen=True)
class ReappearanceEvent:
    step: int
    x: float
    y: float
    hidden_start: int
    hidden_end: int


def extract_reappearance(sequence: RenderedSequence) -> ReappearanceEvent | None:
    if sequence.rejection_reason is not None:
        return None
    for hidden_start, hidden_end in _hidden_runs(sequence.fully_hidden):
        if hidden_end - hidden_start + 1 < sequence.config.min_hidden_steps:
            continue
        later_visible = sequence.visible[hidden_end + 1 :]
        visible_offsets = later_visible.nonzero()[0]
        if visible_offsets.size == 0:
            continue
        step = hidden_end + 1 + int(visible_offsets[0])
        state = sequence.latent_states[step]
        return ReappearanceEvent(
            step=step,
            x=state.x,
            y=state.y,
            hidden_start=hidden_start,
            hidden_end=hidden_end,
        )
    return None
