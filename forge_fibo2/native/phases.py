# FIBO 2 support for Forge Neo
# One FIBO 2 prompt can carry two text conditionings. When the caption has text_render entries, upstream shows
# Qwen3-VL a picture of each text, but only from sigma 0.9 down; before that the caption is encoded without them
# (ComfyUI's TextEncodeFibo2 schedules the two with start and end percents). A Forge conditioning is one tensor per
# prompt, so both go into it: the rows of each, one after the other, with an extra last feature saying when a row
# applies. The DiT keeps the rows of the step's sigma and drops the marker.
#
# No Forge imports: the CPU tests run it as is.

import torch

ALWAYS, EARLY, LATE = 0.0, 1.0, 2.0
# sigma at ComfyUI's TEXT_IMAGES_START, e^1.15 / (9 + e^1.15) of the way under the shift-1.15 schedule: exactly 0.9
TEXT_IMAGES_SIGMA = 0.9
# Forge's schedule computes that sigma in float; a step landing on it uses the text pictures, as in ComfyUI
TOLERANCE = 1e-6


def pack(states: torch.Tensor, early: torch.Tensor | None = None) -> torch.Tensor:
    """states (T, D) -> (T, D + 1). With early (T', D): its rows first, for sigma above 0.9, then states' for the rest."""
    if early is None:
        return torch.cat([states, states.new_full((states.shape[0], 1), ALWAYS)], dim=-1)
    return torch.cat([torch.cat([early, early.new_full((early.shape[0], 1), EARLY)], dim=-1),
                      torch.cat([states, states.new_full((states.shape[0], 1), LATE)], dim=-1)])


def is_late(sigma: float) -> bool:
    return sigma <= TEXT_IMAGES_SIGMA + TOLERANCE


def split(context: torch.Tensor, sigma: float) -> list[torch.Tensor]:
    """context (B, T, D + 1) -> one (T_i, D) per sample: the rows the step at sigma reads."""
    marker = context[..., -1].round()
    drop = EARLY if is_late(sigma) else LATE
    return [sample[:, :-1][marks != drop] for sample, marks in zip(context, marker)]
