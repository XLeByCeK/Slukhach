
from __future__ import annotations

import torch

_PEAK_CEILING = 0.97


def _db_to_amplitude(gain_db: float) -> float:

    return float(10.0 ** (gain_db / 20.0))


def remix(
    foreground: torch.Tensor,
    background: torch.Tensor,
    *,
    foreground_gain_db: float,
    background_gain_db: float,
) -> torch.Tensor:

    mixed = (
        foreground * _db_to_amplitude(foreground_gain_db)
        + background * _db_to_amplitude(background_gain_db)
    )

    peak = mixed.abs().max()
    if peak > _PEAK_CEILING:
        mixed = mixed * (_PEAK_CEILING / peak)

    return mixed
