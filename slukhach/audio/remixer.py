"""Remix separated stems: attenuate the foreground voice, amplify the background."""

from __future__ import annotations

import torch

# Leave a touch of headroom below full scale to avoid inter-sample clipping.
_PEAK_CEILING = 0.97


def _db_to_amplitude(gain_db: float) -> float:
    """Convert a gain in decibels into a linear amplitude multiplier."""

    return float(10.0 ** (gain_db / 20.0))


def remix(
    foreground: torch.Tensor,
    background: torch.Tensor,
    *,
    foreground_gain_db: float,
    background_gain_db: float,
) -> torch.Tensor:
    """Combine the two stems with independent gains and prevent clipping.

    Args:
        foreground: Foreground (voice) stem, shaped (channels, samples).
        background: Background stem, shaped (channels, samples).
        foreground_gain_db: Gain for the voice (negative to make it quieter).
        background_gain_db: Gain for the background (positive to make it louder).

    Returns:
        A mixed waveform shaped (channels, samples), peak-limited to a safe level.
    """

    mixed = (
        foreground * _db_to_amplitude(foreground_gain_db)
        + background * _db_to_amplitude(background_gain_db)
    )

    peak = mixed.abs().max()
    if peak > _PEAK_CEILING:
        mixed = mixed * (_PEAK_CEILING / peak)

    return mixed
