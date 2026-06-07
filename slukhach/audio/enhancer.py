"""Forensic-style enhancement: lift quiet background speech, tame the loud foreground.

Unlike source separation (which splits voice from instruments), this strategy works
on *dynamics*: a recording of "someone close + people talking behind a wall" differs
mainly by **level**. Heavy compression squashes the loud near speaker, dynamic
normalization lifts the quiet passages where the distant speech lives, and a
high-pass + spectral denoise clean up rumble and hiss.

The whole chain is expressed as a single ffmpeg filtergraph, so it is fast and needs
no model download.
"""

from __future__ import annotations

import math
from pathlib import Path

from . import io_utils


def _db_to_linear(value_db: float) -> float:
    """Convert a dBFS level into a linear amplitude (0..1+) for ffmpeg parameters."""

    return 10.0 ** (value_db / 20.0)


class EnhancementProcessor:
    """Rebalances a recording so quiet background voices become audible.

    Args:
        highpass_hz: Cutoff for the rumble-removing high-pass filter.
        compressor_threshold_db: Level above which the loud foreground is compressed.
        compressor_ratio: Compression ratio (higher squashes the loud foreground more).
        makeup_gain_db: Output gain applied after compression.
        dynaudnorm_max_gain: Max boost the dynamic normalizer applies to quiet parts.
        noise_reduction_db: Spectral noise-reduction strength (0 disables it).
    """

    def __init__(
        self,
        *,
        highpass_hz: float,
        compressor_threshold_db: float,
        compressor_ratio: float,
        makeup_gain_db: float,
        dynaudnorm_max_gain: float,
        noise_reduction_db: float,
    ) -> None:
        self._highpass_hz = highpass_hz
        self._compressor_threshold_db = compressor_threshold_db
        self._compressor_ratio = compressor_ratio
        self._makeup_gain_db = makeup_gain_db
        self._dynaudnorm_max_gain = dynaudnorm_max_gain
        self._noise_reduction_db = noise_reduction_db

    def process(self, source: Path, workdir: Path) -> Path:
        destination = workdir / "result.ogg"
        return io_utils.apply_filters(source, destination, self._build_filter_chain())

    def _build_filter_chain(self) -> str:
        """Assemble the ordered ffmpeg filtergraph from the configured parameters."""

        threshold_linear = _db_to_linear(self._compressor_threshold_db)
        # dynaudnorm's max gain must be an integer >= 1.
        max_gain = max(1, round(self._dynaudnorm_max_gain))

        stages = [f"highpass=f={self._highpass_hz:g}"]

        if self._noise_reduction_db > 0:
            stages.append(f"afftdn=nr={self._noise_reduction_db:g}:nt=w")

        stages.extend(
            [
                (
                    "acompressor="
                    f"threshold={threshold_linear:.6f}:"
                    f"ratio={self._compressor_ratio:g}:"
                    "attack=5:release=200"
                ),
                f"dynaudnorm=f=150:g=15:p=0.9:m={max_gain}",
                f"volume={self._makeup_gain_db:g}dB",
                "alimiter=limit=0.95",
            ]
        )

        return ",".join(stages)
