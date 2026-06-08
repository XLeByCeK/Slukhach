
from __future__ import annotations

import math
from pathlib import Path

from . import io_utils


def _db_to_linear(value_db: float) -> float:

    return 10.0 ** (value_db / 20.0)


# ffmpeg acompressor accepts threshold in [0.000976563, 1] (~ -60.2 dB .. 0 dB).
_FFMPEG_ACOMPRESSOR_MIN_THRESHOLD = 0.000976563


class EnhancementProcessor:

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

        threshold_linear = max(
            _db_to_linear(self._compressor_threshold_db),
            _FFMPEG_ACOMPRESSOR_MIN_THRESHOLD,
        )
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
