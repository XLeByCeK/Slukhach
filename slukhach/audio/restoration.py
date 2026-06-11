
from __future__ import annotations

from pathlib import Path

from . import io_utils


class SpectralRestoration:
    """Lightweight de-clip and de-reverb cleanup via ffmpeg."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    def apply(self, source_wav: Path, destination_wav: Path) -> Path:
        if not self._enabled:
            return source_wav

        # adeclip: restore clipped peaks; afftdn: reduce reverb tail and residual noise
        filter_chain = "adeclip,afftdn=nf=-20:nt=w"
        io_utils.apply_filters(
            source_wav,
            destination_wav,
            filter_chain,
            codec="pcm_s16le",
        )
        return destination_wav
