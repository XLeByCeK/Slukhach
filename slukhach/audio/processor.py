"""Common interface for the different audio-processing strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Processor(Protocol):
    """Anything that turns an input audio file into a processed result file."""

    def process(self, source: Path, workdir: Path) -> Path:
        """Process ``source`` using ``workdir`` for scratch files; return the result path."""
        ...
