

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Processor(Protocol):

    def process(self, source: Path, workdir: Path) -> Path:
        ...
