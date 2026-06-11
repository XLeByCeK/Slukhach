
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ProcessingResult:
    enhance: Path
    separate: Path


class SinglePathProcessor(Protocol):

    def process(self, source: Path, workdir: Path) -> Path:
        ...


class Processor(Protocol):

    def process(self, source: Path, workdir: Path) -> ProcessingResult:
        ...


class DualProcessor:

    def __init__(self, enhancer: SinglePathProcessor, separator: SinglePathProcessor) -> None:
        self._enhancer = enhancer
        self._separator = separator

    def process(self, source: Path, workdir: Path) -> ProcessingResult:
        enhance_path = self._enhancer.process(source, workdir / "enhance")
        separate_path = self._separator.process(source, workdir / "separate")
        return ProcessingResult(enhance=enhance_path, separate=separate_path)
