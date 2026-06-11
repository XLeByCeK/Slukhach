
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ProcessingResult:
    enhance: Path
    separate: Path
    enhance_transcript: str = ""
    separate_transcript: str = ""


class SinglePathProcessor(Protocol):

    def process(self, source: Path, workdir: Path) -> Path:
        ...


class Processor(Protocol):

    def process(self, source: Path, workdir: Path) -> ProcessingResult:
        ...


class DualProcessor:

    def __init__(
        self,
        enhancer: SinglePathProcessor,
        separator: SinglePathProcessor,
        *,
        transcriber: object | None = None,
    ) -> None:
        self._enhancer = enhancer
        self._separator = separator
        self._transcriber = transcriber

    def process(self, source: Path, workdir: Path) -> ProcessingResult:
        with ThreadPoolExecutor(max_workers=2) as pool:
            enhance_future = pool.submit(self._enhancer.process, source, workdir / "enhance")
            separate_future = pool.submit(self._separator.process, source, workdir / "separate")
            enhance_path = enhance_future.result()
            separate_path = separate_future.result()

        enhance_transcript = ""
        separate_transcript = ""
        if self._transcriber is not None:
            enhance_transcript = self._transcriber.transcribe(enhance_path)
            separate_transcript = self._transcriber.transcribe(separate_path)

        return ProcessingResult(
            enhance=enhance_path,
            separate=separate_path,
            enhance_transcript=enhance_transcript,
            separate_transcript=separate_transcript,
        )
