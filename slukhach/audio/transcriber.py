
from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_device(preference: str) -> str:
    if preference == "cuda":
        return "cuda"
    if preference == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Transcriber:
    """Automatic speech recognition via faster-whisper (Whisper large-v3 by default)."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device_preference: str = "auto",
        *,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._model_name = model_name
        self._device = _resolve_device(device_preference)
        self._compute_type = "float16" if self._device == "cuda" else "int8"
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device

    def _ensure_loaded(self) -> None:
        if not self._enabled or self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return
            self._load_model()

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel

        logger.info(
            "Downloading/loading Whisper model %r on %s (first run may take several minutes)...",
            self._model_name,
            self._device,
        )
        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Whisper model ready.")

    def transcribe(self, source: Path) -> str:
        if not self._enabled:
            return ""

        self._ensure_loaded()
        if self._model is None:
            return ""

        segments, _info = self._model.transcribe(
            str(source),
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return text.strip()
