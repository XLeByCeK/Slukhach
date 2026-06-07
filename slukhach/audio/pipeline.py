"""End-to-end audio pipeline tying decoding, separation, remixing and encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import io_utils, remixer
from .separator import VoiceSeparator


class AudioPipeline:
    """Processes a single audio file: quiet the voice, amplify the background."""

    def __init__(
        self,
        separator: VoiceSeparator,
        *,
        foreground_gain_db: float,
        background_gain_db: float,
    ) -> None:
        self._separator = separator
        self._foreground_gain_db = foreground_gain_db
        self._background_gain_db = background_gain_db

    def process(self, source: Path, workdir: Path) -> Path:
        """Run the full pipeline and return the path to the resulting OGG file.

        Args:
            source: The original file downloaded from Telegram.
            workdir: A scratch directory for intermediate artifacts.

        Returns:
            Path to the encoded OGG/Opus result, ready to send back.
        """

        decoded_wav = io_utils.decode_to_wav(
            source,
            workdir / "decoded.wav",
            sample_rate=self._separator.sample_rate,
        )

        waveform = self._load_waveform(decoded_wav)
        separated = self._separator.separate(waveform)
        mixed = remixer.remix(
            separated.foreground,
            separated.background,
            foreground_gain_db=self._foreground_gain_db,
            background_gain_db=self._background_gain_db,
        )

        processed_wav = self._save_waveform(mixed, separated.sample_rate, workdir / "processed.wav")
        return io_utils.encode_to_ogg(processed_wav, workdir / "result.ogg")

    @staticmethod
    def _load_waveform(wav_path: Path) -> torch.Tensor:
        """Load a WAV file into a float32 tensor shaped (channels, samples)."""

        data, _sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        # soundfile returns (samples, channels); the model wants (channels, samples).
        return torch.from_numpy(data.T.copy())

    @staticmethod
    def _save_waveform(waveform: torch.Tensor, sample_rate: int, wav_path: Path) -> Path:
        """Write a (channels, samples) tensor to a WAV file."""

        data: np.ndarray = waveform.T.contiguous().numpy()
        sf.write(wav_path, data, sample_rate)
        return wav_path
