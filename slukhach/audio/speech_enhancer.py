
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from . import io_utils

logger = logging.getLogger(__name__)

EnhancerBackend = Literal["auto", "deepfilternet", "rnnoise", "off"]


class SpeechEnhancer:
    """Neural speech enhancement for separated background audio."""

    def __init__(self, backend: EnhancerBackend = "auto") -> None:
        self._backend = backend
        self._df_model = None
        self._df_state = None
        self._df_enhance = None

        if backend in ("auto", "deepfilternet"):
            self._try_load_deepfilter()

    @property
    def active_backend(self) -> str:
        if self._backend == "off":
            return "off"
        if self._backend == "rnnoise":
            return "rnnoise"
        if self._df_model is not None:
            return "deepfilternet"
        return "rnnoise"

    def _try_load_deepfilter(self) -> None:
        try:
            from df.enhance import enhance, init_df

            self._df_model, self._df_state, _ = init_df()
            self._df_enhance = enhance
            logger.info("DeepFilterNet loaded for speech enhancement.")
        except ImportError:
            logger.info(
                "DeepFilterNet is not available (install deepfilternet on Python <= 3.11). "
                "Falling back to ffmpeg afftdn denoiser."
            )
        except Exception:
            logger.exception("Failed to load DeepFilterNet; falling back to ffmpeg afftdn denoiser.")

    def enhance(self, source_wav: Path, destination_wav: Path) -> Path:
        if self._backend == "off":
            return source_wav

        if self.active_backend == "deepfilternet":
            return self._enhance_deepfilter(source_wav, destination_wav)
        return self._enhance_rnnoise(source_wav, destination_wav)

    def _enhance_deepfilter(self, source_wav: Path, destination_wav: Path) -> Path:
        import numpy as np
        import soundfile as sf
        import torch

        audio, sample_rate = sf.read(source_wav, dtype="float32")
        if audio.ndim == 1:
            audio = audio[:, None]

        enhanced_chunks: list[np.ndarray] = []
        for channel in range(audio.shape[1]):
            channel_audio = torch.from_numpy(audio[:, channel].copy())
            enhanced = self._df_enhance(self._df_model, self._df_state, channel_audio)
            enhanced_chunks.append(enhanced.cpu().numpy())

        enhanced_audio = np.stack(enhanced_chunks, axis=1)
        sf.write(destination_wav, enhanced_audio, sample_rate)
        return destination_wav

    @staticmethod
    def _enhance_rnnoise(source_wav: Path, destination_wav: Path) -> Path:
        # RNNoise via ffmpeg needs a bundled .rnnn model that is often missing on Windows builds.
        # afftdn is a reliable spectral speech denoiser available in standard ffmpeg.
        io_utils.apply_filters(
            source_wav,
            destination_wav,
            "afftdn=nr=12:nf=-25:nt=w",
            codec="pcm_s16le",
        )
        return destination_wav
