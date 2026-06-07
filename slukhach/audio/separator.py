"""Source separation: split a recording into a foreground voice and a background.

This wraps the Demucs model. Demucs produces four stems (drums, bass, other,
vocals); we treat ``vocals`` as the dominant foreground voice and the sum of the
remaining stems as the "background" we want to amplify.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

_VOCALS_STEM = "vocals"


@dataclass(frozen=True)
class SeparatedAudio:
    """Result of separation.

    Attributes:
        foreground: Tensor (channels, samples) holding the dominant voice.
        background: Tensor (channels, samples) holding everything else.
        sample_rate: Sample rate of both tensors, in Hz.
    """

    foreground: torch.Tensor
    background: torch.Tensor
    sample_rate: int


def _resolve_device(preference: str) -> str:
    """Map a user preference ("auto"/"cpu"/"cuda") to an actual torch device."""

    if preference == "cuda":
        return "cuda"
    if preference == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class VoiceSeparator:
    """Loads a Demucs model once and reuses it for every request.

    Loading model weights is expensive, so a single long-lived instance is shared
    across all incoming messages.
    """

    def __init__(self, model_name: str, device_preference: str = "auto") -> None:
        self._device = _resolve_device(device_preference)
        self._model = get_model(model_name)
        self._model.to(self._device)
        self._model.eval()

    @property
    def sample_rate(self) -> int:
        return int(self._model.samplerate)

    @property
    def device(self) -> str:
        return self._device

    def separate(self, waveform: torch.Tensor) -> SeparatedAudio:
        """Split a waveform into foreground voice and background.

        Args:
            waveform: Tensor shaped (channels, samples) at :attr:`sample_rate`.

        Returns:
            A :class:`SeparatedAudio` with foreground and background stems.
        """

        # apply_model expects a batch dimension: (batch, channels, samples).
        batch = waveform.unsqueeze(0).to(self._device)
        with torch.no_grad():
            stems = apply_model(self._model, batch, split=True, overlap=0.25)[0]

        sources = list(self._model.sources)
        vocals_index = sources.index(_VOCALS_STEM)

        foreground = stems[vocals_index]
        background = torch.zeros_like(foreground)
        for index, _name in enumerate(sources):
            if index != vocals_index:
                background += stems[index]

        return SeparatedAudio(
            foreground=foreground.cpu(),
            background=background.cpu(),
            sample_rate=self.sample_rate,
        )
