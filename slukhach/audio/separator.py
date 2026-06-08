
from __future__ import annotations

from dataclasses import dataclass

import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

_VOCALS_STEM = "vocals"


@dataclass(frozen=True)
class SeparatedAudio:

    foreground: torch.Tensor
    background: torch.Tensor
    sample_rate: int


def _resolve_device(preference: str) -> str:

    if preference == "cuda":
        return "cuda"
    if preference == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class VoiceSeparator:

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
