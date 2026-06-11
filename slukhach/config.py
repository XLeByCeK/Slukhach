

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import get_type_hints

from dotenv import load_dotenv

load_dotenv()

DEVICES = ("auto", "cpu", "cuda")
SPEECH_ENHANCERS = ("auto", "deepfilternet", "rnnoise", "off")
BACKGROUND_STEMS = ("non_vocal", "other")

_LOWERCASE_FIELDS = ("device", "whisper_device", "speech_enhancer", "background_stems")


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or invalid."""

    pass


@dataclass(frozen=True)
class Settings:

    bot_token: str

    # General
    max_file_size_mb: int = 20

    # enhance mode (mathematical filters)
    highpass_hz: float = 100.0
    compressor_threshold_db: float = -28.0
    compressor_ratio: float = 8.0
    makeup_gain_db: float = 6.0
    dynaudnorm_max_gain: float = 20.0
    noise_reduction_db: float = 6.0

    # separate mode (Demucs + neural post-processing)
    vocal_gain_db: float = -12.0
    background_gain_db: float = 8.0
    demucs_model: str = "htdemucs"
    device: str = "auto"
    background_stems: str = "non_vocal"
    speech_enhancer: str = "auto"
    restoration_enabled: bool = True

    # ASR (Whisper via faster-whisper)
    whisper_enabled: bool = True
    whisper_model: str = "large-v3"
    whisper_device: str = "auto"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


_PARSERS = {"int": int, "float": float, "str": str, "bool": lambda v: v.lower() in ("1", "true", "yes", "on")}


def _coerce(field_name: str, field_type: type, raw: str):

    type_name = field_type.__name__
    parser = _PARSERS.get(type_name, str)
    try:
        return parser(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {field_name.upper()} must be a valid {type_name}, got {raw!r}."
        ) from exc


def _read_overrides() -> dict:

    type_hints = get_type_hints(Settings)
    overrides: dict = {}
    for field in fields(Settings):
        raw = os.getenv(field.name.upper())
        if raw is None or raw.strip() == "":
            continue
        value = _coerce(field.name, type_hints[field.name], raw.strip())
        if field.name in _LOWERCASE_FIELDS and isinstance(value, str):
            value = value.lower()
        overrides[field.name] = value
    return overrides


def _validate(settings: Settings) -> None:

    if settings.device not in DEVICES:
        raise ConfigError(f"DEVICE must be one of {list(DEVICES)}, got {settings.device!r}.")
    if settings.whisper_device not in DEVICES:
        raise ConfigError(
            f"WHISPER_DEVICE must be one of {list(DEVICES)}, got {settings.whisper_device!r}."
        )
    if settings.speech_enhancer not in SPEECH_ENHANCERS:
        raise ConfigError(
            f"SPEECH_ENHANCER must be one of {list(SPEECH_ENHANCERS)}, got {settings.speech_enhancer!r}."
        )
    if settings.background_stems not in BACKGROUND_STEMS:
        raise ConfigError(
            f"BACKGROUND_STEMS must be one of {list(BACKGROUND_STEMS)}, got {settings.background_stems!r}."
        )


def load_settings() -> Settings:


    overrides = _read_overrides()
    if not overrides.get("bot_token"):
        raise ConfigError(
            "BOT_TOKEN is not set. Copy .env.example to .env and paste your @BotFather token."
        )

    settings = Settings(**overrides)
    _validate(settings)
    return settings
