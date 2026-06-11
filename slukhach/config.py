

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from dotenv import load_dotenv

load_dotenv()

DEVICES = ("auto", "cpu", "cuda")

_LOWERCASE_FIELDS = ("device",)


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

    # separate mode (Demucs)
    vocal_gain_db: float = -12.0
    background_gain_db: float = 8.0
    demucs_model: str = "htdemucs"
    device: str = "auto"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


_PARSERS = {"int": int, "float": float, "str": str}


def _coerce(field_name: str, type_name: str, raw: str):

    parser = _PARSERS.get(type_name, str)
    try:
        return parser(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {field_name.upper()} must be a valid {type_name}, got {raw!r}."
        ) from exc


def _read_overrides() -> dict:

    overrides: dict = {}
    for field in fields(Settings):
        raw = os.getenv(field.name.upper())
        if raw is None or raw.strip() == "":
            continue
        value = _coerce(field.name, str(field.type), raw.strip())
        if field.name in _LOWERCASE_FIELDS and isinstance(value, str):
            value = value.lower()
        overrides[field.name] = value
    return overrides


def _validate(settings: Settings) -> None:

    if settings.device not in DEVICES:
        raise ConfigError(f"DEVICE must be one of {list(DEVICES)}, got {settings.device!r}.")


def load_settings() -> Settings:


    overrides = _read_overrides()
    if not overrides.get("bot_token"):
        raise ConfigError(
            "BOT_TOKEN is not set. Copy .env.example to .env and paste your @BotFather token."
        )

    settings = Settings(**overrides)
    _validate(settings)
    return settings
