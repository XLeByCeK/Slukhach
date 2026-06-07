"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Immutable, validated application settings.

    Attributes:
        bot_token: Telegram bot token issued by @BotFather.
        vocal_gain_db: Gain applied to the foreground voice (negative = quieter).
        background_gain_db: Gain applied to the background (positive = louder).
        demucs_model: Name of the Demucs model used for separation.
        device: Compute device ("auto", "cpu" or "cuda").
        max_file_size_mb: Maximum accepted input file size in megabytes.
    """

    bot_token: str
    vocal_gain_db: float
    background_gain_db: float
    demucs_model: str
    device: str
    max_file_size_mb: int

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a number, got {raw!r}.") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def load_settings() -> Settings:
    """Build and validate :class:`Settings` from the current environment.

    Raises:
        ConfigError: If a required variable is missing or a value is malformed.
    """

    bot_token = os.getenv("BOT_TOKEN", "").strip()
    if not bot_token:
        raise ConfigError(
            "BOT_TOKEN is not set. Copy .env.example to .env and paste your @BotFather token."
        )

    device = os.getenv("DEVICE", "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigError(f"DEVICE must be one of auto/cpu/cuda, got {device!r}.")

    return Settings(
        bot_token=bot_token,
        vocal_gain_db=_get_float("VOCAL_GAIN_DB", -12.0),
        background_gain_db=_get_float("BACKGROUND_GAIN_DB", 8.0),
        demucs_model=os.getenv("DEMUCS_MODEL", "htdemucs").strip() or "htdemucs",
        device=device,
        max_file_size_mb=_get_int("MAX_FILE_SIZE_MB", 20),
    )
