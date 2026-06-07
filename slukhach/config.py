"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or invalid."""


PROCESSING_MODES = {"enhance", "separate"}


@dataclass(frozen=True)
class Settings:
    """Immutable, validated application settings.

    Attributes:
        bot_token: Telegram bot token issued by @BotFather.
        processing_mode: "enhance" (dynamics/EQ, lifts quiet background speech) or
            "separate" (Demucs voice-vs-instruments separation).
        max_file_size_mb: Maximum accepted input file size in megabytes.

        highpass_hz: Cutoff of the high-pass filter removing low-frequency rumble.
        compressor_threshold_db: Level above which the compressor starts taming
            the loud foreground, in dBFS.
        compressor_ratio: Compression ratio (higher = louder foreground is squashed
            more relative to the quiet background).
        makeup_gain_db: Output gain applied after compression, in dB.
        dynaudnorm_max_gain: Maximum boost the dynamic normalizer may apply to quiet
            passages (1 = off, higher = quiet background lifted more aggressively).
        noise_reduction_db: Spectral noise reduction strength in dB (0 disables it).

        vocal_gain_db: [separate mode] gain applied to the voice stem (negative = quieter).
        background_gain_db: [separate mode] gain applied to the rest (positive = louder).
        demucs_model: [separate mode] name of the Demucs model.
        device: [separate mode] compute device ("auto", "cpu" or "cuda").
    """

    bot_token: str
    processing_mode: str
    max_file_size_mb: int

    highpass_hz: float
    compressor_threshold_db: float
    compressor_ratio: float
    makeup_gain_db: float
    dynaudnorm_max_gain: float
    noise_reduction_db: float

    vocal_gain_db: float
    background_gain_db: float
    demucs_model: str
    device: str

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

    processing_mode = os.getenv("PROCESSING_MODE", "enhance").strip().lower()
    if processing_mode not in PROCESSING_MODES:
        raise ConfigError(
            f"PROCESSING_MODE must be one of {sorted(PROCESSING_MODES)}, got {processing_mode!r}."
        )

    return Settings(
        bot_token=bot_token,
        processing_mode=processing_mode,
        max_file_size_mb=_get_int("MAX_FILE_SIZE_MB", 20),
        highpass_hz=_get_float("HIGHPASS_HZ", 100.0),
        compressor_threshold_db=_get_float("COMPRESSOR_THRESHOLD_DB", -28.0),
        compressor_ratio=_get_float("COMPRESSOR_RATIO", 8.0),
        makeup_gain_db=_get_float("MAKEUP_GAIN_DB", 6.0),
        dynaudnorm_max_gain=_get_float("DYNAUDNORM_MAX_GAIN", 20.0),
        noise_reduction_db=_get_float("NOISE_REDUCTION_DB", 6.0),
        vocal_gain_db=_get_float("VOCAL_GAIN_DB", -12.0),
        background_gain_db=_get_float("BACKGROUND_GAIN_DB", 8.0),
        demucs_model=os.getenv("DEMUCS_MODEL", "htdemucs").strip() or "htdemucs",
        device=device,
    )
