"""Audio decoding/encoding helpers built on top of ffmpeg.

Telegram sends voice notes as OGG/Opus and music as MP3/M4A, none of which can be
read by pure-Python libraries without a codec backend. We therefore shell out to
``ffmpeg`` (which Demucs/torchaudio also rely on) to normalize everything into the
WAV PCM that the rest of the pipeline understands.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FFMPEG_BINARY = "ffmpeg"


class FfmpegError(RuntimeError):
    """Raised when ffmpeg is missing or fails to process a file."""


def ensure_ffmpeg_available() -> None:
    """Verify ffmpeg is on PATH, failing fast with a helpful message otherwise."""

    if shutil.which(FFMPEG_BINARY) is None:
        raise FfmpegError(
            "ffmpeg was not found on PATH. Install it (e.g. `winget install Gyan.FFmpeg` "
            "on Windows or `apt install ffmpeg` on Debian/Ubuntu) and restart the bot."
        )


def _run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg with the given arguments, raising :class:`FfmpegError` on failure."""

    command = [FFMPEG_BINARY, "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise FfmpegError(f"ffmpeg failed ({result.returncode}): {result.stderr.strip()}")


def decode_to_wav(source: Path, destination: Path, *, sample_rate: int) -> Path:
    """Decode any ffmpeg-readable file into stereo WAV at ``sample_rate``.

    Args:
        source: Path to the original (compressed) audio file.
        destination: Where the decoded WAV should be written.
        sample_rate: Target sample rate in Hz (should match the separation model).

    Returns:
        The ``destination`` path, for convenient chaining.
    """

    _run_ffmpeg(
        [
            "-i", str(source),
            "-ac", "2",            # force stereo so separation always has 2 channels
            "-ar", str(sample_rate),
            str(destination),
        ]
    )
    return destination


def apply_filters(
    source: Path,
    destination: Path,
    filter_chain: str,
    *,
    codec: str = "libopus",
    bitrate: str = "128k",
) -> Path:
    """Run an ffmpeg audio-filter chain and encode the result.

    Args:
        source: Path to the input audio file.
        destination: Where the filtered/encoded output should be written.
        filter_chain: A comma-separated ffmpeg ``-af`` filtergraph.
        codec: Audio codec for the output (default Opus, for Telegram voice).
        bitrate: Target bitrate.

    Returns:
        The ``destination`` path.
    """

    _run_ffmpeg(
        [
            "-i", str(source),
            "-af", filter_chain,
            "-c:a", codec,
            "-b:a", bitrate,
            str(destination),
        ]
    )
    return destination


def encode_to_ogg(source: Path, destination: Path, *, bitrate: str = "128k") -> Path:
    """Encode a WAV file into OGG/Opus suitable for sending back via Telegram.

    Args:
        source: Path to the processed WAV file.
        destination: Where the encoded OGG should be written.
        bitrate: Target Opus bitrate.

    Returns:
        The ``destination`` path.
    """

    _run_ffmpeg(
        [
            "-i", str(source),
            "-c:a", "libopus",
            "-b:a", bitrate,
            str(destination),
        ]
    )
    return destination
