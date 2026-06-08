
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FFMPEG_BINARY = "ffmpeg"


class FfmpegError(RuntimeError):
    """Raised when ffmpeg is missing or fails to process a file."""

    pass


def ensure_ffmpeg_available() -> None:

    if shutil.which(FFMPEG_BINARY) is None:
        raise FfmpegError(
            "ffmpeg was not found on PATH. Install it (e.g. `winget install Gyan.FFmpeg` "
            "on Windows or `apt install ffmpeg` on Debian/Ubuntu) and restart the bot."
        )


def _run_ffmpeg(args: list[str]) -> None:

    command = [FFMPEG_BINARY, "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise FfmpegError(f"ffmpeg failed ({result.returncode}): {result.stderr.strip()}")


def decode_to_wav(source: Path, destination: Path, *, sample_rate: int) -> Path:

    _run_ffmpeg(
        [
            "-i", str(source),
            "-ac", "2",
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


    _run_ffmpeg(
        [
            "-i", str(source),
            "-c:a", "libopus",
            "-b:a", bitrate,
            str(destination),
        ]
    )
    return destination
