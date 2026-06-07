"""Application entrypoint: run with ``python -m slukhach``."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot

from .audio import io_utils
from .audio.pipeline import AudioPipeline
from .audio.separator import VoiceSeparator
from .bot import build_dispatcher
from .config import ConfigError, load_settings


async def _run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("slukhach")

    settings = load_settings()
    io_utils.ensure_ffmpeg_available()

    logger.info("Loading Demucs model %r...", settings.demucs_model)
    separator = VoiceSeparator(settings.demucs_model, settings.device)
    logger.info("Model ready on device: %s", separator.device)

    pipeline = AudioPipeline(
        separator,
        foreground_gain_db=settings.vocal_gain_db,
        background_gain_db=settings.background_gain_db,
    )

    bot = Bot(token=settings.bot_token)
    dispatcher = build_dispatcher(settings, pipeline)

    logger.info("Bot started. Waiting for messages...")
    await dispatcher.start_polling(bot)


def main() -> None:
    try:
        asyncio.run(_run())
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
