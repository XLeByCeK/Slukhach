from __future__ import annotations

import asyncio
import logging

from aiogram import Bot

from .audio import io_utils
from .audio.enhancer import EnhancementProcessor
from .audio.processor import Processor
from .bot import build_dispatcher
from .config import ConfigError, Settings, load_settings


async def _run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("slukhach")

    settings = load_settings()
    io_utils.ensure_ffmpeg_available()

    processor = _build_processor(settings, logger)

    bot = Bot(token=settings.bot_token)
    dispatcher = build_dispatcher(settings, processor)

    logger.info("Bot started in %r mode. Waiting for messages...", settings.processing_mode)
    await dispatcher.start_polling(bot)


def _build_processor(settings: Settings, logger: logging.Logger) -> Processor:

    if settings.processing_mode == "enhance":
        logger.info("Using dynamics-based enhancement (no model download needed).")
        return EnhancementProcessor(
            highpass_hz=settings.highpass_hz,
            compressor_threshold_db=settings.compressor_threshold_db,
            compressor_ratio=settings.compressor_ratio,
            makeup_gain_db=settings.makeup_gain_db,
            dynaudnorm_max_gain=settings.dynaudnorm_max_gain,
            noise_reduction_db=settings.noise_reduction_db,
        )

    from .audio.pipeline import AudioPipeline
    from .audio.separator import VoiceSeparator

    logger.info("Loading Demucs model %r...", settings.demucs_model)
    separator = VoiceSeparator(settings.demucs_model, settings.device)
    logger.info("Model ready on device: %s", separator.device)
    return AudioPipeline(
        separator,
        foreground_gain_db=settings.vocal_gain_db,
        background_gain_db=settings.background_gain_db,
    )


def main() -> None:
    try:
        asyncio.run(_run())
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
