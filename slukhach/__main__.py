from __future__ import annotations

import asyncio
import logging

from aiogram import Bot

from .audio import io_utils
from .audio.enhancer import EnhancementProcessor
from .audio.pipeline import AudioPipeline
from .audio.processor import DualProcessor, Processor
from .audio.restoration import SpectralRestoration
from .audio.separator import VoiceSeparator
from .audio.speech_enhancer import SpeechEnhancer
from .audio.transcriber import Transcriber
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

    logger.info("Bot started. Both enhance and Demucs pipelines are active.")
    await dispatcher.start_polling(bot)


def _build_processor(settings: Settings, logger: logging.Logger) -> Processor:
    logger.info("Using dynamics-based enhancement (no model download needed).")
    enhancer = EnhancementProcessor(
        highpass_hz=settings.highpass_hz,
        compressor_threshold_db=settings.compressor_threshold_db,
        compressor_ratio=settings.compressor_ratio,
        makeup_gain_db=settings.makeup_gain_db,
        dynaudnorm_max_gain=settings.dynaudnorm_max_gain,
        noise_reduction_db=settings.noise_reduction_db,
    )

    logger.info("Loading Demucs model %r...", settings.demucs_model)
    separator = VoiceSeparator(settings.demucs_model, settings.device)
    logger.info("Demucs ready on device: %s", separator.device)

    speech_enhancer = SpeechEnhancer(settings.speech_enhancer)
    logger.info("Speech enhancer backend: %s", speech_enhancer.active_backend)

    restoration = SpectralRestoration(enabled=settings.restoration_enabled)
    demucs_pipeline = AudioPipeline(
        separator,
        speech_enhancer,
        restoration,
        foreground_gain_db=settings.vocal_gain_db,
        background_gain_db=settings.background_gain_db,
        background_stems=settings.background_stems,
    )

    transcriber = Transcriber(
        model_name=settings.whisper_model,
        device_preference=settings.whisper_device,
        enabled=settings.whisper_enabled,
    )
    if settings.whisper_enabled:
        logger.info(
            "Whisper ASR enabled (model=%s, device=%s) — loads on first audio request.",
            settings.whisper_model,
            transcriber.device,
        )

    return DualProcessor(enhancer, demucs_pipeline, transcriber=transcriber)


def main() -> None:
    try:
        asyncio.run(_run())
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
