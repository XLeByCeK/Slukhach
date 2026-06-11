from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Audio, Document, FSInputFile, Message, Voice

from .audio.processor import ProcessingResult, Processor
from .config import Settings

logger = logging.getLogger(__name__)

_WELCOME = (
    "👂 Привет! Я приглушаю основной голос и усиливаю задний фон,\n"
    "чтобы можно было расслышать, что происходит на фоне.\n\n"
    "Пришли голосовое или аудиофайл — верну два варианта обработки:\n"
    "математические фильтры и Demucs, чтобы можно было сравнить."
)

_HELP = (
    "Пришли аудио (голосовое, музыку или документ-аудио).\n"
    "Я верну два результата:\n"
    "• математические фильтры — быстрая обработка без нейросети;\n"
    "• Demucs — разделение голоса и фона через модель.\n\n"
    "Обработка может занять некоторое время — Demucs работает на CPU/GPU."
)


def _extract_media(message: Message) -> Voice | Audio | Document | None:

    if message.voice is not None:
        return message.voice
    if message.audio is not None:
        return message.audio
    if message.document is not None and (message.document.mime_type or "").startswith("audio"):
        return message.document
    return None


class BotHandlers:

    def __init__(self, settings: Settings, processor: Processor) -> None:
        self._settings = settings
        self._processor = processor

    async def on_start(self, message: Message) -> None:
        await message.answer(_WELCOME)

    async def on_help(self, message: Message) -> None:
        await message.answer(_HELP)

    async def on_audio(self, message: Message, bot: Bot) -> None:
        media = _extract_media(message)
        if media is None:
            await message.answer("Это не похоже на аудио. Пришли голосовое или аудиофайл.")
            return

        if media.file_size and media.file_size > self._settings.max_file_size_bytes:
            await message.answer(
                f"Файл слишком большой. Максимум — {self._settings.max_file_size_mb} МБ."
            )
            return

        status = await message.answer("⏳ Обрабатываю... это может занять минуту.")
        try:
            result = await self._handle_media(bot, media)
            await message.answer_voice(
                FSInputFile(result.enhance),
                caption="🔢 Математические фильтры",
            )
            await message.answer_voice(
                FSInputFile(result.separate),
                caption="🎵 Demucs",
            )
        except Exception:
            logger.exception("Failed to process audio")
            await message.answer("⚠️ Не получилось обработать аудио. Попробуй другой файл.")
        finally:
            await status.delete()

    async def _handle_media(self, bot: Bot, media: Voice | Audio | Document) -> ProcessingResult:

        workdir = Path(tempfile.mkdtemp(prefix="slukhach_"))
        source = workdir / "input"

        file = await bot.get_file(media.file_id)
        await bot.download_file(file.file_path, destination=source)

        return await asyncio.to_thread(self._processor.process, source, workdir)


def build_dispatcher(settings: Settings, processor: Processor) -> Dispatcher:

    handlers = BotHandlers(settings, processor)
    router = Router()

    router.message.register(handlers.on_start, CommandStart())
    router.message.register(handlers.on_help, Command("help"))
    router.message.register(
        handlers.on_audio,
        F.voice | F.audio | F.document,
    )

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher
