from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Audio, Document, FSInputFile, Message, Video, VideoNote, Voice

from .audio.processor import ProcessingResult, Processor
from .config import Settings

logger = logging.getLogger(__name__)

_WELCOME = (
    "👂 Привет! Я приглушаю основной голос и усиливаю задний фон,\n"
    "чтобы можно было расслышать, что происходит на фоне.\n\n"
    "Пришли голосовое или аудиофайл — верну два варианта обработки:\n"
    "• математические фильтры (быстро);\n"
    "• Demucs + нейроочистка + расшифровку Whisper."
)

_HELP = (
    "Пришли аудио (голосовое, музыку или документ-аудио).\n"
    "Я верну два результата:\n"
    "• 🔢 Математические фильтры — быстрая обработка без разделения источников;\n"
    "• 🎵 Demucs — разделение голоса и фона, DeepFilterNet/RNNoise, de-clip/de-reverb;\n"
    "• 📝 Расшифровку текста (Whisper) для обоих вариантов.\n\n"
    "Обработка может занять несколько минут — Demucs и Whisper работают на CPU/GPU."
)


_AUDIO_EXTENSIONS = {
    ".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".wma", ".webm", ".mp4", ".mkv",
}


def _extract_media(message: Message) -> Voice | Audio | Document | None:

    if message.voice is not None:
        return message.voice
    if message.audio is not None:
        return message.audio
    if message.video is not None:
        return message.video
    if message.video_note is not None:
        return message.video_note
    if message.document is not None:
        mime = message.document.mime_type or ""
        if mime.startswith("audio") or mime.startswith("video"):
            return message.document
        file_name = message.document.file_name or ""
        if Path(file_name).suffix.lower() in _AUDIO_EXTENSIONS:
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

        status = await message.answer("⏳ Скачиваю файл...")
        try:
            result = await self._handle_media(bot, media, status)
            await message.answer_voice(
                FSInputFile(result.enhance),
                caption="🔢 Математические фильтры",
            )
            if result.enhance_transcript:
                await message.answer(f"📝 Расшифровка (фильтры):\n{result.enhance_transcript}")

            await message.answer_voice(
                FSInputFile(result.separate),
                caption="🎵 Demucs + нейроочистка",
            )
            if result.separate_transcript:
                await message.answer(f"📝 Расшифровка (Demucs):\n{result.separate_transcript}")
        except Exception:
            logger.exception("Failed to process audio")
            await message.answer("⚠️ Не получилось обработать аудио. Попробуй другой файл.")
        finally:
            await status.delete()

    async def _handle_media(
        self,
        bot: Bot,
        media: Voice | Audio | Document | Video | VideoNote,
        status: Message,
    ) -> ProcessingResult:

        workdir = Path(tempfile.mkdtemp(prefix="slukhach_"))
        source = workdir / "input"

        file = await bot.get_file(media.file_id)
        await bot.download_file(file.file_path, destination=source)

        await status.edit_text(
            "⏳ Обрабатываю аудио (фильтры + Demucs + Whisper)...\n"
            "Первый запуск Whisper скачивает модель (~3 ГБ), это может занять 10–20 мин."
        )
        return await asyncio.to_thread(self._processor.process, source, workdir)


def build_dispatcher(settings: Settings, processor: Processor) -> Dispatcher:

    handlers = BotHandlers(settings, processor)
    router = Router()

    router.message.register(handlers.on_start, CommandStart())
    router.message.register(handlers.on_help, Command("help"))
    router.message.register(
        handlers.on_audio,
        F.voice | F.audio | F.video | F.video_note | F.document,
    )

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher
