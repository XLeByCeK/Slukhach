from __future__ import annotations

import math
from pathlib import Path

from . import io_utils


def _db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 20.0)


_FFMPEG_ACOMPRESSOR_MIN_THRESHOLD = 0.000976563


class EnhancementProcessor:
    def __init__(
        self,
        *,
        highpass_hz: float,
        compressor_threshold_db: float,
        compressor_ratio: float,
        makeup_gain_db: float,
        dynaudnorm_max_gain: float,
        noise_reduction_db: float,
    ) -> None:
        self._highpass_hz = highpass_hz
        self._compressor_threshold_db = compressor_threshold_db
        self._compressor_ratio = compressor_ratio
        self._makeup_gain_db = makeup_gain_db
        self._dynaudnorm_max_gain = dynaudnorm_max_gain
        self._noise_reduction_db = noise_reduction_db

    def process(self, source: Path, workdir: Path) -> Path:
        destination = workdir / "result.ogg"
        return io_utils.apply_filters(source, destination, self._build_filter_chain())

    def _build_filter_chain(self) -> str:
        # 1. Подготовка: Срезаем гул и немного акцентируем детали фона
        stages = [
            f"highpass=f={self._highpass_hz:g}",
            # Поднимаем область разборчивости (2-4 кГц), где живут детали фона
            "equalizer=f=3000:width_type=h:width=1500:g=10"
        ]

        if self._noise_reduction_db > 0:
            stages.append(f"afftdn=nr={self._noise_reduction_db:g}:nt=w")

        # 2. ГЛАВНЫЙ ЭТАП: Восходящая компрессия (Upward Compression)
        # Мы настраиваем кривую так, чтобы:
        # - Тихие звуки (-60дБ) поднимались ОЧЕНЬ сильно (до -25дБ). Это и есть усиление в 2-3 раза и более.
        # - Средние звуки (-30дБ) поднимались умеренно (до -20дБ).
        # - Громкие звуки (-10дБ и выше) ОСТАВАЛИСЬ КАК ЕСТЬ (-10дБ).
        # Точки (вход/выход):
        points = "-80/-80 -60/-25 -30/-18 -10/-10 0/0"
        
        # Чтобы не было эффекта "ныряния", ставим ОЧЕНЬ длинный релиз (2.0 секунды).
        # Это заставит компрессор менять громкость плавно, без рывков.
        stages.append(f"compand=attacks=0.1:decays=2.0:points={points}:soft-knee=10")

        # 3. Медленная нормализация (Slow Leveler)
        # Используем огромный размер окна (f=1000), чтобы громкость не прыгала внутри фраз.
        max_gain = max(1, round(self._dynaudnorm_max_gain / 2))
        stages.append(f"dynaudnorm=f=1000:g=51:p=0.95:m={max_gain}:s=5")

        # 4. Финальный лимитер для защиты от перегрузок
        stages.append(f"volume={self._makeup_gain_db:g}dB")
        stages.append("alimiter=limit=0.98")

        return ",".join(stages)