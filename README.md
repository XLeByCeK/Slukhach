# Slukhach 👂

Telegram-бот, который приглушает основной (передний) голос в записи и усиливает
задний фон — чтобы можно было расслышать, что происходит на фоне (например,
разговор людей позади говорящего).

## Как это работает

```
Аудио из Telegram
      │
      ▼
[ ffmpeg ]  → декодирование в WAV (стерео, нужный sample rate)
      │
      ▼
[ Demucs ]  → разделение на «голос» (vocals) и «фон» (остальные стемы)
      │
      ▼
[ Remixer ] → голос тише (−дБ), фон громче (+дБ) + защита от клиппинга
      │
      ▼
[ ffmpeg ]  → кодирование в OGG/Opus
      │
      ▼
Ответ пользователю
```

Демукс (Demucs от Meta) — модель разделения источников звука. Голос (`vocals`)
трактуется как основной голос, сумма остальных стемов — как фон.

## Стек

- **Python 3.10+**
- **aiogram 3** — асинхронный фреймворк для Telegram-ботов
- **Demucs + PyTorch / torchaudio** — разделение источников звука
- **ffmpeg** — декодирование/кодирование аудио
- **soundfile / numpy** — работа с WAV и сэмплами

## Установка

1. Установите [ffmpeg](https://ffmpeg.org/) и убедитесь, что он в `PATH`:

```bash
# Windows (winget)
winget install Gyan.FFmpeg
# Debian/Ubuntu
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

2. Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

3. Создайте файл `.env` на основе примера и впишите токен от
   [@BotFather](https://t.me/BotFather):

```bash
cp .env.example .env
```

## Запуск

```bash
python -m slukhach
```

При первом запуске Demucs скачает веса модели (~уже несколько сотен МБ).

## Настройка

Все параметры задаются через переменные окружения (см. `.env.example`):

| Переменная           | По умолчанию | Описание                                          |
| -------------------- | ------------ | ------------------------------------------------- |
| `BOT_TOKEN`          | —            | Токен бота от @BotFather (обязательно)            |
| `VOCAL_GAIN_DB`      | `-12`        | Насколько приглушить основной голос (дБ)          |
| `BACKGROUND_GAIN_DB` | `8`          | Насколько усилить фон (дБ)                         |
| `DEMUCS_MODEL`       | `htdemucs`   | Имя модели Demucs                                 |
| `DEVICE`             | `auto`       | `auto` / `cpu` / `cuda`                           |
| `MAX_FILE_SIZE_MB`   | `20`         | Максимальный размер входного файла                |

## Структура проекта

```
slukhach/
├── __main__.py        # точка входа (python -m slukhach)
├── bot.py             # хендлеры aiogram
├── config.py          # загрузка и валидация настроек
└── audio/
    ├── io_utils.py    # декодирование/кодирование через ffmpeg
    ├── separator.py   # обёртка над Demucs
    ├── remixer.py     # микширование стемов с усилением
    └── pipeline.py    # оркестрация всего процесса
```

## Замечания по производительности

- На CPU обработка минутного аудио может занимать заметное время. Для скорости
  используйте GPU (`DEVICE=cuda` при наличии CUDA-сборки PyTorch).
- Лимит размера файла (`MAX_FILE_SIZE_MB`) защищает от слишком долгих задач.
